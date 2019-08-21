//*****************************************************************************
// Copyright 2017-2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <cmath>
#include <functional>

#include "ngraph/builder/reshape.hpp"
#include "ngraph/builder/split.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/dot.hpp"
#include "ngraph/op/fused/gru_cell.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/type/element_type.hpp"

using namespace std;
using namespace ngraph;

const string op::GRUCell::type_name{"GRUCell"};

op::GRUCell::GRUCell(const Output<Node>& X,
                     const Output<Node>& W,
                     const Output<Node>& R,
                     const Output<Node>& H_t,
                     size_t hidden_size)
    : GRUCell(X,
              W,
              R,
              H_t,
              hidden_size,
              vector<string>{"sigmoid", "tanh"},
              vector<float>{},
              vector<float>{},
              0.f,
              false)
{
}

op::GRUCell::GRUCell(const Output<Node>& X,
                     const Output<Node>& W,
                     const Output<Node>& R,
                     const Output<Node>& H_t,
                     size_t hidden_size,
                     const vector<string>& activations,
                     const vector<float>& activation_alpha,
                     const vector<float>& activation_beta,
                     float clip,
                     bool linear_before_reset)
    : FusedOp({X, W, R, H_t})
    , RNNCellBase(hidden_size, clip, activations, activation_alpha, activation_beta)
    , m_activation_f{get_activation_function(0)}
    , m_activation_g{get_activation_function(1)}
    , m_linear_before_reset{linear_before_reset}
{
    add_default_bias_input();
    constructor_validate_and_infer_types();
}

op::GRUCell::GRUCell(const Output<Node>& X,
                     const Output<Node>& W,
                     const Output<Node>& R,
                     const Output<Node>& H_t,
                     size_t hidden_size,
                     const Output<Node>& B,
                     const vector<string>& activations,
                     const vector<float>& activation_alpha,
                     const vector<float>& activation_beta,
                     float clip,
                     bool linear_before_reset)
    : FusedOp({X, W, R, H_t, B})
    , RNNCellBase(hidden_size, clip, activations, activation_alpha, activation_beta)
    , m_activation_f{get_activation_function(0)}
    , m_activation_g{get_activation_function(1)}
    , m_linear_before_reset{linear_before_reset}
{
    constructor_validate_and_infer_types();
}

void op::GRUCell::pre_validate_and_infer_types()
{
    const auto& x_pshape = get_input_partial_shape(0);
    const auto& w_pshape = get_input_partial_shape(1);
    const auto& r_pshape = get_input_partial_shape(2);
    const auto& ht_pshape = get_input_partial_shape(3);

    NODE_VALIDATION_CHECK(this,
                          (x_pshape.is_static() || w_pshape.is_static() || r_pshape.is_static() ||
                           ht_pshape.is_static()),
                          "GRUCell supports only static input tensors.");

    const Shape& x_shape{x_pshape.to_shape()};

    const size_t batch_size = x_shape.at(0);
    const size_t input_size = x_shape.at(1);

    const Shape& w_shape{w_pshape.to_shape()};
    const Shape& r_shape{r_pshape.to_shape()};
    const Shape& ht_shape{ht_pshape.to_shape()};

    NODE_VALIDATION_CHECK(this,
                          (w_shape == Shape{s_gates_count * get_hidden_size(), input_size}),
                          "Input tensor W must have shape (",
                          s_gates_count * get_hidden_size(),
                          ", ",
                          input_size,
                          "). Actual shape is:",
                          w_shape,
                          ".");
    NODE_VALIDATION_CHECK(this,
                          (r_shape == Shape{s_gates_count * get_hidden_size(), get_hidden_size()}),
                          "Input tensor R must have shape (",
                          s_gates_count * get_hidden_size(),
                          ", ",
                          get_hidden_size(),
                          "). Actual shape is:",
                          w_shape,
                          ".");
    NODE_VALIDATION_CHECK(this,
                          (ht_shape == Shape{batch_size, get_hidden_size()}),
                          "Input tensor H_t must have shape (",
                          batch_size,
                          ", ",
                          get_hidden_size(),
                          "). Actual shape is:",
                          w_shape,
                          ".");

    const auto& b_pshape = get_input_partial_shape(4);

    NODE_VALIDATION_CHECK(
        this, b_pshape.is_static(), "GRUCell supports only static input tensors.");

    const Shape& b_shape{b_pshape.to_shape()};

    NODE_VALIDATION_CHECK(this,
                          (b_shape == Shape{2 * s_gates_count * get_hidden_size()}),
                          "Input tensor B must have shape (",
                          2 * s_gates_count * get_hidden_size(),
                          "). Actual shape is:",
                          b_shape,
                          ".");
}

NodeVector op::GRUCell::decompose_op() const
{
    // ------ VARIABLE'S NAMES AND ACRONYM DEFINITIONS ------
    // The names used below are analogous to the one used in ONNX documentation.
    //
    // ------ ACRONYMS ------
    // z_t - update gate at current time step
    // r_t - reset gate at current time step
    // h_t - hidden gate at current time step
    // t - time step (t-1 means previous time step)
    // X        The input data tensor. Shape: [batch_size, input_size].
    // W[zrh] - The weight tensor for update, reset and hidden gates.
    //          Shape: [gates_count * hidden_size, input_size].
    // R[zrh] - The recurrence weight tensor for update, reset and hidden gates.
    //          Shape: [gates_count * hidden_size, hidden_size].
    // H_t    - The hidden state tensor at current time step. Shape: [batch_size, hidden_size].
    // B      - The bias tensor for the gates. Shape: [2 * gates_count * hidden_size]
    //          Concatenation of `[Wb[zrh], Rb[zrh]]`.
    // Wb[zrh] - W bias vectors for update, reset and hidden gates.
    // Rb[zrh] - R bias vectors for update, reset and hidden gates.

    // (.) - Denotes element-wise multiplication.
    // *   - Denotes dot product.

    // ---- Equations ----
    // f, g  - are activation functions
    // zt = f(Xt*(Wz^T) + Ht-1*(Rz^T) + Wbz + Rbz)
    // rt = f(Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr)
    // ht = g(Xt*(Wh^T) + (rt (.) Ht-1)*(Rh^T) + Rbh + Wbh) # when linear_before_reset := false
    //                                                      # (default)
    // ht = g(Xt*(Wh^T) + (rt (.) (Ht-1*(Rh^T) + Rbh)) + Wbh) # when linear_before_reset := true
    // Ht = (1 - zt) (.) ht + zt (.) Ht-1
    // -------------------

    Output<Node> X = input_value(0);
    Output<Node> W = input_value(1);
    Output<Node> R = input_value(2);
    Output<Node> H_t = input_value(3);
    Output<Node> B = input_value(4);

    // Get W and R biases separately.
    NodeVector b_W_R = builder::split(B, 2);
    // Each tensor has shape: [gates_count * hidden_size]
    const auto& Wb = b_W_R.at(0);
    const auto& Rb = b_W_R.at(1);

    // Split W bias into zr and h gates.
    NodeVector Wb_zr_h =
        builder::split(Wb, vector<size_t>{2 * get_hidden_size(), get_hidden_size()});
    // Tensor shape: [2 * hidden_size]
    const auto& Wb_zr = Wb_zr_h.at(0);
    // Tensor shape: [hidden_size]
    const auto& Wb_h = Wb_zr_h.at(1);

    // Split R bias into zr and h gates.
    NodeVector Rb_zr_h =
        builder::split(Rb, vector<size_t>{2 * get_hidden_size(), get_hidden_size()});
    // Tensor shape: [2 * hidden_size]
    const auto& Rb_zr = Rb_zr_h.at(0);
    // Tensor shape: [hidden_size]
    const auto& Rb_h = Rb_zr_h.at(1);

    // Split R weights into zr and h gates.
    NodeVector R_zr_h = builder::split(R, vector<size_t>{2 * get_hidden_size(), get_hidden_size()});
    // Tensor shape: [2 * hidden_size, hidden_size]
    const auto& R_zr = R_zr_h.at(0);
    // Tensor shape: [hidden_size, hidden_size]
    const auto& R_h = R_zr_h.at(1);

    // Xt*(W^T)
    auto Xt_W = make_shared<op::Dot>(X, builder::transpose(W));
    // Split Xt_W into zr and h gates.
    NodeVector Xt_W_zr_h =
        builder::split(Xt_W, vector<size_t>{2 * get_hidden_size(), get_hidden_size()}, 1);
    // Tensor shape: [batch_size, 2 * hidden_size]
    const auto& Xt_W_zr = Xt_W_zr_h.at(0);
    // Tensor shape: [batch_size, hidden_size]
    const auto& Xt_W_h = Xt_W_zr_h.at(1);

    // Ht-1*(R^T) for update and reset gates. Tensor shape: [batch_size, 2 * hidden_size]
    auto Ht_R_zr = make_shared<op::Dot>(H_t, builder::transpose(R_zr));
    // f(Xt*(W^T) + Ht-1*(R^T) + Wb + Rb) for update and reset gates.
    // Tensor shape: [batch_size, 2 * hidden_size]
    auto zr_t = m_activation_f(clip(add(Xt_W_zr, add(Ht_R_zr, add(Wb_zr, Rb_zr)))));
    // Split into update and reset gates.
    NodeVector zr_t_gates = builder::split(zr_t, 2, 1);
    // Tensor shape: [batch_size, hidden_size]
    const auto& z_t = zr_t_gates.at(0);
    const auto& r_t = zr_t_gates.at(1);

    Output<Node> h_t;

    if (m_linear_before_reset)
    {
        // ht = g(Xt*(Wh^T) + (rt (.) (Ht-1*(Rh^T) + Rbh)) + Wbh)
        auto Ht_Rh_Rb = add(make_shared<op::Dot>(H_t, builder::transpose(R_h)), Rb_h);
        h_t = m_activation_g(clip(add(Xt_W_h, add(mul(r_t, Ht_Rh_Rb), Wb_h))));
    }
    else
    {
        // ht = g(Xt*(Wh^T) + (rt (.) Ht-1)*(Rh^T) + Rbh + Wbh)
        auto rt_Ht = mul(r_t, H_t);
        auto rt_Ht_Rh = make_shared<op::Dot>(rt_Ht, builder::transpose(R_h));
        // Tensor shape: [batch_size, hidden_size]
        h_t = m_activation_g(clip(add(Xt_W_h, add(rt_Ht_Rh, add(Rb_h, Wb_h)))));
    }

    auto one = op::Constant::create(z_t->get_element_type(),
                                    z_t->get_shape(),
                                    vector<float>(shape_size(z_t->get_shape()), 1.f));

    // Ht = (1 - zt) (.) ht + zt (.) Ht-1
    H_t = add(mul(sub(one, z_t), h_t), mul(z_t, H_t));

    return {H_t.get_node_shared_ptr()};
}

void op::GRUCell::add_default_bias_input()
{
    Output<Node> B =
        op::Constant::create(input(0).get_element_type(),
                             Shape{2 * s_gates_count * get_hidden_size()},
                             vector<float>(2 * s_gates_count * get_hidden_size(), 0.f));
    set_argument(4, B);
}

shared_ptr<Node> op::GRUCell::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    if (new_args.size() == 4)
    {
        return make_shared<GRUCell>(new_args.at(0),
                                    new_args.at(1),
                                    new_args.at(2),
                                    new_args.at(3),
                                    get_hidden_size(),
                                    get_activations(),
                                    get_activation_alpha(),
                                    get_activation_beta(),
                                    get_clip(),
                                    m_linear_before_reset);
    }
    else if (new_args.size() == 5)
    {
        return make_shared<GRUCell>(new_args.at(0),
                                    new_args.at(1),
                                    new_args.at(2),
                                    new_args.at(3),
                                    get_hidden_size(),
                                    new_args.at(4),
                                    get_activations(),
                                    get_activation_alpha(),
                                    get_activation_beta(),
                                    get_clip(),
                                    m_linear_before_reset);
    }
    else
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
}
