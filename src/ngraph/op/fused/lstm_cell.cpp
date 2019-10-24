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
#include "ngraph/op/add.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/dot.hpp"
#include "ngraph/op/fused/lstm_cell.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/type/element_type.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::LSTMCell::type_info;

op::LSTMCell::LSTMCell(const Output<Node>& X,
                       const Output<Node>& W,
                       const Output<Node>& R,
                       const Output<Node>& H_t,
                       const Output<Node>& C_t,
                       size_t hidden_size)
    : LSTMCell(X,
               W,
               R,
               H_t,
               C_t,
               hidden_size,
               vector<string>{"sigmoid", "tanh", "tanh"},
               vector<float>{},
               vector<float>{},
               0.f,
               false)
{
}

op::LSTMCell::LSTMCell(const Output<Node>& X,
                       const Output<Node>& W,
                       const Output<Node>& R,
                       const Output<Node>& H_t,
                       const Output<Node>& C_t,
                       size_t hidden_size,
                       const vector<string>& activations,
                       const vector<float>& activation_alpha,
                       const vector<float>& activation_beta,
                       float clip,
                       bool input_forget)
    : FusedOp({X, W, R, H_t, C_t})
    , RNNCellBase(hidden_size, clip, activations, activation_alpha, activation_beta)
    , m_activation_f{get_activation_function(0)}
    , m_activation_g{get_activation_function(1)}
    , m_activation_h{get_activation_function(2)}
    , m_input_forget{input_forget}
{
    add_default_bias_input();
    add_default_peepholes_input();
    constructor_validate_and_infer_types();
}

op::LSTMCell::LSTMCell(const Output<Node>& X,
                       const Output<Node>& W,
                       const Output<Node>& R,
                       const Output<Node>& H_t,
                       const Output<Node>& C_t,
                       size_t hidden_size,
                       const Output<Node>& B,
                       const Output<Node>& P,
                       const vector<string>& activations,
                       const vector<float>& activation_alpha,
                       const vector<float>& activation_beta,
                       float clip,
                       bool input_forget)
    : FusedOp({X, W, R, H_t, C_t, B, P})
    , RNNCellBase(hidden_size, clip, activations, activation_alpha, activation_beta)
    , m_activation_f{get_activation_function(0)}
    , m_activation_g{get_activation_function(1)}
    , m_activation_h{get_activation_function(2)}
    , m_input_forget{input_forget}
{
    constructor_validate_and_infer_types();
}

void op::LSTMCell::pre_validate_and_infer_types()
{
    const auto& x_pshape = get_input_partial_shape(0);
    const auto& w_pshape = get_input_partial_shape(1);
    const auto& r_pshape = get_input_partial_shape(2);
    const auto& ht_pshape = get_input_partial_shape(3);
    const auto& ct_pshape = get_input_partial_shape(4);

    NODE_VALIDATION_CHECK(this,
                          (x_pshape.is_static() || w_pshape.is_static() || r_pshape.is_static() ||
                           ht_pshape.is_static() || ct_pshape.is_static()),
                          "LSTMCell supports only static input tensors.");

    const Shape& x_shape{x_pshape.to_shape()};

    const size_t batch_size = x_shape.at(0);
    const size_t input_size = x_shape.at(1);

    const Shape& w_shape{w_pshape.to_shape()};
    const Shape& r_shape{r_pshape.to_shape()};
    const Shape& ht_shape{ht_pshape.to_shape()};
    const Shape& ct_shape{ct_pshape.to_shape()};

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
    NODE_VALIDATION_CHECK(this,
                          (ct_shape == Shape{batch_size, get_hidden_size()}),
                          "Input tensor C_t must have shape (",
                          batch_size,
                          ", ",
                          get_hidden_size(),
                          "). Actual shape is:",
                          w_shape,
                          ".");

    const auto& b_pshape = get_input_partial_shape(5);
    const auto& p_pshape = get_input_partial_shape(6);

    NODE_VALIDATION_CHECK(this,
                          (b_pshape.is_static() || p_pshape.is_static()),
                          "LSTMCell supports only static input tensors.");

    const Shape& b_shape{b_pshape.to_shape()};
    const Shape& p_shape{p_pshape.to_shape()};

    NODE_VALIDATION_CHECK(this,
                          (b_shape == Shape{2 * s_gates_count * get_hidden_size()}),
                          "Input tensor B must have shape (",
                          8 * get_hidden_size(),
                          "). Actual shape is:",
                          b_shape,
                          ".");

    NODE_VALIDATION_CHECK(this,
                          (p_shape == Shape{s_peepholes_count * get_hidden_size()}),
                          "Input tensor P must have shape (",
                          s_peepholes_count * get_hidden_size(),
                          "). Actual shape is:",
                          p_shape,
                          ".");
}

NodeVector op::LSTMCell::decompose_op() const
{
    // ------ VARIABLE'S NAMES AND ACRONYM DEFINITIONS ------
    // The names used below are analogous to the one used in ONNX documentation.
    //
    // ------ ACRONYMS ------
    // i - input gate
    // o - output gate
    // f - forget gate
    // c - cell gate
    // t - time step (t-1 means previous time step)
    // Wb - W bias vectors for input, output, forget, and cell gates.
    // Rb - R bias vectors for input, output, forget, and cell gates.
    // P  - The peephole weights for input, output and forget gates.
    // ------ VARIABLE NAMES ------
    // X       - The input data tensor. Shape: [batch_size, input_size].
    // W       - The weight matrix for input, output, forget, and cell gates
    //           Shape: [4*hidden_size, input_size]
    // R       - The recurrence weight matrix for input, output, forget, and cell gates.
    //           Shape: [4*hidden_size, hidden_size].
    // H_t     - The hidden state tensor at current time step. Shape: [batch_size, hidden_size].
    // C_t     - The cell state tensor at current time step. Shape: [batch_size, hidden_size].
    // bias    - The sum of biases (weight and recurrence) for input, output, forget, and cell
    //           gates. Shape: [4 * hidden_size]
    // p_[iof] - The peephole weight vector for respectively: input, output, and forget gates.
    //           Each peephole has shape [hidden_size].
    //
    // (.) - Denotes element-wise multiplication.
    // *   - Denotes dot product.
    //
    // ---- Equations ----
    // f, g, h - are activation functions.
    // it = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Pi (.) Ct-1 + Wbi + Rbi)
    // ft = f(Xt*(Wf^T) + Ht-1*(Rf^T) + Pf (.) Ct-1 + Wbf + Rbf)
    // ct = g(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)
    // Ct = ft (.) Ct-1 + it (.) ct
    // ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Wbo + Rbo)
    // Ht = ot (.) h(Ct)
    // --------------------

    Output<Node> X = input_value(0);
    Output<Node> W = input_value(1);
    Output<Node> R = input_value(2);
    Output<Node> H_t = input_value(3);
    Output<Node> C_t = input_value(4);
    Output<Node> bias = get_bias();
    NodeVector p_iof = get_peephole_weights();

    const auto& p_i = p_iof.at(0);
    const auto& p_o = p_iof.at(1);
    const auto& p_f = p_iof.at(2);

    // Xt*(W^T) -- for [iofc] gates.
    auto Xt_W = make_shared<op::Dot>(X, builder::transpose(W));
    // Ht-1*(R^T)  -- for [iofc] gates.
    auto Ht_R = make_shared<op::Dot>(H_t, builder::transpose(R));
    // Xt*(W^T) + Ht-1*(R^T) + Wb + Rb  -- for [iofc] gates.
    auto gates = add(Xt_W, add(Ht_R, bias));

    NodeVector split_gates = builder::split(gates, 4, -1);
    auto i_t = split_gates.at(0);
    auto o_t = split_gates.at(1);
    auto f_t = split_gates.at(2);
    auto c_t = split_gates.at(3);

    // f(Xt*(Wi^T) + Ht-1*(Ri^T) + Pi (.) Ct-1 + Wbi + Rbi)
    i_t = m_activation_f(clip(add(i_t, mul(p_i, C_t))));
    if (m_input_forget)
    {
        // Couple input with forget gate: 1 - i_t
        f_t = sub(op::Constant::create(i_t->get_element_type(),
                                       i_t->get_shape(),
                                       vector<float>(shape_size(i_t->get_shape()), 1.f)),
                  i_t);
    }
    else
    {
        // f(Xt*(Wf^T) + Ht-1*(Rf^T) + Pf (.) Ct-1 + Wbf + Rbf)
        f_t = m_activation_f(clip(add(f_t, mul(p_f, C_t))));
    }
    // ft (.) Ct-1 + it (.) ct
    auto C = add(mul(f_t, C_t), mul(i_t, m_activation_g(clip(c_t))));
    // f(Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Wbo + Rbo)
    o_t = m_activation_f(clip(add(o_t, mul(p_o, C))));
    // ot (.) h(Ct)
    auto H = mul(o_t, m_activation_h(clip(C)));

    return {H, C};
}

Output<Node> op::LSTMCell::get_bias() const
{
    Output<Node> bias;
    // Split B onto Wb an Rb and add them.
    NodeVector b_W_R = builder::split(input_value(5), 2);
    bias = b_W_R.at(0) + b_W_R.at(1);
    return bias;
}

NodeVector op::LSTMCell::get_peephole_weights() const
{
    Output<Node> P;
    P = input_value(6);
    return builder::split(P, s_peepholes_count);
}

void op::LSTMCell::add_default_bias_input()
{
    Output<Node> B =
        op::Constant::create(input(0).get_element_type(),
                             Shape{2 * s_gates_count * get_hidden_size()},
                             vector<float>(2 * s_gates_count * get_hidden_size(), 0.f));
    set_argument(5, B);
}

void op::LSTMCell::add_default_peepholes_input()
{
    Output<Node> P =
        op::Constant::create(input(0).get_element_type(),
                             Shape{s_peepholes_count * get_hidden_size()},
                             vector<float>(s_peepholes_count * get_hidden_size(), 0.f));
    set_argument(6, P);
}

shared_ptr<Node> op::LSTMCell::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    if (new_args.size() == 5)
    {
        return make_shared<LSTMCell>(new_args.at(0),
                                     new_args.at(1),
                                     new_args.at(2),
                                     new_args.at(3),
                                     new_args.at(4),
                                     get_hidden_size(),
                                     get_activations(),
                                     get_activation_alpha(),
                                     get_activation_beta(),
                                     get_clip(),
                                     m_input_forget);
    }
    else if (new_args.size() == 7)
    {
        return make_shared<LSTMCell>(new_args.at(0),
                                     new_args.at(1),
                                     new_args.at(2),
                                     new_args.at(3),
                                     new_args.at(4),
                                     get_hidden_size(),
                                     new_args.at(5),
                                     new_args.at(6),
                                     get_activations(),
                                     get_activation_alpha(),
                                     get_activation_beta(),
                                     get_clip(),
                                     m_input_forget);
    }
    else
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
}
