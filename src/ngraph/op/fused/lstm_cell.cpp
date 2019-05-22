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

#include <algorithm>
#include <cmath>
#include <functional>

#include "ngraph/builder/split.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/dot.hpp"
#include "ngraph/op/fused/lstm_cell.hpp"
#include "ngraph/op/maximum.hpp"
#include "ngraph/op/minimum.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/subtract.hpp"
#include "ngraph/op/util/broadcasting.hpp"
#include "ngraph/op/util/reshape.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/type/element_type.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

// ------------- HELPER FUNCTIONS ---------------------------------------------

static shared_ptr<Node> add(const shared_ptr<Node>& lhs, const shared_ptr<Node>& rhs)
{
    auto args = op::numpy_style_broadcast({lhs, rhs});
    return {make_shared<op::Add>(args.at(0), args.at(1))};
}

static shared_ptr<Node> sub(const shared_ptr<Node>& lhs, const shared_ptr<Node>& rhs)
{
    auto args = op::numpy_style_broadcast({lhs, rhs});
    return {make_shared<op::Subtract>(args.at(0), args.at(1))};
}

static shared_ptr<Node> mul(const shared_ptr<Node>& lhs, const shared_ptr<Node>& rhs)
{
    auto args = op::numpy_style_broadcast({lhs, rhs});
    return {make_shared<op::Multiply>(args.at(0), args.at(1))};
}

static shared_ptr<Node> clip(const shared_ptr<Node>& data, float threshold)
{
    if (threshold == 0.f)
    {
        return data;
    }

    float min_val = -threshold;
    float max_val = threshold;
    size_t size = shape_size(data->get_shape());
    const shared_ptr<Node> min_val_node = op::Constant::create(
        data->get_element_type(), data->get_shape(), vector<float>(size, min_val));
    const shared_ptr<Node> max_val_node = op::Constant::create(
        data->get_element_type(), data->get_shape(), vector<float>(size, max_val));

    return make_shared<op::Minimum>(max_val_node, make_shared<op::Maximum>(data, min_val_node));
}

// ------------- LSTM_CELL ----------------------------------------------------

op::LSTMCell::LSTMCell(const shared_ptr<Node>& X,
                       const shared_ptr<Node>& W,
                       const shared_ptr<Node>& R,
                       const shared_ptr<Node>& H_t,
                       const shared_ptr<Node>& C_t,
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

op::LSTMCell::LSTMCell(const shared_ptr<Node>& X,
                       const shared_ptr<Node>& W,
                       const shared_ptr<Node>& R,
                       const shared_ptr<Node>& H_t,
                       const shared_ptr<Node>& C_t,
                       size_t hidden_size,
                       const vector<string>& activations,
                       const vector<float>& activation_alpha,
                       const vector<float>& activation_beta,
                       float clip,
                       bool input_forget)
    : FusedOp("LSTMCell", {X, W, R, H_t, C_t})
    , RNNCellBase(hidden_size, clip, activations, activation_alpha, activation_beta)
    , m_X{X}
    , m_W{W}
    , m_R{R}
    , m_H_t{H_t}
    , m_C_t{C_t}
    , m_activation_f{get_activation_function(0)}
    , m_activation_g{get_activation_function(1)}
    , m_activation_h{get_activation_function(2)}
    , m_input_forget{input_forget}
{
    // Normally we would split B onto Wb an Rb and add them, however here they are all zeros,
    // thus just initialize bias with appropriate shape and zeros.
    m_bias = ngraph::op::Constant::create(element::f32,
                                          Shape{m_gates_count * get_hidden_size()},
                                          vector<float>(m_gates_count * get_hidden_size(), 0.f));

    const auto& peephole_weights =
        ngraph::op::Constant::create(element::f32,
                                     Shape{m_peepholes_count * get_hidden_size()},
                                     vector<float>(m_peepholes_count * get_hidden_size(), 0.f));
    m_p_iof = builder::split(peephole_weights, m_peepholes_count);
    constructor_validate_and_infer_types();
}

op::LSTMCell::LSTMCell(const shared_ptr<Node>& X,
                       const shared_ptr<Node>& W,
                       const shared_ptr<Node>& R,
                       const shared_ptr<Node>& H_t,
                       const shared_ptr<Node>& C_t,
                       size_t hidden_size,
                       const shared_ptr<Node>& B,
                       const shared_ptr<Node>& P,
                       const vector<string>& activations,
                       const vector<float>& activation_alpha,
                       const vector<float>& activation_beta,
                       float clip,
                       bool input_forget)
    : FusedOp("LSTMCell", {X, W, R, H_t, C_t, B, P})
    , RNNCellBase(hidden_size, clip, activations, activation_alpha, activation_beta)
    , m_X{X}
    , m_W{W}
    , m_R{R}
    , m_H_t{H_t}
    , m_C_t{C_t}
    , m_activation_f{get_activation_function(0)}
    , m_activation_g{get_activation_function(1)}
    , m_activation_h{get_activation_function(2)}
    , m_input_forget{input_forget}
{
    // Split B onto Wb and Rb and add them.
    NODE_VALIDATION_CHECK(this,
                          (B->get_shape() == Shape{2 * m_gates_count * get_hidden_size()}),
                          "Input tensor B must have shape (",
                          8 * get_hidden_size(),
                          "). Actual shape is:",
                          B->get_shape(),
                          ".");

    NodeVector b_W_R = builder::split(B, 2);
    m_bias = b_W_R.at(0) + b_W_R.at(1);

    NODE_VALIDATION_CHECK(this,
                          (P->get_shape() == Shape{m_peepholes_count * get_hidden_size()}),
                          "Input tensor P must have shape (",
                          m_peepholes_count * get_hidden_size(),
                          "). Actual shape is:",
                          P->get_shape(),
                          ".");

    m_p_iof = builder::split(P, m_peepholes_count);
    constructor_validate_and_infer_types();
}

void op::LSTMCell::pre_validate_and_infer_types()
{
    const auto& x_shape = input(0).get_shape();

    const size_t batch_size = x_shape.at(0);
    const size_t input_size = x_shape.at(1);

    const auto& w_shape = input(1).get_shape();
    const auto& r_shape = input(2).get_shape();
    const auto& ht_shape = input(3).get_shape();
    const auto& ct_shape = input(4).get_shape();

    NODE_VALIDATION_CHECK(this,
                          (w_shape == Shape{4 * get_hidden_size(), input_size}),
                          "Input tensor W must have shape (",
                          4 * get_hidden_size(),
                          ", ",
                          input_size,
                          "). Actual shape is:",
                          w_shape,
                          ".");
    NODE_VALIDATION_CHECK(this,
                          (r_shape == Shape{4 * get_hidden_size(), get_hidden_size()}),
                          "Input tensor R must have shape (",
                          4 * get_hidden_size(),
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
    // W  - W parameter weight matrix for input, output, forget, and
    //      cell gates.
    // R  - R recurrence weight matrix for input, output, forget, and
    //      cell gates.
    // Wb - W bias vectors for input, output, forget, and cell gates.
    // Rb - R bias vectors for input, output, forget, and cell gates.
    // ------ VARIABLE NAMES ------
    // p_[iof] - P peephole weight vector for respectively: input, output,
    //           and forget gates.
    // Xt_W    - Input sequence multiplied by weights tensor at current time
    //           step.
    // Ht_R    - Hidden state multiplied by weights tensor at current time step.

    // (.) - Denotes element-wise multiplication.
    // *   - Denotes dot product.

    // ---- Equations ----
    // f, g, h - are activation functions.
    // it = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Pi (.) Ct-1 + Wbi + Rbi)
    // ft = f(Xt*(Wf^T) + Ht-1*(Rf^T) + Pf (.) Ct-1 + Wbf + Rbf)
    // ct = g(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)
    // Ct = ft (.) Ct-1 + it (.) ct
    // ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Wbo + Rbo)
    // Ht = ot (.) h(Ct)
    // --------------------

    const auto& p_i = m_p_iof.at(0);
    const auto& p_o = m_p_iof.at(1);
    const auto& p_f = m_p_iof.at(2);

    // Xt*(W^T) -- for [iofc] gates.
    auto Xt_W = std::make_shared<ngraph::op::Dot>(m_X, ngraph::op::util::transpose(m_W));
    // Ht-1*(R^T)  -- for [iofc] gates.
    auto Ht_R = std::make_shared<ngraph::op::Dot>(m_H_t, ngraph::op::util::transpose(m_R));
    // Xt*(W^T) + Ht-1*(R^T) + Wb + Rb  -- for [iofc] gates.
    auto gates = add(Xt_W, add(Ht_R, m_bias));

    NodeVector split_gates = builder::split(gates, 4, -1);
    auto i_t = split_gates.at(0);
    auto o_t = split_gates.at(1);
    auto f_t = split_gates.at(2);
    auto c_t = split_gates.at(3);

    // f(Xt*(Wi^T) + Ht-1*(Ri^T) + Pi (.) Ct-1 + Wbi + Rbi)
    i_t = m_activation_f(clip(add(i_t, mul(p_i, m_C_t)), get_clip()));
    if (m_input_forget)
    {
        // Couple input with forget gate: 1 - i_t
        f_t =
            sub(ngraph::op::Constant::create(i_t->get_element_type(),
                                             i_t->get_shape(),
                                             std::vector<float>(shape_size(i_t->get_shape()), 1.f)),
                i_t);
    }
    else
    {
        // f(Xt*(Wf^T) + Ht-1*(Rf^T) + Pf (.) Ct-1 + Wbf + Rbf)
        f_t = m_activation_f(clip(add(f_t, mul(p_f, m_C_t)), get_clip()));
    }
    // ft (.) Ct-1 + it (.) ct
    auto C = add(mul(f_t, m_C_t), mul(i_t, m_activation_g(clip(c_t, get_clip()))));
    // f(Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Wbo + Rbo)
    o_t = m_activation_f(clip(add(o_t, mul(p_o, C)), get_clip()));
    // ot (.) h(Ct)
    auto H = mul(o_t, m_activation_h(C));

    return {H, C};
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
