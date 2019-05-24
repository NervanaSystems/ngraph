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
#include "ngraph/op/fused/rnn_cell.hpp"
#include "ngraph/op/util/reshape.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/type/element_type.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

op::RNNCell::RNNCell(const shared_ptr<Node>& X,
                       const shared_ptr<Node>& W,
                       const shared_ptr<Node>& R,
                       const shared_ptr<Node>& H_t,
                       size_t hidden_size)
    : RNNCell(X,
               W,
               R,
               H_t,
               hidden_size,
               vector<string>{"tanh"},
               vector<float>{},
               vector<float>{},
               0.f)
{
}

op::RNNCell::RNNCell(const shared_ptr<Node>& X,
                       const shared_ptr<Node>& W,
                       const shared_ptr<Node>& R,
                       const shared_ptr<Node>& H_t,
                       size_t hidden_size,
                       const vector<string>& activations,
                       const vector<float>& activation_alpha,
                       const vector<float>& activation_beta,
                       float clip)
    : FusedOp("RNNCell", {X, W, R, H_t})
    , RNNCellBase(hidden_size, clip, activations, activation_alpha, activation_beta)
    , m_X{X}
    , m_W{W}
    , m_R{R}
    , m_H_t{H_t}
    , m_activation_f{get_activation_function(0)}
{
    // Normally we would split B onto Wb an Rb and add them, however here they are all zeros,
    // thus just initialize bias with appropriate shape and zeros.
    m_bias = ngraph::op::Constant::create(element::f32,
                                          Shape{m_gates_count * get_hidden_size()},
                                          vector<float>(m_gates_count * get_hidden_size(), 0.f));

    constructor_validate_and_infer_types();
}

op::RNNCell::RNNCell(const shared_ptr<Node>& X,
                       const shared_ptr<Node>& W,
                       const shared_ptr<Node>& R,
                       const shared_ptr<Node>& H_t,
                       size_t hidden_size,
                       const shared_ptr<Node>& B,
                       const vector<string>& activations,
                       const vector<float>& activation_alpha,
                       const vector<float>& activation_beta,
                       float clip)
    : FusedOp("RNNCell", {X, W, R, H_t, B})
    , RNNCellBase(hidden_size, clip, activations, activation_alpha, activation_beta)
    , m_X{X}
    , m_W{W}
    , m_R{R}
    , m_H_t{H_t}
    , m_activation_f{get_activation_function(0)}
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

    constructor_validate_and_infer_types();
}

void op::RNNCell::pre_validate_and_infer_types()
{
    const auto& x_shape = input(0).get_shape();

    const size_t batch_size = x_shape.at(0);
    const size_t input_size = x_shape.at(1);

    const auto& w_shape = input(1).get_shape();
    const auto& r_shape = input(2).get_shape();
    const auto& ht_shape = input(3).get_shape();

    NODE_VALIDATION_CHECK(this,
                          (w_shape == Shape{get_hidden_size(), input_size}),
                          "Input tensor W must have shape (",
                          get_hidden_size(),
                          ", ",
                          input_size,
                          "). Actual shape is:",
                          w_shape,
                          ".");
    NODE_VALIDATION_CHECK(this,
                          (r_shape == Shape{get_hidden_size(), get_hidden_size()}),
                          "Input tensor R must have shape (",
                          get_hidden_size(),
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
}

NodeVector op::RNNCell::decompose_op() const
{
    // ------ VARIABLE'S NAMES AND ACRONYM DEFINITIONS ------
    // The names used below are analogous to the one used in ONNX documentation.
    //
    // ------ ACRONYMS ------
    // i_t - input gate at current time step
    // t - time step (t-1 means previous time step)
    // W  - W parameter weight matrix for input, output, forget, and
    //      cell gates.
    // R  - R recurrence weight matrix for input, output, forget, and
    //      cell gates.
    // Wb - W bias vectors for input, output, forget, and cell gates.
    // Rb - R bias vectors for input, output, forget, and cell gates.
    // ------ VARIABLE NAMES ------
    // Xt_W    - Input sequence multiplied by weights tensor at current time
    //           step.
    // Ht_R    - Hidden state multiplied by weights tensor at current time step.

    // (.) - Denotes element-wise multiplication.
    // *   - Denotes dot product.

    // ---- Equations ----
    // f - is activation functions.
    // Ht = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Wbi + Rbi)
    // --------------------

    // Xt*(W^T)
    auto Xt_W = std::make_shared<ngraph::op::Dot>(m_X, ngraph::op::util::transpose(m_W));
    // Ht-1*(R^T)
    auto Ht_R = std::make_shared<ngraph::op::Dot>(m_H_t, ngraph::op::util::transpose(m_R));
    // Xt*(W^T) + Ht-1*(R^T) + Wb + Rb
    auto i_t = add(Xt_W, add(Ht_R, m_bias));

    // f(Xt*(Wi^T) + Ht-1*(Ri^T) + Wbi + Rbi)
    i_t = m_activation_f(clip(i_t));

    return {i_t};
}

shared_ptr<Node> op::RNNCell::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    if (new_args.size() == 4)
    {
        return make_shared<RNNCell>(new_args.at(0),
                                     new_args.at(1),
                                     new_args.at(2),
                                     new_args.at(3),
                                     get_hidden_size(),
                                     get_activations(),
                                     get_activation_alpha(),
                                     get_activation_beta(),
                                     get_clip());
    }
    else if (new_args.size() == 5)
    {
        return make_shared<RNNCell>(new_args.at(0),
                                     new_args.at(1),
                                     new_args.at(2),
                                     new_args.at(3),
                                     get_hidden_size(),
                                     new_args.at(4),
                                     get_activations(),
                                     get_activation_alpha(),
                                     get_activation_beta(),
                                     get_clip());
    }
    else
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
}
