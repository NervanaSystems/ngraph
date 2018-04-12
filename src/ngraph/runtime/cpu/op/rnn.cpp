/*******************************************************************************
* Copyright 2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "ngraph/runtime/cpu/op/rnn.hpp"
#include "ngraph/log.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

shared_ptr<Node> op::LSTM::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() != 6)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }

    return make_shared<LSTM>(new_args.at(0),
                             new_args.at(1),
                             new_args.at(2),
                             new_args.at(3),
                             new_args.at(4),
                             new_args.at(5));
}

op::LSTM::LSTM(std::shared_ptr<Node> param1_1,
               std::shared_ptr<Node> param1_2,
               std::shared_ptr<Node> param2_1,
               std::shared_ptr<Node> param2_2,
               std::shared_ptr<Node> bias1,
               std::shared_ptr<Node> bias2)
    : RequiresTensorViewArgs("LSTM", {param1_1, param1_2, param2_1, param2_2, bias1, bias2})
    , m_shape_input(param1_1->get_shape())
{
    add_output(param1_1->get_element_type(), m_shape_input);
    add_output(param1_1->get_element_type(), param1_1->get_shape());
}

// op::LSTMBackprop::LSTMBackprop(shared_ptr<Node> arg, shared_ptr<Node> delta)
//     : RequiresTensorViewArgs("LSTMBackprop", {arg, delta})
// {
//     if (arg->get_element_type() != delta->get_element_type())
//     {
//         throw ngraph_error("Argument and delta element types for LSTM backprop do not match");
//     }
//     if (arg->get_shape() != delta->get_shape())
//     {
//         throw ngraph_error("Argument and delta shape for LSTM backprop do not match");
//     }
//     set_value_type_checked(delta->get_element_type(), delta->get_shape());
// }

// shared_ptr<Node> op::LSTMBackprop::copy_with_new_args(const NodeVector& new_args) const
// {
//     if (new_args.size() != 2)
//     {
//         throw ngraph_error("Incorrect number of new arguments");
//     }
//     return make_shared<LSTMBackprop>(new_args.at(0), new_args.at(1));
// }

// void op::LSTM::generate_adjoints(autodiff::Adjoints& adjoints, const shared_ptr<Node>& delta)
// {
//     auto backprop = make_shared<op::LSTMBackprop>(get_input_op(0), delta);
//     adjoints.add_delta(get_input_op(0), backprop);
// }
