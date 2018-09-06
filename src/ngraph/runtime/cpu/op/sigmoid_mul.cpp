//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
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

#include "sigmoid_mul.hpp"
#include "ngraph/log.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/op/tanh.hpp"
#include "ngraph/runtime/cpu/op/sigmoid.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

ngraph::op::SigmoidMultiply::FunctionType
    op::SigmoidMultiply::identify_node_type(const std::shared_ptr<ngraph::Node>& node)
{
    if (std::dynamic_pointer_cast<ngraph::op::Tanh>(node) != nullptr)
    {
        return ngraph::op::SigmoidMultiply::FunctionType::Tanh;
    }
    else if (std::dynamic_pointer_cast<ngraph::op::Sigmoid>(node) != nullptr)
    {
        return ngraph::op::SigmoidMultiply::FunctionType::Logistic;
    }
    else if (std::dynamic_pointer_cast<ngraph::op::Broadcast>(node) != nullptr)
    {
        return ngraph::op::SigmoidMultiply::FunctionType::Identity;
    }
    else if (std::dynamic_pointer_cast<ngraph::op::Add>(node) != nullptr)
    {
        return ngraph::op::SigmoidMultiply::FunctionType::Identity;
    }
    else
    {
        throw ngraph::ngraph_error("SigmoidMultiply input function type not supported: " +
                                   node->get_name());
    }
}

op::SigmoidMultiply::SigmoidMultiply(shared_ptr<Node> input_0,
                                     shared_ptr<Node> input_1,
                                     const FunctionType input_0_type,
                                     const FunctionType input_1_type)
    : Op("SigmoidMultiply", check_single_output_args({input_0, input_1}))
{
    constructor_validate_and_infer_types();

    if (input_0->get_element_type() != input_1->get_element_type())
    {
        throw ngraph_error("SigmoidMultiply input element type mismatch");
    }
    if (input_0->get_shape() != input_1->get_shape())
    {
        throw ngraph_error("SigmoidMultiply input shape mismatch: " +
                           vector_to_string(input_0->get_shape()) + " != " +
                           vector_to_string(input_1->get_shape()));
    }

    m_input_type[0] = input_0_type;
    m_input_type[1] = input_1_type;

    set_output_type(0, input_0->get_element_type(), input_0->get_shape());
}

shared_ptr<Node> op::SigmoidMultiply::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() != 2)
    {
        throw ngraph_error("SigmoidMultiply incorrect number of new arguments");
    }

    // WARNING: implicitly expecting new args must match the original input function types.
    return make_shared<SigmoidMultiply>(
        new_args.at(0), new_args.at(1), m_input_type[0], m_input_type[1]);
}

void op::SigmoidMultiply::generate_adjoints(autodiff::Adjoints& adjoints, const NodeVector& deltas)
{
    auto delta = deltas.at(0);
    auto input_0 = get_argument(0);
    auto input_1 = get_argument(1);

    auto sigmoid_mul_backprop =
        make_shared<op::SigmoidMultiplyBackprop>(input_0, input_1, delta, m_input_type);

    auto input_0_delta = make_shared<op::GetOutputElement>(sigmoid_mul_backprop, 0);
    auto input_1_delta = make_shared<op::GetOutputElement>(sigmoid_mul_backprop, 1);

    adjoints.add_delta(input_0, input_0_delta);
    adjoints.add_delta(input_1, input_1_delta);
}

op::SigmoidMultiplyBackprop::SigmoidMultiplyBackprop(std::shared_ptr<Node> input_0,
                                                     std::shared_ptr<Node> input_1,
                                                     shared_ptr<Node> delta,
                                                     const std::array<FunctionType, 2>& input_type)
    : Op("SigmoidMultiplyBackprop", check_single_output_args({input_0, input_1, delta}))
    , m_input_type(input_type)
{
    constructor_validate_and_infer_types();

    if (input_0->get_element_type() != input_1->get_element_type())
    {
        throw ngraph_error("Argument element types for SigmoidMultiply backprop do not match");
    }
    if (input_0->get_shape() != input_1->get_shape())
    {
        throw ngraph_error("Argument shapes for SigmoidMultiply backprop do not match");
    }
    if (input_0->get_element_type() != delta->get_element_type())
    {
        throw ngraph_error(
            "Argument and delta element types for SigmoidMultiply backprop do not match");
    }
    if (input_0->get_shape() != delta->get_shape())
    {
        throw ngraph_error("Argument and delta shape for SigmoidMultiply backprop do not match");
    }
    set_output_size(2);
    set_output_type(0, get_input_element_type(0), get_input_shape(0));
    set_output_type(1, get_input_element_type(1), get_input_shape(1));
}

shared_ptr<Node> op::SigmoidMultiplyBackprop::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() != 3)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
    return make_shared<SigmoidMultiplyBackprop>(
        new_args.at(0), new_args.at(1), new_args.at(2), m_input_type);
}
