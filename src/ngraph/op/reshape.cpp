/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
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

#include "ngraph/op/reshape.hpp"
#include "ngraph/function.hpp"

#include <algorithm>

using namespace std;
using namespace ngraph;

op::Reshape::Reshape(const shared_ptr<Node>& arg,
                     const AxisVector& input_order,
                     const Shape& output_shape)
    : RequiresTensorViewArgs("Reshape", {arg})
    , m_input_order(input_order)
    , m_output_shape(output_shape)
{
    auto& input = get_inputs().at(0);
    auto input_shape = input.get_shape();
    auto input_rank = input_shape.size();

    if (m_input_order.size() != input_rank)
    {
        throw ngraph_error("Input axis order for reshape is not a permutation of argument's axes");
    }

    for (size_t i = 0; i < input_rank; i++)
    {
        auto it = find(begin(m_input_order), end(m_input_order), i);
        if (end(m_input_order) == it)
        {
            throw ngraph_error(
                "Input axis order for reshape is not a permutation of argument's axes");
        }
    }

    size_t input_shape_product = 1;
    for (auto i : input_shape)
    {
        input_shape_product *= i;
    }

    size_t output_shape_product = 1;
    for (auto i : m_output_shape)
    {
        output_shape_product *= i;
    }

    if (input_shape_product != output_shape_product)
    {
        throw ngraph_error(
            "Product of output shape dimensions does not match product of argument shape "
            "dimensions for reshape");
    }

    set_value_type_checked(input.get_element_type(), m_output_shape);
}

shared_ptr<Node> op::Reshape::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() != 1)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
    return make_shared<Reshape>(new_args.at(0), m_input_order, m_output_shape);
}

void op::Reshape::generate_adjoints(autodiff::Adjoints& adjoints, const NodeVector& deltas)
{
    auto delta = deltas.at(0);

    auto x_shape = get_inputs().at(0).get_shape();
    auto x_rank = x_shape.size();
    Shape permuted_x_shape(x_rank);
    AxisVector x_input_order(x_rank);
    bool is_permuted = false;
    for (size_t i = 0; i < x_rank; ++i)
    {
        size_t permuted_i = m_input_order[i];
        if (i != permuted_i)
        {
            is_permuted = true;
        }
        permuted_x_shape[i] = x_shape[permuted_i];
        x_input_order[permuted_i] = i;
    }
    AxisVector input_order(m_output_shape.size());
    for (size_t i = 0; i < m_output_shape.size(); i++)
    {
        input_order[i] = i;
    }
    auto reshape = make_shared<op::Reshape>(delta, input_order, permuted_x_shape);
    if (is_permuted)
    {
        reshape = make_shared<op::Reshape>(reshape, x_input_order, x_shape);
    }

    adjoints.add_delta(get_argument(0), reshape);
}
