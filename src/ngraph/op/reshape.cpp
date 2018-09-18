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

#include <algorithm>
#include <iostream>

#include "ngraph/function.hpp"
#include "ngraph/op/reshape.hpp"

using namespace std;
using namespace ngraph;

op::Reshape::Reshape(const shared_ptr<Node>& arg,
                     const AxisVector& input_order,
                     const Shape& output_shape)
    : Op("Reshape", check_single_output_args({arg}))
    , m_input_order(input_order)
    , m_output_shape(output_shape)
{
    constructor_validate_and_infer_types();
}

void op::Reshape::validate_and_infer_types()
{
    auto& input = get_inputs().at(0);
    auto input_shape = input.get_shape();
    auto input_rank = input_shape.size();

    NODE_VALIDATION_ASSERT(this, m_input_order.size() == input_rank)
        << "Input axis order is not a permutation of argument's axis indices (axis order: "
        << m_input_order << ", argument shape: " << input_shape << ").";

    for (size_t i = 0; i < input_rank; i++)
    {
        auto it = find(begin(m_input_order), end(m_input_order), i);
        NODE_VALIDATION_ASSERT(this, it != end(m_input_order))
            << "Input axis order is not a permutation of argument's axis indices (axis order: "
            << m_input_order << ", argument shape: " << input_shape << ").";
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

    NODE_VALIDATION_ASSERT(this, input_shape_product == output_shape_product)
        << "Product of output shape dimensions does not match product of argument shape dimensions "
        << "(output shape: " << m_output_shape << ", argument shape: " << input_shape << ").";

    if (!std::is_sorted(m_input_order.begin(), m_input_order.end()))
    {
        m_is_transpose = true;
    }
    set_output_type(0, input.get_element_type(), m_output_shape);

    // Static value propagation.
    // We will only propagate when reshaping to/from scalars/vectors, and when
    // this op is not a transpose. This has the very useful property of being
    // the identity. :/
    if (!m_is_transpose && input_shape.size() < 2 && m_output_shape.size() < 2 &&
        get_inputs()[0].get_output().has_static_value())
    {
        auto& sv = get_inputs()[0].get_output().get_static_value();

        // This check should be redundant but you never know.
        NODE_VALIDATION_ASSERT(this, m_output_shape.size() == 1 || sv.size() == 1)
            << "Reshaping to a scalar but static value has more than one element "
            << "(input 0 static value: " << sv << ")";
        set_output_static_value(0, sv);
    }
    else
    {
        clear_output_static_value(0);
    }
}

shared_ptr<Node> op::Reshape::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
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
