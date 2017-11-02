// ----------------------------------------------------------------------------
// Copyright 2017 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------

#include "ngraph/ops/reshape.hpp"
#include <algorithm>
#include <numeric>
#include "ngraph/function.hpp"

using namespace std;
using namespace ngraph::op;

ngraph::AxisVector Reshape::get_default_order(size_t size)
{
    AxisVector iorder(size);
    std::iota(iorder.begin(), iorder.end(), 0);
    return iorder;
};

void Reshape::propagate_types()
{
    if (m_arguments.size() != 1)
    {
        throw ngraph_error("Wrong number of arguments.");
    }

    auto arg_type = m_arguments.at(0)->get_value_type();
    if (nullptr == arg_type)
    {
        throw ngraph_error("Argument to reshape is missing type.");
    }
    auto arg_tensor_view_type = dynamic_pointer_cast<const TensorViewType>(arg_type);
    if (nullptr == arg_tensor_view_type)
    {
        throw ngraph_error("Argument to reshape is not a tensor view");
    }

    auto arg_shape = arg_tensor_view_type->get_shape();
    auto arg_rank = arg_shape.size();

    if (m_input_order.size() != arg_rank)
    {
        throw ngraph_error("Input axis order for reshape is not a permutation of argument's axes");
    }

    for (size_t i = 0; i < arg_rank; i++)
    {
        auto it = std::find(std::begin(m_input_order), std::end(m_input_order), i);
        if (std::end(m_input_order) == it)
        {
            throw ngraph_error(
                "Input axis order for reshape is not a permutation of argument's axes");
        }
    }

    size_t arg_shape_product = 1;
    for (auto i : arg_shape)
    {
        arg_shape_product *= i;
    }

    size_t output_shape_product = 1;
    for (auto i : m_output_shape)
    {
        output_shape_product *= i;
    }

    if (arg_shape_product != output_shape_product)
    {
        throw ngraph_error(
            "Product of output shape dimensions does not match product of argument shape "
            "dimensions for reshape");
    }

    set_value_type_checked(
        make_shared<TensorViewType>(arg_tensor_view_type->get_element_type(), m_output_shape));
}

void ngraph::op::Reshape::generate_adjoints(autodiff::Adjoints& adjoints,
                                            const std::shared_ptr<Node>& delta)
{
    auto x = m_arguments[0];
    auto x_type = x->get_value_type();
    auto x_tensor_view_type = dynamic_pointer_cast<const TensorViewType>(x_type);
    if (nullptr == x_type)
    {
        throw ngraph_error("Argument to reshape is not a tensor view");
    }
    auto x_shape = x_tensor_view_type->get_shape();
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

    adjoints.add_delta(x, reshape);
}
