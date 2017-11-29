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

#include <cassert>
#include <memory>

#include "ngraph/ops/concatenate.hpp"
#include "ngraph/ops/slice.hpp"

using namespace std;
using namespace ngraph;

op::Concat::Concat(const Nodes& args, size_t concatenation_axis)
    : RequiresTensorViewArgs("Concat", args)
    , m_concatenation_axis(concatenation_axis)
{
    if (m_arguments.size() < 1)
    {
        throw ngraph_error("At least one argument required");
    }

    auto arg0_tensor_view_type = get_inputs().at(0).get_tensor_view_type();
    auto arg0_shape = arg0_tensor_view_type->get_shape();
    if (m_concatenation_axis >= arg0_shape.size())
    {
        throw ngraph_error("Concatenation axis is out of bounds");
    }

    size_t concatenation_axis_length = arg0_shape.at(m_concatenation_axis);
    auto& arg0_element_type = arg0_tensor_view_type->get_element_type();

    for (auto i = 1; i < get_inputs().size(); i++)
    {
        auto argi_tensor_view_type = get_inputs().at(i).get_tensor_view_type();
        auto argi_shape = argi_tensor_view_type->get_shape();
        if (argi_shape.size() != arg0_shape.size())
        {
            throw ngraph_error("Arguments to concat do not have same rank");
        }

        if (argi_tensor_view_type->get_element_type() != arg0_element_type)
        {
            throw ngraph_error("Argument element types do not match");
        }

        for (auto j = 0; j < argi_shape.size(); j++)
        {
            if (j != m_concatenation_axis && arg0_shape.at(j) != argi_shape.at(j))
            {
                throw ngraph_error(
                    "Arguments to concat do not have same dimension on a non-concatenation axis");
            }
            else if (j == m_concatenation_axis)
            {
                concatenation_axis_length += argi_shape.at(j);
            }
        }
    }
    vector<size_t> concatenated_shape = arg0_shape;
    concatenated_shape.at(m_concatenation_axis) = concatenation_axis_length;

    set_value_type_checked(make_shared<TensorViewType>(arg0_element_type, concatenated_shape));
}

void op::Concat::generate_adjoints(autodiff::Adjoints& adjoints, const std::shared_ptr<Node>& delta)
{
    auto value_type = get_value_type();
    auto tensor_view_type = std::dynamic_pointer_cast<const TensorViewType>(value_type);

    assert(nullptr != tensor_view_type);

    auto concat_result_shape = tensor_view_type->get_shape();

    Coordinate arg_delta_slice_lower = Coordinate(concat_result_shape.size(), 0);
    Coordinate arg_delta_slice_upper = concat_result_shape;
    Coordinate arg_delta_slice_step = Coordinate(concat_result_shape.size(), 1);

    size_t pos = 0;

    for (auto arg : m_arguments)
    {
        auto arg_value_type = arg->get_value_type();
        auto arg_tensor_view_type = std::dynamic_pointer_cast<const TensorViewType>(arg_value_type);
        assert(nullptr != arg_tensor_view_type);
        auto arg_shape = arg_tensor_view_type->get_shape();

        auto slice_width = arg_shape[m_concatenation_axis];

        size_t next_pos = pos + slice_width;

        arg_delta_slice_lower[m_concatenation_axis] = pos;
        arg_delta_slice_upper[m_concatenation_axis] = next_pos;

        adjoints.add_delta(
            arg,
            make_shared<op::Slice>(
                delta, arg_delta_slice_lower, arg_delta_slice_upper, arg_delta_slice_step));

        pos = next_pos;
    }
}
