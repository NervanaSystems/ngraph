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

#include "ngraph/op/one_hot.hpp"
#include "ngraph/op/sum.hpp"

using namespace std;
using namespace ngraph;

op::OneHot::OneHot(const shared_ptr<Node>& arg, const PartialShape& shape, size_t one_hot_axis)
    : Op("OneHot", check_single_output_args({arg}))
    , m_shape(shape)
    , m_one_hot_axis(one_hot_axis)
{
    constructor_validate_and_infer_types();
}

void op::OneHot::validate_and_infer_types()
{
    element::Type arg_et = get_input_element_type(0);
    PartialShape arg_shape = get_input_partial_shape(0);
    Rank arg_rank = arg_shape.rank();

    NODE_VALIDATION_ASSERT(this, m_shape.rank().is_static())
        << "Requested result shape has dynamic rank.";

    NODE_VALIDATION_ASSERT(this, m_one_hot_axis < static_cast<size_t>(m_shape.rank()))
        << "One-hot axis (" << m_one_hot_axis
        << ") is out of bounds (requested result shape: " << m_shape << ").";

    NODE_VALIDATION_ASSERT(this, m_shape[m_one_hot_axis].is_static())
        << "Requested result shape (" << m_shape << ") has dynamic dimension at the one-hot axis "
        << "(" << m_one_hot_axis << ").";

    PartialShape result_shape{m_shape};

    if (arg_rank.is_static())
    {
        std::vector<Dimension> expected_input_dims(static_cast<size_t>(m_shape.rank()));
        for (size_t i = 0; i < static_cast<size_t>(m_shape.rank()); i++)
        {
            expected_input_dims[i] = m_shape[i];
        }
        expected_input_dims.erase(expected_input_dims.begin() + m_one_hot_axis);
        PartialShape expected_input_shape{expected_input_dims};

        PartialShape merged_input_shape{expected_input_shape};
        NODE_VALIDATION_ASSERT(this, PartialShape::merge_into(merged_input_shape, arg_shape))
            << "Argument shape " << arg_shape << " does not match the expected shape of "
            << expected_input_shape << ".";

        std::vector<Dimension> output_dims(static_cast<size_t>(merged_input_shape.rank()));
        for (size_t i = 0; i < static_cast<size_t>(merged_input_shape.rank()); i++)
        {
            output_dims[i] = merged_input_shape[i];
        }
        output_dims.insert(output_dims.begin() + m_one_hot_axis, m_shape[m_one_hot_axis]);
        result_shape = PartialShape{output_dims};
    }

    set_output_type(0, arg_et, result_shape);
}

shared_ptr<Node> op::OneHot::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<OneHot>(new_args.at(0), m_shape, m_one_hot_axis);
}
