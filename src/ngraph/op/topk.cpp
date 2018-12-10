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

#include <memory>

#include "ngraph/axis_vector.hpp"
#include "ngraph/op/topk.hpp"
#include "ngraph/shape.hpp"

using namespace std;
using namespace ngraph;

op::TopK::TopK(const shared_ptr<Node>& arg,
               size_t top_k_axis,
               const element::Type& index_element_type,
               size_t k,
               bool compute_max)
    : Op("TopK", check_single_output_args({arg}))
    , m_top_k_axis(top_k_axis)
    , m_index_element_type(index_element_type)
    , m_k(k)
    , m_compute_max(compute_max)
{
    constructor_validate_and_infer_types();
}

void op::TopK::validate_and_infer_types()
{
    const PartialShape& input_shape = get_input_partial_shape(0);
    Rank input_rank = input_shape.rank();
    element::Type input_element_type = get_input_element_type(0);

    NODE_VALIDATION_ASSERT(this, !m_index_element_type.is_dynamic())
        << "Argument element type must not be dynamic.";

    NODE_VALIDATION_ASSERT(
        this, m_index_element_type == element::i32 || m_index_element_type == element::i64)
        << "Argument element type must be i64 or i32 (got " << m_index_element_type << ").";

    NODE_VALIDATION_ASSERT(this, input_rank.is_dynamic() || static_cast<size_t>(input_rank) > 0)
        << "Argument rank must be greater than 0.";

    NODE_VALIDATION_ASSERT(
        this, input_rank.is_dynamic() || m_top_k_axis < static_cast<size_t>(input_rank))
        << "TopK axis (" << m_top_k_axis << ") is out of bounds.";

    NODE_VALIDATION_ASSERT(this,
                           input_rank.is_dynamic() || input_shape[m_top_k_axis].is_dynamic() ||
                               m_k <= static_cast<size_t>(input_shape[m_top_k_axis]))
        << "K (" << m_k << ") exceeds the dimension ("
        << (input_rank.is_static() ? input_shape[m_top_k_axis] : 0) << ") of the TopK axis (axis "
        << m_top_k_axis << ").";

    PartialShape output_shape{input_shape};

    if (input_rank.is_static())
    {
        if (m_k != 0)
        {
            output_shape[m_top_k_axis] = m_k;
        }
        else if (input_shape[m_top_k_axis].is_static())
        {
            m_k = static_cast<size_t>(input_shape[m_top_k_axis]);
        }
    }

    set_output_size(2);
    set_output_type(0, m_index_element_type, output_shape);
    set_output_type(1, input_element_type, output_shape);
}

shared_ptr<Node> op::TopK::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<TopK>(
        new_args.at(0), m_top_k_axis, m_index_element_type, m_k, m_compute_max);
}

void op::TopK::generate_adjoints(autodiff::Adjoints& adjoints, const NodeVector& deltas)
{
    throw ngraph_error("Forward-propagation-only operation");
}
