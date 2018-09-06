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
    auto& input = get_inputs().at(0);
    auto rank = input.get_shape().size();

    NODE_VALIDATION_ASSERT(this, rank > 0) << "Input Tensor's rank must be greater than 0";
    NODE_VALIDATION_ASSERT(this, m_top_k_axis < rank) << "TopK axis must be less than rank";
    NODE_VALIDATION_ASSERT(
        this, m_index_element_type == element::i32 || m_index_element_type == element::i64)
        << "Index element type must be i64 or i32";
    NODE_VALIDATION_ASSERT(this, m_k <= input.get_shape()[m_top_k_axis])
        << "K should not exceed TopK axis length";

    Shape input_shape = input.get_shape();
    Shape output_shape(input_shape);
    if (m_k != 0)
    {
        output_shape[m_top_k_axis] = m_k;
    }
    else
    {
        m_k = input_shape[m_top_k_axis];
    }

    set_output_size(2);
    set_output_type(0, m_index_element_type, output_shape);
    set_output_type(1, input.get_element_type(), output_shape);
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
