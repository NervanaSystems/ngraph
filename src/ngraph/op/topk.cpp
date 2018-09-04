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

#include <functional>
#include <memory>
#include <utility>

#include "ngraph/axis_vector.hpp"
#include "ngraph/op/topk.hpp"
#include "ngraph/shape.hpp"

using namespace std;
using namespace ngraph;

op::TopK::TopK(const std::shared_ptr<Node>& arg,
               size_t topk_axis,
               const element::Type& index_element_type,
               size_t k,
               bool compute_max)
               : Op("TopK", check_single_output_args({arg})),
                 m_topk_axis(topk_axis),
                 m_index_element_type(index_element_type),
                 m_k(k),
                 m_compute_max(compute_max)
{
    constructor_validate_and_infer_types();
}

void op::TopK::validate_and_infer_types()
{
    auto& input = get_inputs().at(0);
    auto rank = input.get_shape().size();

    if( rank < 1 )
    {
        throw ngraph_error("Input Tensor's rank must be at least 1");
    }
    if( m_topk_axis >= rank )
    {
        throw ngraph_error("TopK axis is greater than rank");
    }
    if( !(m_index_element_type == element::i32 || m_index_element_type == element::i64) )
    {
        throw ngraph_error("Index element type must be i64 or i32");
    }
    if( m_k > input.get_shape()[m_topk_axis] )
    {
        throw ngraph_error("K is greater than TopK axis length");
    }

    Shape input_shape = input.get_shape();
    Shape output_shape(input_shape);
    output_shape[m_topk_axis] = m_k;

    set_output_size(2);
    set_output_type(0, m_index_element_type, output_shape);
    set_output_type(1, input.get_element_type(), output_shape);
}

shared_ptr<Node> op::TopK::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() != 1)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
    return make_shared<TopK>(new_args.at(0), m_topk_axis, m_index_element_type, m_k, m_compute_max);
}

void op::TopK::generate_adjoints(autodiff::Adjoints& adjoints,
                                 const NodeVector& deltas)
{
    throw ngraph_error("Forward-propagation-only operation");
}
