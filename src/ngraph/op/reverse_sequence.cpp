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
#include <memory>
#include <typeindex>
#include <typeinfo>

#include "ngraph/node.hpp"
#include "ngraph/op/reverse_sequence.hpp"

using namespace std;
using namespace ngraph;

op::ReverseSequence::ReverseSequence(const std::shared_ptr<Node> arg,
                                     const std::shared_ptr<Node> seq_indices,
                                     size_t batch_axis,
                                     size_t seq_axis)
    : Op("ReverseSequence", check_single_output_args({arg, seq_indices}))
    , m_batch_axis(batch_axis)
    , m_seq_axis(seq_axis)
{
    constructor_validate_and_infer_types();
}

void op::ReverseSequence::validate_and_infer_types()
{
    auto input_shape = get_input_partial_shape(0);
    auto input_rank = input_shape.rank();

    NODE_VALIDATION_ASSERT(this, input_rank.is_dynamic() || m_batch_axis < size_t(input_rank))
        << "Batch axis index (" << m_batch_axis
        << ") is out of bounds (argument shape: " << input_shape << ").";

    NODE_VALIDATION_ASSERT(this, input_rank.is_dynamic() || m_seq_axis < size_t(input_rank))
        << "Sequence axis index (" << m_seq_axis
        << ") is out of bounds (argument shape: " << input_shape << ").";

    auto indices_shape = get_input_partial_shape(1);
    auto indices_rank = indices_shape.rank();

    NODE_VALIDATION_ASSERT(this, indices_rank.is_dynamic() || size_t(indices_rank) == 1)
        << "Sequence indices must be a 1-dimensional tensor (sequence indices shape: "
        << get_input_partial_shape(1) << ").";

    PartialShape output_shape{input_shape};

    if (input_rank.is_static() && indices_rank.is_static())
    {
        Dimension merged_sequence_length;

        NODE_VALIDATION_ASSERT(
            this,
            Dimension::merge(merged_sequence_length, input_shape[m_batch_axis], indices_shape[0]))
            << "Sequence length (" << indices_shape[0] << ") is not equal to batch axis "
            << "dimension (" << input_shape[m_batch_axis] << ") (argument shape: " << input_shape
            << ", sequence indices shape: " << indices_shape << ").";
        output_shape[m_batch_axis] = merged_sequence_length;
    }

    set_output_type(0, get_input_element_type(0), output_shape);
}

shared_ptr<Node> op::ReverseSequence::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    auto res =
        make_shared<ReverseSequence>(new_args.at(0), new_args.at(1), m_batch_axis, m_seq_axis);
    return res;
}

void op::ReverseSequence::generate_adjoints(autodiff::Adjoints& adjoints, const NodeVector& deltas)
{
    auto x = get_argument(0);
    auto rs_delta =
        make_shared<ReverseSequence>(deltas.at(0), get_argument(1), m_batch_axis, m_seq_axis);
    adjoints.add_delta(x, rs_delta);
}
