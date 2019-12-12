//*****************************************************************************
// Copyright 2017-2019 Intel Corporation
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

#include "ngraph/node.hpp"
#include "ngraph/op/reverse_sequence.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::ReverseSequence::type_info;

op::ReverseSequence::ReverseSequence(const Output<Node>& arg,
                                     const Output<Node>& seq_indices,
                                     int64_t batch_axis,
                                     int64_t seq_axis)
    : Op({arg, seq_indices})
    , m_batch_axis(batch_axis)
    , m_seq_axis(seq_axis)
    , m_normalized_batch_axis{0}
    , m_normalized_seq_axis{0}
{
    constructor_validate_and_infer_types();
}

void op::ReverseSequence::validate_and_infer_types()
{
    auto input_shape = get_input_partial_shape(0);
    auto input_rank = input_shape.rank();

    if (m_batch_axis < 0 || m_seq_axis < 0)
    {
        NODE_VALIDATION_CHECK(this,
                              input_rank.is_static(),
                              "In order to handle negative axes input_rank must be static (",
                              "batch_axis=",
                              m_batch_axis,
                              ", seq_axis=",
                              m_seq_axis,
                              ")");
    }
    else
    {
        m_normalized_batch_axis = m_batch_axis;
        m_normalized_seq_axis = m_seq_axis;
    }

    if (input_rank.is_static())
    {
        m_normalized_batch_axis =
            ngraph::normalize_axis(this, m_batch_axis, static_cast<int64_t>(input_rank));
        m_normalized_seq_axis =
            ngraph::normalize_axis(this, m_seq_axis, static_cast<int64_t>(input_rank));
    }

    auto indices_shape = get_input_partial_shape(1);
    auto indices_rank = indices_shape.rank();

    NODE_VALIDATION_CHECK(
        this,
        indices_rank.is_dynamic() || size_t(indices_rank) == 1,
        "Sequence indices must be a 1-dimensional tensor (sequence indices shape: ",
        get_input_partial_shape(1),
        ").");

    PartialShape output_shape{input_shape};

    if (input_rank.is_static() && indices_rank.is_static())
    {
        Dimension merged_sequence_length;

        NODE_VALIDATION_CHECK(this,
                              Dimension::merge(merged_sequence_length,
                                               input_shape[m_normalized_batch_axis],
                                               indices_shape[0]),
                              "Sequence length (",
                              indices_shape[0],
                              ") is not equal to batch axis ",
                              "dimension (",
                              input_shape[m_normalized_batch_axis],
                              ") (argument shape: ",
                              input_shape,
                              ", sequence indices shape: ",
                              indices_shape,
                              ").");
        output_shape[m_normalized_batch_axis] = merged_sequence_length;
    }

    set_output_type(0, get_input_element_type(0), output_shape);
}

shared_ptr<Node> op::ReverseSequence::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    auto res =
        make_shared<ReverseSequence>(new_args.at(0), new_args.at(1), m_batch_axis, m_seq_axis);
    return move(res);
}

void op::ReverseSequence::generate_adjoints(autodiff::Adjoints& adjoints,
                                            const OutputVector& deltas)
{
    auto x = input_value(0);
    auto rs_delta =
        make_shared<ReverseSequence>(deltas.at(0), input_value(1), m_batch_axis, m_seq_axis);
    adjoints.add_delta(x, rs_delta);
}
