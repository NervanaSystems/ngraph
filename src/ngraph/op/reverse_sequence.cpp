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
    : RequiresTensorViewArgs("ReverseSequence", {arg, seq_indices})
    , m_batch_axis(batch_axis)
    , m_seq_axis(seq_axis)
{
    NODE_VALIDATION_ASSERT(this, seq_indices->get_shape().size() == 1)
        << "Sequence indices must be a 1-dimensional tensor (sequence indices shape: "
        << seq_indices->get_shape() << ").";

    NODE_VALIDATION_ASSERT(this, batch_axis < arg->get_shape().size())
        << "Batch axis index (" << batch_axis
        << ") is out of bounds (argument shape: " << arg->get_shape() << ").";

    NODE_VALIDATION_ASSERT(this, seq_axis < arg->get_shape().size())
        << "Sequence axis index (" << seq_axis
        << ") is out of bounds (argument shape: " << arg->get_shape() << ").";

    NODE_VALIDATION_ASSERT(this, arg->get_shape()[batch_axis] == seq_indices->get_shape()[0])
        << "Sequence length (" << seq_indices->get_shape()[0] << ") is not equal to batch axis "
        << "dimension (" << arg->get_shape()[batch_axis]
        << ") (argument shape: " << arg->get_shape()
        << ", sequence indices shape: " << seq_indices->get_shape() << ").";

    set_value_type_checked(arg->get_element_type(), arg->get_shape());
}

shared_ptr<Node> op::ReverseSequence::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args, 2);
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
