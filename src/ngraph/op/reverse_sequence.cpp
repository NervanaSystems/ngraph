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
    if (get_input_shape(1).size() != 1)
    {
        throw ngraph_error("indices should be a 1-dimensional array");
    }

    if (m_batch_axis >= get_input_shape(0).size())
    {
        throw ngraph_error("batch axis index is out of bounds");
    }

    if (m_seq_axis >= get_input_shape(0).size())
    {
        throw ngraph_error("sequence axis index is out of bounds");
    }

    if (get_input_shape(0).at(m_batch_axis) != get_input_shape(1).at(0))
    {
        throw ngraph_error("Sequence length size should be equal to batch axis dimension");
    }

    set_output_type(0, get_input_element_type(0), get_input_shape(0));
}

shared_ptr<Node> op::ReverseSequence::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() != 2)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }

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
