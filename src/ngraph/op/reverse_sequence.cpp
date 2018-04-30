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

#include <algorithm>
#include <memory>
#include <typeindex>
#include <typeinfo>

#include "ngraph/node.hpp"
#include "ngraph/op/reverse_sequence.hpp"

using namespace std;
using namespace ngraph;

op::ReverseSequence::ReverseSequence(const std::shared_ptr<Node> arg,
                                     size_t batch_axis,
                                     size_t seq_axis,
                                     const Shape& seq_lengths)
    : RequiresTensorViewArgs("ReverseSequence", {arg})
    , m_batch_axis(batch_axis)
    , m_seq_axis(seq_axis)
{
    //this->m_batch_axis = batch_axis;
    std::cout << "batch_axis = " << batch_axis << std::endl;
    std::cout << "seq_axis = " << seq_axis << std::endl;
    std::cout << "m_batch_axis (2) = " << this->m_batch_axis << std::endl;
    std::cout << "m_seq_axis (2) = " << m_seq_axis << std::endl;
    if (arg->get_shape().at(batch_axis) != seq_lengths.size())
    {
        throw ngraph_error("Sequence length size should be equal to batch axis dimension");
    }

    for (auto d : seq_lengths)
    {
        if (d > arg->get_shape().at(seq_axis))
        {
            throw ngraph_error(
                "One of the elements of sequence lengths is greater than sequence axis dimension");
        }
        size_t min_seq_index = 1;
        m_seq_lengths.push_back(std::max(min_seq_index, d));
    }

    set_value_type_checked(arg->get_element_type(), arg->get_shape());
}

shared_ptr<Node> op::ReverseSequence::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() != 1)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }

    auto res =
        make_shared<ReverseSequence>(new_args.at(0), m_batch_axis, m_seq_axis, m_seq_lengths);
    return res;
}

void op::ReverseSequence::generate_adjoints(autodiff::Adjoints& adjoints, const NodeVector& deltas)
{
    throw ngraph_error("NYI");
}
