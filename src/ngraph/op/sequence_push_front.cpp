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

#include "ngraph/op/sequence_push_front.hpp"

using namespace std;
using namespace ngraph;

const string op::SequencePushFront::type_name("SequencePushFront");

op::SequencePushFront::SequencePushFront(const Output<Node>& value,
                                         const Output<Node>& sequence,
                                         const AutoBroadcastSpec& autob)
    : Op({value, sequence})
{
}

void op::SequencePushFront::validate_and_infer_types()
{
    NODE_VALIDATION_CHECK(this,
                          get_input_element_type(0) == get_input_element_type(1),
                          "Argument element types are inconsistent.");

    PartialShape result_shape;
    const PartialShape& value_shape = get_input_partial_shape(0);
    const PartialShape& sequence_shape = get_input_partial_shape(1);
    if (sequence_shape.is_dynamic() || value_shape.is_dynamic())
    {
        result_shape = PartialShape::dynamic();
    }
    else
    {
        result_shape = sequence_shape;
        NODE_VALIDATION_CHECK(
            this,
            PartialShape::broadcast_merge_into(result_shape, value_shape, m_autob),
            "Incompatible shapes");
        if (result_shape[0].is_static())
        {
            result_shape[0] += 1;
        }
    }
    set_output_size(1);
    set_output_type(0, get_input_element_type(0), result_shape);
}

shared_ptr<Node> op::SequencePushFront::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<SequencePushFront>(new_args.at(0), new_args.at(1));
}
