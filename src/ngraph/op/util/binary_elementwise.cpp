//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#include "ngraph/op/util/binary_elementwise_arithmetic.hpp"
#include "ngraph/attribute_visitor.hpp"

using namespace std;
using namespace ngraph;

op::util::BinaryElementwise::BinaryElementwise(const AutoBroadcastSpec& autob)
    : m_autob(autob)
{
}

op::util::BinaryElementwise::BinaryElementwise(const Output<Node>& arg0,
                                                                   const Output<Node>& arg1,
                                                                   const AutoBroadcastSpec& autob)
    : Op({arg0, arg1})
    , m_autob(autob)
{
}

void op::util::BinaryElementwise::validate_and_infer_elementwise_args()
{
    element::Type element_type = get_input_element_type(0);
    PartialShape pshape = get_input_partial_shape(0);

    if (get_input_size() > 1)
    {
        for (size_t i = 1; i < get_input_size(); ++i)
        {
            NODE_VALIDATION_CHECK(
                this,
                element::Type::merge(element_type, element_type, get_input_element_type(i)),
                "Argument element types are inconsistent.");

            if (autob.m_type == op::AutoBroadcastType::NONE)
            {
                NODE_VALIDATION_CHECK(this,
                                      PartialShape::merge_into(pshape, get_input_partial_shape(i)),
                                      "Argument shapes are inconsistent.");
            }
            else if (autob.m_type == op::AutoBroadcastType::NUMPY ||
                     autob.m_type == op::AutoBroadcastType::PDPD)
            {
                NODE_VALIDATION_CHECK(
                    this,
                    PartialShape::broadcast_merge_into(pshape, get_input_partial_shape(i), autob),
                    "Argument shapes are inconsistent.");
            }
            else
            {
                NODE_VALIDATION_CHECK(this, false, "Unsupported auto broadcast specification");
            }
        }
    }

    return std::make_tuple(element_type, pshape);
}
