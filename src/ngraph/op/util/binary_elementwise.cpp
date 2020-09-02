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

#include "ngraph/op/util/binary_elementwise.hpp"

using namespace ngraph;
using namespace std;

op::util::BinaryElementwise::BinaryElementwise() {}

op::util::BinaryElementwise::BinaryElementwise(const Output<Node>& arg0,
                                               const Output<Node>& arg1,
                                               const AutoBroadcastSpec& autob)
    : Op({arg0, arg1})
    , m_autob(autob)
{
}

bool op::util::BinaryElementwise::visit_attributes(AttributeVisitor& visitor)
{
    visitor.on_attribute("auto_broadcast", m_autob);
    return true;
}

Shape op::util::BinaryElementwise::compute_output_shape(const Shape& arg0_shape,
                                                        const Shape& arg1_shape) const
{
    PartialShape arg0_partial_shape = arg0_shape;
    if (m_autob.m_type == op::AutoBroadcastType::NONE)
    {
        NGRAPH_CHECK(PartialShape::merge_into(arg0_partial_shape, arg1_shape),
                     "Argument shapes are inconsistent.");
    }
    else if (m_autob.m_type == op::AutoBroadcastType::NUMPY ||
             m_autob.m_type == op::AutoBroadcastType::PDPD)
    {
        NGRAPH_CHECK(PartialShape::broadcast_merge_into(arg0_partial_shape, arg1_shape, m_autob),
                     "Argument shapes are inconsistent.");
    }
    else
    {
        NGRAPH_CHECK(false, "Unsupported auto broadcast specification");
    }
    return arg0_partial_shape.get_shape();
}
