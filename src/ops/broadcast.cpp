// ----------------------------------------------------------------------------
// Copyright 2017 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------

#include "ngraph/ngraph.hpp"

using namespace std;
using namespace ngraph;

/**
 ** /param arg The tensor view to be broadcast.
 ** /param shape The shape of the result
 ** /param broadcast_axes The axis positions (0-based) in the result that are being broadcast.
 **  the remaining axes in shape must be the same as the shape of arg.
 **/
Node::ptr ngraph::op::broadcast(const Node::ptr&      tensor,
                                const Shape&          shape,
                                AxisSet&& broadcast_axes)
{
    return make_shared<BroadcastOp>(tensor, shape, broadcast_axes);
}

void BroadcastOp::propagate_types()
{
    auto arg_type = m_arguments.at(0)->value_type();
    if (nullptr == arg_type)
    {
        throw ngraph_error("Argument to broadcast is missing type.");
    }
    auto arg_tensor_view_type = dynamic_pointer_cast<TensorViewType>(arg_type);
    if (nullptr == arg_tensor_view_type)
    {
        throw ngraph_error("Argument to broadcast is not a tensor view");
    }
    vector<size_t> target_shape = m_shape;
    for (auto i = m_broadcast_axes.rbegin(); i != m_broadcast_axes.rend(); ++i)
    {
        target_shape.erase(target_shape.begin() + *i);
    }
    if (Shape{target_shape} != arg_tensor_view_type->shape())
    {
        throw ngraph_error("Broadcast arg, shape, and axes are incompatible");
    }
    // TODO If m_type is already set (by framework), this should verify that the type
    // we expect is consistent with the type the framework expects.
    m_value_type = make_shared<TensorViewType>(arg_tensor_view_type->element_type(), m_shape);
}
