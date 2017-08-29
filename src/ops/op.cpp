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

#include <algorithm>

#include "ngraph/ngraph.hpp"

using namespace ngraph;
using namespace std;

Node::ptr ngraph::op::abs(const Node::ptr& arg)
{
    return make_shared<AbsOp>(arg);
}

Node::ptr ngraph::op::add(const Node::ptr& arg0, const Node::ptr& arg1)
{
    return make_shared<AddOp>(arg0, arg1);
}

/**
 ** /param arg The tensor view to be broadcast.
 ** /param shape The shape of the result
 ** /param broadcast_axes The axis positions (0-based) in the result that are being broadcast.
 **  the remaining axes in shape must be the same as the shape of arg.
 **/
Node::ptr ngraph::op::broadcast(const Node::ptr&      tensor,
                                const Shape&          shape,
                                const vector<size_t>& broadcast_axes)
{
    return make_shared<BroadcastOp>(tensor, shape, broadcast_axes);
}

void BroadcastOp::propagate_types()
{
    auto arg_type = m_arguments.at(0)->type();
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
    m_type = make_shared<TensorViewType>(arg_tensor_view_type->element_type(), m_shape);
}

Node::ptr ngraph::op::ceiling(const Node::ptr& arg0, const Node::ptr& arg1)
{
    return make_shared<CeilingOp>(arg0, arg1);
}

// 'concatenate',
// 'constant',
// 'convert',
// 'convolution',

Node::ptr ngraph::op::divide(const Node::ptr& arg0, const Node::ptr& arg1)
{
    return make_shared<DivideOp>(arg0, arg1);
}

/// TODO: Semantics of arg0 and arg1 axes wrt reduction.
Node::ptr ngraph::op::dot(const Node::ptr& arg0, const Node::ptr& arg1)
{
    return make_shared<DotOp>(arg0, arg1);
}

void DotOp::propagate_types()
{
    auto arg0_tensor_type = dynamic_pointer_cast<TensorViewType>(m_arguments.at(0)->type());
    auto arg1_tensor_type = dynamic_pointer_cast<TensorViewType>(m_arguments.at(1)->type());
    if (nullptr == arg0_tensor_type || nullptr == arg1_tensor_type)
    {
        throw ngraph_error("Arguments to dot must be tensor views");
    }
    if (arg0_tensor_type->element_type() != arg1_tensor_type->element_type())
    {
        throw ngraph_error("Arguments to dot must have the same element type");
    }

    // Use NumPy semantics for now
    // Last axis of first arg reduces against second to last of second arg if more than one axis, else axis.
    vector<size_t> arg0_shape     = arg0_tensor_type->shape();
    vector<size_t> arg1_shape     = arg1_tensor_type->shape();
    size_t         arg0_reduction = arg0_shape.size() - 1;
    size_t         arg1_reduction;
    if (arg1_shape.size() > 1)
    {
        arg1_reduction = arg1_shape.size() - 2;
    }
    else
    {
        arg1_reduction = arg1_shape.size() - 1;
    }
    if (arg0_shape.at(arg0_reduction) != arg1_shape.at(arg1_reduction))
    {
        throw ngraph_error("Dot reduction axes not compatible");
    }
    vector<size_t> result_shape;
    copy(arg0_shape.begin(), arg0_shape.begin() + arg1_reduction, result_shape.end());
    copy(arg1_shape.begin(), arg1_shape.begin() + arg1_reduction, result_shape.end());
    copy(arg1_shape.begin() + arg1_reduction, arg1_shape.end(), result_shape.end());
    m_type = make_shared<TensorViewType>(arg0_tensor_type->element_type(), result_shape);
}

Node::ptr ngraph::op::exponential(const Node::ptr& arg0)
{
    return make_shared<ExponentialOp>(arg0);
}

Node::ptr ngraph::op::floor(const Node::ptr& arg0, const Node::ptr& arg1)
{
    return make_shared<FloorOp>(arg0, arg1);
}

Node::ptr ngraph::op::log(const Node::ptr& arg0)
{
    return make_shared<LogOp>(arg0);
}

Node::ptr ngraph::op::maximum(const Node::ptr& arg0, const Node::ptr& arg1)
{
    return make_shared<MaximumOp>(arg0, arg1);
}

Node::ptr ngraph::op::minimum(const Node::ptr& arg0, const Node::ptr& arg1)
{
    return make_shared<MinimumOp>(arg0, arg1);
}

Node::ptr ngraph::op::multiply(const Node::ptr& arg0, const Node::ptr& arg1)
{
    return make_shared<MultiplyOp>(arg0, arg1);
}

Node::ptr ngraph::op::negate(const Node::ptr& arg0)
{
    return make_shared<NegateOp>(arg0);
}

// 'pad',
// 'parameter',

Node::ptr ngraph::op::power(const Node::ptr& arg0, const Node::ptr& arg1)
{
    return make_shared<PowerOp>(arg0, arg1);
}

//'reduce',

Node::ptr ngraph::op::remainder(const Node::ptr& arg0, const Node::ptr& arg1)
{
    return make_shared<RemainderOp>(arg0, arg1);
}

Node::ptr ngraph::op::reshape(const Node::ptr& arg0, const Shape& shape)
{
    return make_shared<ReshapeOp>(arg0, shape);
}

//'reverse',
//'rng',
// 'select',
//'slice',

Node::ptr ngraph::op::subtract(const Node::ptr& arg0, const Node::ptr& arg1)
{
    return make_shared<SubtractOp>(arg0, arg1);
}

// 'transpose',
//'tuple',
// 'while'
