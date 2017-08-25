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

using namespace ngraph;
using namespace std;

std::shared_ptr<BuiltinOp> BroadcastCall::s_op = make_shared<BuiltinOp>("broadcast");

/**
 ** /param arg The tensor view to be broadcast.
 ** /param shape The shape of the result
 ** /param broadcast_axes The axis positions (0-based) in the result that are being broadcast.
 **  the remaining axes in shape must be the same as the shape of arg.
 **/
 shared_ptr<Node> ngraph::op::broadcast(const Node::ptr&      tensor,
                                       const Shape&          shape,
                                       const vector<size_t>& broadcast_axes)
{
    return make_shared<BroadcastCall>(tensor, shape, broadcast_axes);
}

std::shared_ptr<BuiltinOp> DotCall::s_op = make_shared<BuiltinOp>("dot");

/// TODO: Semantics of arg0 and arg1 axes wrt reduction.
shared_ptr<Node> ngraph::op::dot(const Node::ptr& arg0, const Node::ptr& arg1)
{
    return make_shared<DotCall>(arg0, arg1);
}
