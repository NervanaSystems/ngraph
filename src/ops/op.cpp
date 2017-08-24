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

shared_ptr<Broadcast> ngraph::Broadcast::s_op = make_shared<ngraph::Broadcast>();

shared_ptr<Node> ngraph::op::broadcast(const Node::ptr& tensor, size_t axis)
{
    return make_shared<Broadcast::BroadcastCall>(Broadcast::s_op->shared_from_this(), tensor, axis);
}

shared_ptr<Dot> ngraph::Dot::s_op = make_shared<ngraph::Dot>();

shared_ptr<Node> ngraph::op::dot(const Node::ptr& arg0, const Node::ptr& arg1)
{
    return make_shared<Call>(Dot::s_op->shared_from_this(), std::vector<Node::ptr>{arg0, arg1});
}
