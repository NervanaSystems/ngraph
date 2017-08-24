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

#pragma once

#include <memory>

#include "ngraph/node.hpp"
#include "ngraph/type.hpp"

namespace ngraph
{
    class Op;

    /**
     ** Call nodes are nodes whose value is the result of some operation, the op,
     ** applied to its arguments. We use the op as a callable to construct the
     ** call nodes.
     **/
    class Call : public Node
    {
    public:
        std::shared_ptr<Op> op() const { return m_op; }

        Call(const std::shared_ptr<Op>& op, const std::vector<Node::ptr>& arguments)
            : Node(arguments, nullptr)
            , m_op(op)
        {
        }

    protected:
        std::shared_ptr<Op> m_op;
    };

    /**
     ** The Op class provides the behavior for a Call.
     **/
    class Op
    {
    };

    class Broadcast : public Op, public std::enable_shared_from_this<Broadcast>
    {
    protected:
        class BroadcastCall : public Call
        {
            friend class Broadcast;

        public:
            BroadcastCall(const std::shared_ptr<Op>& op, const Node::ptr& arg, size_t axis)
                : Call(op, {arg})
                , m_axis(axis)
            {
            }

        protected:
            size_t m_axis;
        };

    public:
        std::shared_ptr<BroadcastCall> operator()(const Node::ptr& tensor, size_t axis)
        {
            return std::make_shared<BroadcastCall>(shared_from_this(), tensor, axis);
        }
    };

    namespace op
    {
        extern decltype(*std::shared_ptr<Broadcast>()) broadcast;
    }

    class Dot : public Op, public std::enable_shared_from_this<Dot>
    {
    public:
        Call::ptr operator()(const Node::ptr& arg0, const Node::ptr& arg1)
        {
            return std::make_shared<Call>(shared_from_this(), std::vector<Node::ptr>{arg0, arg1});
        }
    };

    namespace op
    {
        extern decltype(*std::shared_ptr<Dot>()) dot;
    }
}
