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
    class Op
    {
    public:
        using ptr = std::shared_ptr<Op>;
        using ref = decltype(*std::shared_ptr<Op>());
    };

    class Call : public Node
    {
    public:
        using ptr = std::shared_ptr<Call>;

        Op::ptr op() const { return m_op; }

        Call(const Op::ptr& op, const std::vector<Node::ptr>& arguments)
            : Node(arguments, nullptr)
            , m_op(op)
        {
        }
    protected:
        Op::ptr m_op;
    };

    class Broadcast : public Op, public std::enable_shared_from_this<Broadcast>
    {
    public:
        using ptr = std::shared_ptr<Broadcast>;
        using ref = decltype(*std::shared_ptr<Broadcast>());

    protected:

        class BroadcastCall : public Call
        {
            friend class Broadcast;

        public:
            BroadcastCall(const Op::ptr& op, const Node::ptr& arg, size_t axis)
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
        extern Broadcast::ref broadcast;
    }

    class Dot : public Op, public std::enable_shared_from_this<Dot>
    {
    public:
        using ptr = std::shared_ptr<Dot>;
        using ref = decltype(*std::shared_ptr<Dot>());
    
    public:
        Call::ptr operator()(const Node::ptr& arg0, const Node::ptr& arg1)
        {
            return std::make_shared<Call>(shared_from_this(), std::vector<Node::ptr>{arg0, arg1});
        }
    };

    namespace op
    {
        extern Dot::ref dot;
    }
}
