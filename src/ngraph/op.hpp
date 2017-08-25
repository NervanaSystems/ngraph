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
    /**
     ** Every instance of Op corresponds to a unique defined operation.
     **/
    class Op
    {
    protected:
        virtual ~Op() {}

    public:
        virtual const std::string& name() const = 0;
    };

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

        const std::string& description() const override { return m_op->name(); }

    protected:
        std::shared_ptr<Op> m_op;
    };

    /**
     ** There is exactly one instance of builtin op for each pre-defined operation.
     **/
    class BuiltinOp : public Op
    {
        friend class Call;

    public:
        BuiltinOp(const std::string& name)
            : m_name(name)
        {
        }

    public:
        const std::string& name() const override { return m_name; }

    protected:
        std::string m_name;
    };

    class BuiltinCall : public Call
    {
    public:
        const std::string& description() const override
        {
            static std::string name{"BuiltinCall "};
            return name;
        }

    protected:
        BuiltinCall(const std::shared_ptr<Op>& op, const std::vector<Node::ptr>& args)
            : Call(op, args)
        {
        }
    };

    namespace op
    {
        std::shared_ptr<Node> broadcast(const Node::ptr&           tensor,
                                        const Shape&               shape,
                                        const std::vector<size_t>& broadcast_axes);
    }

    class BroadcastCall : public BuiltinCall
    {
    public:
        BroadcastCall(const Node::ptr& arg, const Shape& shape, std::vector<size_t> broadcast_axes)
            : BuiltinCall(s_op, {arg})
            , m_shape(shape)
            , m_broadcast_axes(broadcast_axes)
        {
        }
        Shape               m_shape;
        std::vector<size_t> m_broadcast_axes;

    protected:
        static std::shared_ptr<BuiltinOp> s_op;
    };

    namespace op
    {
        std::shared_ptr<Node> dot(const Node::ptr& arg0, const Node::ptr& arg1);
    }

    class DotCall : public BuiltinCall
    {
    public:
        DotCall(const Node::ptr& arg0, const Node::ptr& arg1)
            : BuiltinCall(s_op, {arg0, arg1})
        {
        }

    protected:
        static std::shared_ptr<BuiltinOp> s_op;
    };
}
