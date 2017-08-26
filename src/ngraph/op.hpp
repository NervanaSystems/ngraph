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
    namespace op
    {
        Node::ptr abs(const Node::ptr& arg);
        Node::ptr add(const Node::ptr& arg0, const Node::ptr& arg1);
        Node::ptr broadcast(const Node::ptr&           tensor,
                            const Shape&               shape,
                            const std::vector<size_t>& broadcast_axes);

        //Node::ptr candidate();
        Node::ptr ceiling(const Node::ptr& arg0, const Node::ptr& arg1);
        //Node::ptr concatenate();
        //Node::ptr constant();
        //Node::ptr convert();
        //Node::ptr convolution();
        Node::ptr divide(const Node::ptr& arg0, const Node::ptr& arg1);
        Node::ptr dot(const Node::ptr& arg0, const Node::ptr& arg1);
        Node::ptr equal(const Node::ptr& arg0, const Node::ptr& arg1);
        Node::ptr exponential(const Node::ptr& arg0);
        Node::ptr floor(const Node::ptr& arg0, const Node::ptr& arg1);
        //Node::ptr get();
        Node::ptr greater(const Node::ptr& arg0, const Node::ptr& arg1);
        Node::ptr less(const Node::ptr& arg0, const Node::ptr& arg1);
        Node::ptr log(const Node::ptr& arg0);
        //Node::ptr logical();
        Node::ptr maximum(const Node::ptr& arg0, const Node::ptr& arg1);
        Node::ptr minimum(const Node::ptr& arg0, const Node::ptr& arg1);
        Node::ptr multiply(const Node::ptr& arg0, const Node::ptr& arg1);
        Node::ptr negate(const Node::ptr& arg0);
        //Node::ptr pad();
        Node::ptr power(const Node::ptr& arg0, const Node::ptr& arg1);
        //Node::ptr reduce();
        Node::ptr remainder(const Node::ptr& arg0, const Node::ptr& arg1);
        Node::ptr reshape(const Node::ptr& arg0, const Shape& shape);
        //Node::ptr reverse();
        //Node::ptr rng();
        //Node::ptr select();
        //Node::ptr slice();
        Node::ptr subtract(const Node::ptr& arg0, const Node::ptr& arg1);
        //Node::ptr transpose();
        //Node::ptr tuple();
        //Node::ptr while();
    }

    /**
     ** Every instance of Op corresponds to a unique defined operation.
     **/
    class Op
    {
    protected:
        virtual ~Op() {}

    public:
        virtual std::string name() const = 0;
    };

    /**
     ** Call nodes are nodes whose value is the result of some operation, the op,
     ** applied to its arguments. We use the op as a callable to construct the
     ** call nodes. For calls to user functions, the op will be the user function.
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

        virtual std::string description() const override { return m_op->name(); }

    protected:
        std::shared_ptr<Op> m_op;
    };

    /**
     ** There is exactly one instance of builtin op for each pre-defined operation. These
     ** are intended to be used when matching calls in different graphs; every FooCall
     ** will have the same op.
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
        std::string name() const override { return m_name; }

    protected:
        std::string m_name;
    };

    class BuiltinCall : public Call
    {
    public:
        virtual std::string description() const override { return "BuiltinCall"; }

        // TODO: Implement for each op
        virtual void propagate_types() override {}

    protected:
        BuiltinCall(const std::shared_ptr<Op>& op, const std::vector<Node::ptr>& args)
            : Call(op, args)
        {
        }
    };

    class AbsCall : public BuiltinCall
    {
    public:
        AbsCall(const Node::ptr& arg0)
            : BuiltinCall(s_op, {arg0})
        {
        }

    protected:
        static std::shared_ptr<BuiltinOp> s_op;
    };

    class AddCall : public BuiltinCall
    {
    public:
        AddCall(const Node::ptr& arg0, const Node::ptr& arg1)
            : BuiltinCall(s_op, {arg0, arg1})
        {
        }
        //virtual void propagate_types() override;
    protected:
        static std::shared_ptr<BuiltinOp> s_op;
    };

    class BroadcastCall : public BuiltinCall
    {
    public:
        /**
         ** /param arg The tensor view to be broadcast.
         ** /param shape The shape of the result
         ** /param broadcast_axes The axis positions (0-based) in the result that are being broadcast.
         **  the remaining axes in shape must be the same as the shape of arg.
         **/
        BroadcastCall(const Node::ptr& arg, const Shape& shape, std::vector<size_t> broadcast_axes)
            : BuiltinCall(s_op, {arg})
            , m_shape(shape)
            , m_broadcast_axes(broadcast_axes)
        {
        }

        virtual void propagate_types() override;

    protected:
        Shape               m_shape;
        std::vector<size_t> m_broadcast_axes;

        static std::shared_ptr<BuiltinOp> s_op;
    };

    class CeilingCall : public BuiltinCall
    {
    public:
        CeilingCall(const Node::ptr& arg0, const Node::ptr& arg1)
            : BuiltinCall(s_op, {arg0, arg1})
        {
        }
        //virtual void propagate_types() override;
    protected:
        static std::shared_ptr<BuiltinOp> s_op;
    };

    class DivideCall : public BuiltinCall
    {
    public:
        DivideCall(const Node::ptr& arg0, const Node::ptr& arg1)
            : BuiltinCall(s_op, {arg0, arg1})
        {
        }
        //virtual void propagate_types() override;
    protected:
        static std::shared_ptr<BuiltinOp> s_op;
    };

    class DotCall : public BuiltinCall
    {
    public:
        /// TODO: Semantics of arg0 and arg1 axes wrt reduction.
        DotCall(const Node::ptr& arg0, const Node::ptr& arg1)
            : BuiltinCall(s_op, {arg0, arg1})
        {
        }
        virtual void propagate_types() override;

    protected:
        static std::shared_ptr<BuiltinOp> s_op;
    };

    class EqualCall : public BuiltinCall
    {
    public:
        EqualCall(const Node::ptr& arg0, const Node::ptr& arg1)
            : BuiltinCall(s_op, {arg0, arg1})
        {
        }
        //virtual void propagate_types() override;
    protected:
        static std::shared_ptr<BuiltinOp> s_op;
    };

    class ExponentialCall : public BuiltinCall
    {
    public:
        ExponentialCall(const Node::ptr& arg0)
            : BuiltinCall(s_op, {arg0})
        {
        }
        //virtual void propagate_types() override;
    protected:
        static std::shared_ptr<BuiltinOp> s_op;
    };

    class FloorCall : public BuiltinCall
    {
    public:
        FloorCall(const Node::ptr& arg0, const Node::ptr& arg1)
            : BuiltinCall(s_op, {arg0, arg1})
        {
        }
        //virtual void propagate_types() override;
    protected:
        static std::shared_ptr<BuiltinOp> s_op;
    };

    class GreaterCall : public BuiltinCall
    {
    public:
        GreaterCall(const Node::ptr& arg0, const Node::ptr& arg1)
            : BuiltinCall(s_op, {arg0, arg1})
        {
        }
        //virtual void propagate_types() override;
    protected:
        static std::shared_ptr<BuiltinOp> s_op;
    };

    class LessCall : public BuiltinCall
    {
    public:
        LessCall(const Node::ptr& arg0, const Node::ptr& arg1)
            : BuiltinCall(s_op, {arg0, arg1})
        {
        }
        //virtual void propagate_types() override;
    protected:
        static std::shared_ptr<BuiltinOp> s_op;
    };

    class LogCall : public BuiltinCall
    {
    public:
        LogCall(const Node::ptr& arg0)
            : BuiltinCall(s_op, {arg0})
        {
        }
        //virtual void propagate_types() override;
    protected:
        static std::shared_ptr<BuiltinOp> s_op;
    };

    class MaximumCall : public BuiltinCall
    {
    public:
        MaximumCall(const Node::ptr& arg0, const Node::ptr& arg1)
            : BuiltinCall(s_op, {arg0, arg1})
        {
        }
        //virtual void propagate_types() override;
    protected:
        static std::shared_ptr<BuiltinOp> s_op;
    };

    class MinimumCall : public BuiltinCall
    {
    public:
        MinimumCall(const Node::ptr& arg0, const Node::ptr& arg1)
            : BuiltinCall(s_op, {arg0, arg1})
        {
        }
        //virtual void propagate_types() override;
    protected:
        static std::shared_ptr<BuiltinOp> s_op;
    };

    class MultiplyCall : public BuiltinCall
    {
    public:
        MultiplyCall(const Node::ptr& arg0, const Node::ptr& arg1)
            : BuiltinCall(s_op, {arg0, arg1})
        {
        }
        //virtual void propagate_types() override;
    protected:
        static std::shared_ptr<BuiltinOp> s_op;
    };

    class NegateCall : public BuiltinCall
    {
    public:
        NegateCall(const Node::ptr& arg0)
            : BuiltinCall(s_op, {arg0})
        {
        }
        //virtual void propagate_types() override;
    protected:
        static std::shared_ptr<BuiltinOp> s_op;
    };

    class PowerCall : public BuiltinCall
    {
    public:
        PowerCall(const Node::ptr& arg0, const Node::ptr& arg1)
            : BuiltinCall(s_op, {arg0, arg1})
        {
        }
        //virtual void propagate_types() override;
    protected:
        static std::shared_ptr<BuiltinOp> s_op;
    };

    class RemainderCall : public BuiltinCall
    {
    public:
        RemainderCall(const Node::ptr& arg0, const Node::ptr& arg1)
            : BuiltinCall(s_op, {arg0, arg1})
        {
        }
        //virtual void propagate_types() override;
    protected:
        static std::shared_ptr<BuiltinOp> s_op;
    };

    class ReshapeCall : public BuiltinCall
    {
    public:
        ReshapeCall(const Node::ptr& arg0, const Shape& shape)
            : BuiltinCall(s_op, {arg0})
            , m_shape(shape)
        {
        }
        //virtual void propagate_types() override;
    protected:
        Shape m_shape;

        static std::shared_ptr<BuiltinOp> s_op;
    };

    class SubtractCall : public BuiltinCall
    {
    public:
        SubtractCall(const Node::ptr& arg0, const Node::ptr& arg1)
            : BuiltinCall(s_op, {arg0, arg1})
        {
        }
        //virtual void propagate_types() override;
    protected:
        static std::shared_ptr<BuiltinOp> s_op;
    };
}
