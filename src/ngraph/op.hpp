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
     ** Call nodes are nodes whose value is the result of some operation, the op,
     ** applied to its arguments. We use the op as a callable to construct the
     ** call nodes. For calls to user functions, the op will be the user function.
     **/
    class Call : public Node
    {
    public:

        Call(const std::vector<Node::ptr>& arguments)
            : Node(arguments, nullptr)
        {
        }

        /**
         ** Return true if this has the same implementing class as call. This
         ** will be used by the pattern matcher when comparing a pattern
         ** graph against the graph.
         **/
        bool has_same_op(Call& call) { return typeid(this) == typeid(&call); }
        virtual std::string description() const override { return "Call"; }
    };

    class BuiltinCall : public Call
    {
    public:
        virtual std::string description() const override { return "BuiltinCall"; }
        /// Name of the builtin op, for debugging and logging.
        virtual std::string op_name() const = 0;
        
        // TODO: Implement for each op
        virtual void propagate_types() override {}

    protected:
        BuiltinCall(const std::vector<Node::ptr>& args)
            : Call(args)
        {
        }
    };

    class AbsCall : public BuiltinCall
    {
    public:
        AbsCall(const Node::ptr& arg0)
            : BuiltinCall({arg0})
        {
        }

        virtual std::string op_name() const override { return "abs"; }
        //virtual void propagate_types() override;  
    };

    class AddCall : public BuiltinCall
    {
    public:
        AddCall(const Node::ptr& arg0, const Node::ptr& arg1)
            : BuiltinCall({arg0, arg1})
        {
        }
        virtual std::string op_name() const override { return "add"; }
        //virtual void propagate_types() override;  
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
            : BuiltinCall({arg})
            , m_shape(shape)
            , m_broadcast_axes(broadcast_axes)
        {
        }

        virtual std::string op_name() const override { return "broadcast"; }
        virtual void propagate_types() override;

    protected:
        Shape               m_shape;
        std::vector<size_t> m_broadcast_axes;
    };

    class CeilingCall : public BuiltinCall
    {
    public:
        CeilingCall(const Node::ptr& arg0, const Node::ptr& arg1)
            : BuiltinCall({arg0, arg1})
        {
        }

        virtual std::string op_name() const override { return "ceiling"; }
        //virtual void propagate_types() override;
    };

    class DivideCall : public BuiltinCall
    {
    public:
        DivideCall(const Node::ptr& arg0, const Node::ptr& arg1)
            : BuiltinCall({arg0, arg1})
        {
        }

        virtual std::string op_name() const override { return "divide"; }
        //virtual void propagate_types() override;
    };

    class DotCall : public BuiltinCall
    {
    public:
        /// TODO: Semantics of arg0 and arg1 axes wrt reduction.
        DotCall(const Node::ptr& arg0, const Node::ptr& arg1)
            : BuiltinCall({arg0, arg1})
        {
        }
        
        virtual std::string op_name() const override { return "dot"; }
        virtual void propagate_types() override;
    };

    class EqualCall : public BuiltinCall
    {
    public:
        EqualCall(const Node::ptr& arg0, const Node::ptr& arg1)
            : BuiltinCall({arg0, arg1})
        {
        }

        virtual std::string op_name() const override { return "equal"; }
        //virtual void propagate_types() override;
    };

    class ExponentialCall : public BuiltinCall
    {
    public:
        ExponentialCall(const Node::ptr& arg0)
            : BuiltinCall({arg0})
        {
        }
        
        virtual std::string op_name() const override { return "exp"; }
        //virtual void propagate_types() override;
    };

    class FloorCall : public BuiltinCall
    {
    public:
        FloorCall(const Node::ptr& arg0, const Node::ptr& arg1)
            : BuiltinCall({arg0, arg1})
        {
        }
        
        virtual std::string op_name() const override { return "floor"; }
        //virtual void propagate_types() override;
    };

    class GreaterCall : public BuiltinCall
    {
    public:
        GreaterCall(const Node::ptr& arg0, const Node::ptr& arg1)
            : BuiltinCall({arg0, arg1})
        {
        }

        virtual std::string op_name() const override { return "greater"; }
        //virtual void propagate_types() override;
    };

    class LessCall : public BuiltinCall
    {
    public:
        LessCall(const Node::ptr& arg0, const Node::ptr& arg1)
            : BuiltinCall({arg0, arg1})
        {
        }
        
        virtual std::string op_name() const override { return "less"; }
        //virtual void propagate_types() override;
    };

    class LogCall : public BuiltinCall
    {
    public:
        LogCall(const Node::ptr& arg0)
            : BuiltinCall({arg0})
        {
        }

        virtual std::string op_name() const override { return "log"; }
        //virtual void propagate_types() override;
    };

    class MaximumCall : public BuiltinCall
    {
    public:
        MaximumCall(const Node::ptr& arg0, const Node::ptr& arg1)
            : BuiltinCall({arg0, arg1})
        {
        }

        virtual std::string op_name() const override { return "max"; }
        //virtual void propagate_types() override;
    };

    class MinimumCall : public BuiltinCall
    {
    public:
        MinimumCall(const Node::ptr& arg0, const Node::ptr& arg1)
            : BuiltinCall({arg0, arg1})
        {
        }

        virtual std::string op_name() const override { return "min"; }
        //virtual void propagate_types() override;
    };

    class MultiplyCall : public BuiltinCall
    {
    public:
        MultiplyCall(const Node::ptr& arg0, const Node::ptr& arg1)
            : BuiltinCall({arg0, arg1})
        {
        }

        virtual std::string op_name() const override { return "multiply"; }
        //virtual void propagate_types() override;
    };

    class NegateCall : public BuiltinCall
    {
    public:
        NegateCall(const Node::ptr& arg0)
            : BuiltinCall({arg0})
        {
        }

        virtual std::string op_name() const override { return "negate"; }
        //virtual void propagate_types() override;
    };

    class PowerCall : public BuiltinCall
    {
    public:
        PowerCall(const Node::ptr& arg0, const Node::ptr& arg1)
            : BuiltinCall({arg0, arg1})
        {
        }
        
        virtual std::string op_name() const override { return "power"; }
        //virtual void propagate_types() override;
    };

    class RemainderCall : public BuiltinCall
    {
    public:
        RemainderCall(const Node::ptr& arg0, const Node::ptr& arg1)
            : BuiltinCall({arg0, arg1})
        {
        }

        virtual std::string op_name() const override { return "remainder"; }
        //virtual void propagate_types() override;
    };

    class ReshapeCall : public BuiltinCall
    {
    public:
        ReshapeCall(const Node::ptr& arg0, const Shape& shape)
            : BuiltinCall({arg0})
            , m_shape(shape)
        {
        }

        virtual std::string op_name() const override { return "reshape"; }
        //virtual void propagate_types() override;
    protected:
        Shape m_shape;
    };

    class SubtractCall : public BuiltinCall
    {
    public:
        SubtractCall(const Node::ptr& arg0, const Node::ptr& arg1)
            : BuiltinCall({arg0, arg1})
        {
        }

        virtual std::string op_name() const override { return "subtract"; }
        //virtual void propagate_types() override;
    };
}
