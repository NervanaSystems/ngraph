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

#include "node.hpp"
#include "ops/parameter.hpp"
#include "type.hpp"

namespace ngraph
{
    namespace op
    {
        
        Node::ptr abs(const Node::ptr& arg);
        Node::ptr add(const Node::ptr& arg0, const Node::ptr& arg1);
        Node::ptr ceiling(const Node::ptr& arg0, const Node::ptr& arg1);
        //Node::ptr convert();
        //Node::ptr convolution();
        Node::ptr divide(const Node::ptr& arg0, const Node::ptr& arg1);
        Node::ptr equal(const Node::ptr& arg0, const Node::ptr& arg1);
        Node::ptr exp(const Node::ptr& arg0);
        Node::ptr floor(const Node::ptr& arg0, const Node::ptr& arg1);
        //Node::ptr get_tuple_element();
        Node::ptr greater(const Node::ptr& arg0, const Node::ptr& arg1);
        //Node::ptr greater_equal(const Node::ptr& arg0, const Node::ptr& arg1);
        Node::ptr less(const Node::ptr& arg0, const Node::ptr& arg1);
        //Node::ptr less_equal(const Node::ptr& arg0, const Node::ptr& arg1);
        Node::ptr log(const Node::ptr& arg0);
        //Node::ptr logical(); and, or, not
        Node::ptr maximum(const Node::ptr& arg0, const Node::ptr& arg1);
        Node::ptr minimum(const Node::ptr& arg0, const Node::ptr& arg1);
        Node::ptr multiply(const Node::ptr& arg0, const Node::ptr& arg1);
        Node::ptr negative(const Node::ptr& arg0);
        //Node::ptr pad();
        Node::ptr power(const Node::ptr& arg0, const Node::ptr& arg1);
        //Node::ptr reduce();
        // Node::ptr reduce_window();
        Node::ptr remainder(const Node::ptr& arg0, const Node::ptr& arg1);
        Node::ptr reshape(const Node::ptr& arg0, const Shape& shape);
        //Node::ptr reverse();
        //Node::ptr rng();
        //Node::ptr select();
        //Node::ptr select_scatter();
        //Node::ptr slice();
        Node::ptr subtract(const Node::ptr& arg0, const Node::ptr& arg1);
        //Node::ptr transpose();
        //Node::ptr while();
    }

    /// Op nodes are nodes whose value is the result of some operation
    /// applied to its arguments. For calls to user functions, the op will
    /// reference the user function.
    class Op : public Node
    {
    public:
        Op(const std::vector<Node::ptr>& arguments)
            : Node(arguments, nullptr)
        {
        }

        virtual std::string get_op_class_name() const = 0;
        virtual std::string get_node_id() const override;
    };

    /// A FunctionOp invokes a function on node arguments. In addition to the argument
    /// we need to preserve the function.
    class FunctionOp : public Op
    {
        virtual std::string description() const override { return "FunctionOp"; }
    protected:
        Node::ptr m_function;
    };

    /// The is an operation we handle directly, i.e. all type checking, etc.
    /// are defined in C++ rather than in terms of ngraph operations.
    class BuiltinOp : public Op
    {
    public:
        virtual std::string description() const override { return "BuiltinOp"; }
        /// Name of the builtin op, for debugging and logging.

        // TODO: Implement for each op. This enables graphs to be built for now.
        virtual void propagate_types() override {}
    protected:
        BuiltinOp(const std::vector<Node::ptr>& args)
            : Op(args)
        {
        }
    };

    class AbsOp : public BuiltinOp
    {
    public:
        AbsOp(const Node::ptr& arg0)
            : BuiltinOp({arg0})
        {
        }

        virtual std::string get_op_class_name() const override { return "abs"; }
        //virtual void propagate_types() override;
    };

    class AddOp : public BuiltinOp
    {
    public:
        AddOp(const Node::ptr& arg0, const Node::ptr& arg1)
            : BuiltinOp({arg0, arg1})
        {
        }
        virtual std::string get_op_class_name() const override { return "add"; }
        //virtual void propagate_types() override;
    };

    class CeilingOp : public BuiltinOp
    {
    public:
        CeilingOp(const Node::ptr& arg0, const Node::ptr& arg1)
            : BuiltinOp({arg0, arg1})
        {
        }

        virtual std::string get_op_class_name() const override { return "ceiling"; }
        //virtual void propagate_types() override;
    };

    class DivideOp : public BuiltinOp
    {
    public:
        DivideOp(const Node::ptr& arg0, const Node::ptr& arg1)
            : BuiltinOp({arg0, arg1})
        {
        }

        virtual std::string get_op_class_name() const override { return "divide"; }
        //virtual void propagate_types() override;
    };

    class EqualOp : public BuiltinOp
    {
    public:
        EqualOp(const Node::ptr& arg0, const Node::ptr& arg1)
            : BuiltinOp({arg0, arg1})
        {
        }

        virtual std::string get_op_class_name() const override { return "equal"; }
        //virtual void propagate_types() override;
    };

    class ExpOp : public BuiltinOp
    {
    public:
        ExpOp(const Node::ptr& arg0)
            : BuiltinOp({arg0})
        {
        }

        virtual std::string get_op_class_name() const override { return "exp"; }
        //virtual void propagate_types() override;
    };

    class FloorOp : public BuiltinOp
    {
    public:
        FloorOp(const Node::ptr& arg0, const Node::ptr& arg1)
            : BuiltinOp({arg0, arg1})
        {
        }

        virtual std::string get_op_class_name() const override { return "floor"; }
        //virtual void propagate_types() override;
    };

    class GreaterOp : public BuiltinOp
    {
    public:
        GreaterOp(const Node::ptr& arg0, const Node::ptr& arg1)
            : BuiltinOp({arg0, arg1})
        {
        }

        virtual std::string get_op_class_name() const override { return "greater"; }
        //virtual void propagate_types() override;
    };

    class LessOp : public BuiltinOp
    {
    public:
        LessOp(const Node::ptr& arg0, const Node::ptr& arg1)
            : BuiltinOp({arg0, arg1})
        {
        }

        virtual std::string get_op_class_name() const override { return "less"; }
        //virtual void propagate_types() override;
    };

    class LogOp : public BuiltinOp
    {
    public:
        LogOp(const Node::ptr& arg0)
            : BuiltinOp({arg0})
        {
        }

        virtual std::string get_op_class_name() const override { return "log"; }
        //virtual void propagate_types() override;
    };

    class MaximumOp : public BuiltinOp
    {
    public:
        MaximumOp(const Node::ptr& arg0, const Node::ptr& arg1)
            : BuiltinOp({arg0, arg1})
        {
        }

        virtual std::string get_op_class_name() const override { return "max"; }
        //virtual void propagate_types() override;
    };

    class MinimumOp : public BuiltinOp
    {
    public:
        MinimumOp(const Node::ptr& arg0, const Node::ptr& arg1)
            : BuiltinOp({arg0, arg1})
        {
        }

        virtual std::string get_op_class_name() const override { return "min"; }
        //virtual void propagate_types() override;
    };

    class MultiplyOp : public BuiltinOp
    {
    public:
        MultiplyOp(const Node::ptr& arg0, const Node::ptr& arg1)
            : BuiltinOp({arg0, arg1})
        {
        }

        virtual std::string get_op_class_name() const override { return "multiply"; }
        //virtual void propagate_types() override;
    };

    class NegativeOp : public BuiltinOp
    {
    public:
        NegativeOp(const Node::ptr& arg0)
            : BuiltinOp({arg0})
        {
        }

        virtual std::string get_op_class_name() const override { return "negative"; }
        //virtual void propagate_types() override;
    };

    class PowerOp : public BuiltinOp
    {
    public:
        PowerOp(const Node::ptr& arg0, const Node::ptr& arg1)
            : BuiltinOp({arg0, arg1})
        {
        }

        virtual std::string get_op_class_name() const override { return "power"; }
        //virtual void propagate_types() override;
    };

    class RemainderOp : public BuiltinOp
    {
    public:
        RemainderOp(const Node::ptr& arg0, const Node::ptr& arg1)
            : BuiltinOp({arg0, arg1})
        {
        }

        virtual std::string get_op_class_name() const override { return "remainder"; }
        //virtual void propagate_types() override;
    };

    class ReshapeOp : public BuiltinOp
    {
    public:
        ReshapeOp(const Node::ptr& arg0, const Shape& shape)
            : BuiltinOp({arg0})
            , m_shape(shape)
        {
        }

        virtual std::string get_op_class_name() const override { return "reshape"; }
        //virtual void propagate_types() override;
    protected:
        Shape m_shape;
    };

    class SubtractOp : public BuiltinOp
    {
    public:
        SubtractOp(const Node::ptr& arg0, const Node::ptr& arg1)
            : BuiltinOp({arg0, arg1})
        {
        }

        virtual std::string get_op_class_name() const override { return "subtract"; }
        //virtual void propagate_types() override;
    };
}
