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
        std::shared_ptr<Node> abs(const std::shared_ptr<Node>& arg);
        std::shared_ptr<Node> add(const std::shared_ptr<Node>& arg0,
                                  const std::shared_ptr<Node>& arg1);
        std::shared_ptr<Node> ceiling(const std::shared_ptr<Node>& arg0,
                                      const std::shared_ptr<Node>& arg1);
        //std::shared_ptr<Node> convert();
        //std::shared_ptr<Node> convolution();
        std::shared_ptr<Node> divide(const std::shared_ptr<Node>& arg0,
                                     const std::shared_ptr<Node>& arg1);
        std::shared_ptr<Node> equal(const std::shared_ptr<Node>& arg0,
                                    const std::shared_ptr<Node>& arg1);
        std::shared_ptr<Node> exp(const std::shared_ptr<Node>& arg0);
        std::shared_ptr<Node> floor(const std::shared_ptr<Node>& arg0,
                                    const std::shared_ptr<Node>& arg1);
        //std::shared_ptr<Node> get_tuple_element();
        std::shared_ptr<Node> greater(const std::shared_ptr<Node>& arg0,
                                      const std::shared_ptr<Node>& arg1);
        //std::shared_ptr<Node> greater_equal(const std::shared_ptr<Node>& arg0, const std::shared_ptr<Node>& arg1);
        std::shared_ptr<Node> less(const std::shared_ptr<Node>& arg0,
                                   const std::shared_ptr<Node>& arg1);
        //std::shared_ptr<Node> less_equal(const std::shared_ptr<Node>& arg0, const std::shared_ptr<Node>& arg1);
        std::shared_ptr<Node> log(const std::shared_ptr<Node>& arg0);
        //std::shared_ptr<Node> logical(); and, or, not
        std::shared_ptr<Node> maximum(const std::shared_ptr<Node>& arg0,
                                      const std::shared_ptr<Node>& arg1);
        std::shared_ptr<Node> minimum(const std::shared_ptr<Node>& arg0,
                                      const std::shared_ptr<Node>& arg1);
        std::shared_ptr<Node> multiply(const std::shared_ptr<Node>& arg0,
                                       const std::shared_ptr<Node>& arg1);
        std::shared_ptr<Node> negative(const std::shared_ptr<Node>& arg0);
        //std::shared_ptr<Node> pad();
        std::shared_ptr<Node> power(const std::shared_ptr<Node>& arg0,
                                    const std::shared_ptr<Node>& arg1);
        //std::shared_ptr<Node> reduce();
        // std::shared_ptr<Node> reduce_window();
        std::shared_ptr<Node> remainder(const std::shared_ptr<Node>& arg0,
                                        const std::shared_ptr<Node>& arg1);
        std::shared_ptr<Node> reshape(const std::shared_ptr<Node>& arg0, const Shape& shape);
        //std::shared_ptr<Node> reverse();
        //std::shared_ptr<Node> rng();
        //std::shared_ptr<Node> select();
        //std::shared_ptr<Node> select_scatter();
        //std::shared_ptr<Node> slice();
        std::shared_ptr<Node> subtract(const std::shared_ptr<Node>& arg0,
                                       const std::shared_ptr<Node>& arg1);
        //std::shared_ptr<Node> transpose();
        //std::shared_ptr<Node> while();
    }

    /// Op nodes are nodes whose value is the result of some operation
    /// applied to its arguments. For calls to user functions, the op will
    /// reference the user function.
    class Op : public Node
    {
    public:
        Op(const std::vector<std::shared_ptr<Node>>& arguments)
            : Node(arguments)
        {
        }

        Op()
            : Node()
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
        std::shared_ptr<Node> m_function;
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
        BuiltinOp(const std::vector<std::shared_ptr<Node>>& args)
            : Op(args)
        {
        }
    };

    class AbsOp : public BuiltinOp
    {
    public:
        AbsOp(const std::shared_ptr<Node>& arg0)
            : BuiltinOp({arg0})
        {
        }

        virtual std::string get_op_class_name() const override { return "abs"; }
        //virtual void propagate_types() override;
    };

    class AddOp : public BuiltinOp
    {
    public:
        AddOp(const std::shared_ptr<Node>& arg0, const std::shared_ptr<Node>& arg1)
            : BuiltinOp({arg0, arg1})
        {
        }
        virtual std::string get_op_class_name() const override { return "add"; }
        //virtual void propagate_types() override;
    };

    class CeilingOp : public BuiltinOp
    {
    public:
        CeilingOp(const std::shared_ptr<Node>& arg0, const std::shared_ptr<Node>& arg1)
            : BuiltinOp({arg0, arg1})
        {
        }

        virtual std::string get_op_class_name() const override { return "ceiling"; }
        //virtual void propagate_types() override;
    };

    class DivideOp : public BuiltinOp
    {
    public:
        DivideOp(const std::shared_ptr<Node>& arg0, const std::shared_ptr<Node>& arg1)
            : BuiltinOp({arg0, arg1})
        {
        }

        virtual std::string get_op_class_name() const override { return "divide"; }
        //virtual void propagate_types() override;
    };

    class EqualOp : public BuiltinOp
    {
    public:
        EqualOp(const std::shared_ptr<Node>& arg0, const std::shared_ptr<Node>& arg1)
            : BuiltinOp({arg0, arg1})
        {
        }

        virtual std::string get_op_class_name() const override { return "equal"; }
        //virtual void propagate_types() override;
    };

    class ExpOp : public BuiltinOp
    {
    public:
        ExpOp(const std::shared_ptr<Node>& arg0)
            : BuiltinOp({arg0})
        {
        }

        virtual std::string get_op_class_name() const override { return "exp"; }
        //virtual void propagate_types() override;
    };

    class FloorOp : public BuiltinOp
    {
    public:
        FloorOp(const std::shared_ptr<Node>& arg0, const std::shared_ptr<Node>& arg1)
            : BuiltinOp({arg0, arg1})
        {
        }

        virtual std::string get_op_class_name() const override { return "floor"; }
        //virtual void propagate_types() override;
    };

    class GreaterOp : public BuiltinOp
    {
    public:
        GreaterOp(const std::shared_ptr<Node>& arg0, const std::shared_ptr<Node>& arg1)
            : BuiltinOp({arg0, arg1})
        {
        }

        virtual std::string get_op_class_name() const override { return "greater"; }
        //virtual void propagate_types() override;
    };

    class LessOp : public BuiltinOp
    {
    public:
        LessOp(const std::shared_ptr<Node>& arg0, const std::shared_ptr<Node>& arg1)
            : BuiltinOp({arg0, arg1})
        {
        }

        virtual std::string get_op_class_name() const override { return "less"; }
        //virtual void propagate_types() override;
    };

    class LogOp : public BuiltinOp
    {
    public:
        LogOp(const std::shared_ptr<Node>& arg0)
            : BuiltinOp({arg0})
        {
        }

        virtual std::string get_op_class_name() const override { return "log"; }
        //virtual void propagate_types() override;
    };

    class MaximumOp : public BuiltinOp
    {
    public:
        MaximumOp(const std::shared_ptr<Node>& arg0, const std::shared_ptr<Node>& arg1)
            : BuiltinOp({arg0, arg1})
        {
        }

        virtual std::string get_op_class_name() const override { return "max"; }
        //virtual void propagate_types() override;
    };

    class MinimumOp : public BuiltinOp
    {
    public:
        MinimumOp(const std::shared_ptr<Node>& arg0, const std::shared_ptr<Node>& arg1)
            : BuiltinOp({arg0, arg1})
        {
        }

        virtual std::string get_op_class_name() const override { return "min"; }
        //virtual void propagate_types() override;
    };

    class MultiplyOp : public BuiltinOp
    {
    public:
        MultiplyOp(const std::shared_ptr<Node>& arg0, const std::shared_ptr<Node>& arg1)
            : BuiltinOp({arg0, arg1})
        {
        }

        virtual std::string get_op_class_name() const override { return "multiply"; }
        //virtual void propagate_types() override;
    };

    class NegativeOp : public BuiltinOp
    {
    public:
        NegativeOp(const std::shared_ptr<Node>& arg0)
            : BuiltinOp({arg0})
        {
        }

        virtual std::string get_op_class_name() const override { return "negative"; }
        //virtual void propagate_types() override;
    };

    class PowerOp : public BuiltinOp
    {
    public:
        PowerOp(const std::shared_ptr<Node>& arg0, const std::shared_ptr<Node>& arg1)
            : BuiltinOp({arg0, arg1})
        {
        }

        virtual std::string get_op_class_name() const override { return "power"; }
        //virtual void propagate_types() override;
    };

    class RemainderOp : public BuiltinOp
    {
    public:
        RemainderOp(const std::shared_ptr<Node>& arg0, const std::shared_ptr<Node>& arg1)
            : BuiltinOp({arg0, arg1})
        {
        }

        virtual std::string get_op_class_name() const override { return "remainder"; }
        //virtual void propagate_types() override;
    };

    class ReshapeOp : public BuiltinOp
    {
    public:
        ReshapeOp(const std::shared_ptr<Node>& arg0, const Shape& shape)
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
        SubtractOp(const std::shared_ptr<Node>& arg0, const std::shared_ptr<Node>& arg1)
            : BuiltinOp({arg0, arg1})
        {
        }

        virtual std::string get_op_class_name() const override { return "subtract"; }
        //virtual void propagate_types() override;
    };
}
