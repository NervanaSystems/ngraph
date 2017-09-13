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
#include "ngraph/ops/parameter.hpp"
#include "ngraph/type.hpp"

namespace ngraph
{
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

    // TODO: These class definitions are to be moved into separate files in the op directory
    namespace op
    {
        /// A Function invokes a function on node arguments. In addition to the argument
        /// we need to preserve the function.
        class FunctionCall : public Op
        {
            virtual std::string description() const override { return "FunctionCall"; }

        protected:
            std::shared_ptr<Node> m_function;
        };

        /// The is an operation we handle directly, i.e. all type checking, etc.
        /// are defined in C++ rather than in terms of ngraph operations.
        class Builtin : public Op
        {
        public:
            virtual std::string description() const override { return "Builtin"; }

        protected:
            Builtin(const std::vector<std::shared_ptr<Node>>& args)
                : Op(args)
            {
            }
        };

        /// Index ops create a new way to index the same tensor elements
        class IndexBuiltin : public Builtin
        {
        protected:
            IndexBuiltin(const std::shared_ptr<Node>& arg)
                : Builtin(Nodes{arg})
            {
            }
        };

        class Reshape : public IndexBuiltin
        {
        public:
            Reshape(const std::shared_ptr<Node>& arg0, const Shape& shape)
                : IndexBuiltin(arg0)
                , m_shape(shape)
            {
            }

            virtual std::string get_op_class_name() const override { return "Reshape"; }
            //virtual void propagate_types() override;
        protected:
            Shape m_shape;
        };

        /// Operations where the same element function is applied to each element
        /// Op(X)[I] = op(X[I])
        class UnaryElementwiseBuiltin : public Builtin
        {
        protected:
            UnaryElementwiseBuiltin(const std::shared_ptr<Node>& arg)
                : Builtin(Nodes{arg})
            {
            }
            virtual const element::Type& infer_result_element_type(
                                            const element::Type& arg_element_type) const;

        public:
            virtual void propagate_types() override;
        };

        class UnaryElementwiseArithmetic : public UnaryElementwiseBuiltin
        {
        protected:
            UnaryElementwiseArithmetic(const std::shared_ptr<Node>& arg)
                : UnaryElementwiseBuiltin({arg})
            {
            }
            virtual const element::Type& infer_result_element_type(
                                            const element::Type& arg_element_type) const override;
        };

        class Abs : public UnaryElementwiseArithmetic
        {
        public:
            Abs(const std::shared_ptr<Node>& arg0)
                : UnaryElementwiseArithmetic({arg0})
            {
            }

            virtual std::string get_op_class_name() const override { return "Abs"; }
            //virtual void propagate_types() override;
        };

        class Exp : public UnaryElementwiseArithmetic
        {
        public:
            Exp(const std::shared_ptr<Node>& arg0)
                : UnaryElementwiseArithmetic(arg0)
            {
            }

            virtual std::string get_op_class_name() const override { return "Exp"; }
            //virtual void propagate_types() override;
        };

        class Log : public UnaryElementwiseArithmetic
        {
        public:
            Log(const std::shared_ptr<Node>& arg0)
                : UnaryElementwiseArithmetic(arg0)
            {
            }

            virtual std::string get_op_class_name() const override { return "Log"; }
            //virtual void propagate_types() override;
        };

        class Negative : public UnaryElementwiseArithmetic
        {
        public:
            Negative(const std::shared_ptr<Node>& arg0)
                : UnaryElementwiseArithmetic(arg0)
            {
            }

            virtual std::string get_op_class_name() const override { return "Negative"; }
            //virtual void propagate_types() override;
        };

        /// Op(X, Y)[I] = op(X[I], Y[I])
        class BinaryElementwiseBuiltin : public Builtin
        {
        protected:
            BinaryElementwiseBuiltin(const std::shared_ptr<Node>& arg0,
                                     const std::shared_ptr<Node>& arg1)
                : Builtin(Nodes{arg0, arg1})
            {
            }
            virtual const element::Type& infer_result_element_type(
                                            const element::Type& arg0_element_type,
                                            const element::Type& arg1_element_type) const;

        public:
            virtual void propagate_types() override;
        };

        class BinaryElementwiseComparison : public BinaryElementwiseBuiltin
        {
        public:
            BinaryElementwiseComparison(
                const std::shared_ptr<Node>& arg0, const std::shared_ptr<Node>& arg1)
                : BinaryElementwiseBuiltin(arg0, arg1)
            {
            }

            virtual std::string get_op_class_name() const override { return "BinaryElementwiseComparison"; }
            //virtual void propagate_types() override;
            virtual const element::Type& infer_result_element_type(
                                             const element::Type& arg0_element_type,
                                             const element::Type& arg1_element_type) const override;
        };

        class Equal : public BinaryElementwiseComparison
        {
        public:
            Equal(const std::shared_ptr<Node>& arg0, const std::shared_ptr<Node>& arg1)
                : BinaryElementwiseComparison(arg0, arg1)
            {
            }

            virtual std::string get_op_class_name() const override { return "Equal"; }
            //virtual void propagate_types() override;
        };

        class Greater : public BinaryElementwiseComparison
        {
        public:
            Greater(const std::shared_ptr<Node>& arg0, const std::shared_ptr<Node>& arg1)
                : BinaryElementwiseComparison(arg0, arg1)
            {
            }

            virtual std::string get_op_class_name() const override { return "Greater"; }
            //virtual void propagate_types() override;
        };

        class Less : public BinaryElementwiseComparison
        {
        public:
            Less(const std::shared_ptr<Node>& arg0, const std::shared_ptr<Node>& arg1)
                : BinaryElementwiseComparison(arg0, arg1)
            {
            }

            virtual std::string get_op_class_name() const override { return "Less"; }
            //virtual void propagate_types() override;
        };

        class BinaryElementwiseArithmetic : public BinaryElementwiseBuiltin
        {
        public:
            BinaryElementwiseArithmetic(
                const std::shared_ptr<Node>& arg0, const std::shared_ptr<Node>& arg1)
                : BinaryElementwiseBuiltin(arg0, arg1)
            {
            }

            virtual std::string get_op_class_name() const override { return "BinaryElementwiseArithmetic"; }
            //virtual void propagate_types() override;
            virtual const element::Type& infer_result_element_type(
                                             const element::Type& arg0_element_type,
                                             const element::Type& arg1_element_type) const override;
        };

        class Maximum : public BinaryElementwiseArithmetic
        {
        public:
            Maximum(const std::shared_ptr<Node>& arg0, const std::shared_ptr<Node>& arg1)
                : BinaryElementwiseArithmetic(arg0, arg1)
            {
            }

            virtual std::string get_op_class_name() const override { return "Max"; }
            //virtual void propagate_types() override;
        };

        class Minimum : public BinaryElementwiseArithmetic
        {
        public:
            Minimum(const std::shared_ptr<Node>& arg0, const std::shared_ptr<Node>& arg1)
                : BinaryElementwiseArithmetic(arg0, arg1)
            {
            }

            virtual std::string get_op_class_name() const override { return "Min"; }
            //virtual void propagate_types() override;
        };

        class Power : public BinaryElementwiseArithmetic
        {
        public:
            Power(const std::shared_ptr<Node>& arg0, const std::shared_ptr<Node>& arg1)
                : BinaryElementwiseArithmetic(arg0, arg1)
            {
            }

            virtual std::string get_op_class_name() const override { return "Power"; }
            //virtual void propagate_types() override;
        };

        class Remainder : public BinaryElementwiseArithmetic
        {
        public:
            Remainder(const std::shared_ptr<Node>& arg0, const std::shared_ptr<Node>& arg1)
                : BinaryElementwiseArithmetic(arg0, arg1)
            {
            }

            virtual std::string get_op_class_name() const override { return "Remainder"; }
            //virtual void propagate_types() override;
        };
    }
}
