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
            /// Name of the builtin op, for debugging and logging.

            // TODO: Implement for each op. This enables graphs to be built for now.
            virtual void propagate_types() override {}

        protected:
            Builtin(const std::vector<std::shared_ptr<Node>>& args)
                : Op(args)
            {
            }
        };

        class Abs : public Builtin
        {
        public:
            Abs(const std::shared_ptr<Node>& arg0)
                : Builtin({arg0})
            {
            }

            virtual std::string get_op_class_name() const override { return "Abs"; }
            //virtual void propagate_types() override;
        };

        class Add : public Builtin
        {
        public:
            Add(const std::shared_ptr<Node>& arg0, const std::shared_ptr<Node>& arg1)
                : Builtin({arg0, arg1})
            {
            }
            virtual std::string get_op_class_name() const override { return "Add"; }
            //virtual void propagate_types() override;
        };

        class Ceiling : public Builtin
        {
        public:
            Ceiling(const std::shared_ptr<Node>& arg0, const std::shared_ptr<Node>& arg1)
                : Builtin({arg0, arg1})
            {
            }

            virtual std::string get_op_class_name() const override { return "Ceiling"; }
            //virtual void propagate_types() override;
        };

        class Divide : public Builtin
        {
        public:
            Divide(const std::shared_ptr<Node>& arg0, const std::shared_ptr<Node>& arg1)
                : Builtin({arg0, arg1})
            {
            }

            virtual std::string get_op_class_name() const override { return "Divide"; }
            //virtual void propagate_types() override;
        };

        class Equal : public Builtin
        {
        public:
            Equal(const std::shared_ptr<Node>& arg0, const std::shared_ptr<Node>& arg1)
                : Builtin({arg0, arg1})
            {
            }

            virtual std::string get_op_class_name() const override { return "Equal"; }
            //virtual void propagate_types() override;
        };

        class Exp : public Builtin
        {
        public:
            Exp(const std::shared_ptr<Node>& arg0)
                : Builtin({arg0})
            {
            }

            virtual std::string get_op_class_name() const override { return "Exp"; }
            //virtual void propagate_types() override;
        };

        class Floor : public Builtin
        {
        public:
            Floor(const std::shared_ptr<Node>& arg0, const std::shared_ptr<Node>& arg1)
                : Builtin({arg0, arg1})
            {
            }

            virtual std::string get_op_class_name() const override { return "Floor"; }
            //virtual void propagate_types() override;
        };

        class Greater : public Builtin
        {
        public:
            Greater(const std::shared_ptr<Node>& arg0, const std::shared_ptr<Node>& arg1)
                : Builtin({arg0, arg1})
            {
            }

            virtual std::string get_op_class_name() const override { return "Greater"; }
            //virtual void propagate_types() override;
        };

        class Less : public Builtin
        {
        public:
            Less(const std::shared_ptr<Node>& arg0, const std::shared_ptr<Node>& arg1)
                : Builtin({arg0, arg1})
            {
            }

            virtual std::string get_op_class_name() const override { return "Less"; }
            //virtual void propagate_types() override;
        };

        class Log : public Builtin
        {
        public:
            Log(const std::shared_ptr<Node>& arg0)
                : Builtin({arg0})
            {
            }

            virtual std::string get_op_class_name() const override { return "Log"; }
            //virtual void propagate_types() override;
        };

        class Maximum : public Builtin
        {
        public:
            Maximum(const std::shared_ptr<Node>& arg0, const std::shared_ptr<Node>& arg1)
                : Builtin({arg0, arg1})
            {
            }

            virtual std::string get_op_class_name() const override { return "Max"; }
            //virtual void propagate_types() override;
        };

        class Minimum : public Builtin
        {
        public:
            Minimum(const std::shared_ptr<Node>& arg0, const std::shared_ptr<Node>& arg1)
                : Builtin({arg0, arg1})
            {
            }

            virtual std::string get_op_class_name() const override { return "Min"; }
            //virtual void propagate_types() override;
        };

        class Multiply : public Builtin
        {
        public:
            Multiply(const std::shared_ptr<Node>& arg0, const std::shared_ptr<Node>& arg1)
                : Builtin({arg0, arg1})
            {
            }

            virtual std::string get_op_class_name() const override { return "Multiply"; }
            //virtual void propagate_types() override;
        };

        class Negative : public Builtin
        {
        public:
            Negative(const std::shared_ptr<Node>& arg0)
                : Builtin({arg0})
            {
            }

            virtual std::string get_op_class_name() const override { return "Negative"; }
            //virtual void propagate_types() override;
        };

        class Power : public Builtin
        {
        public:
            Power(const std::shared_ptr<Node>& arg0, const std::shared_ptr<Node>& arg1)
                : Builtin({arg0, arg1})
            {
            }

            virtual std::string get_op_class_name() const override { return "Power"; }
            //virtual void propagate_types() override;
        };

        class Remainder : public Builtin
        {
        public:
            Remainder(const std::shared_ptr<Node>& arg0, const std::shared_ptr<Node>& arg1)
                : Builtin({arg0, arg1})
            {
            }

            virtual std::string get_op_class_name() const override { return "Remainder"; }
            //virtual void propagate_types() override;
        };

        class Reshape : public Builtin
        {
        public:
            Reshape(const std::shared_ptr<Node>& arg0, const Shape& shape)
                : Builtin({arg0})
                , m_shape(shape)
            {
            }

            virtual std::string get_op_class_name() const override { return "Reshape"; }
            //virtual void propagate_types() override;
        protected:
            Shape m_shape;
        };

        class Subtract : public Builtin
        {
        public:
            Subtract(const std::shared_ptr<Node>& arg0, const std::shared_ptr<Node>& arg1)
                : Builtin({arg0, arg1})
            {
            }

            virtual std::string get_op_class_name() const override { return "Subtract"; }
            //virtual void propagate_types() override;
        };
    }
}
