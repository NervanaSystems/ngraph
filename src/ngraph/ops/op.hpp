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
#include "ngraph/types/type.hpp"

namespace ngraph
{
    // TODO: These class definitions are to be moved into separate files in the op directory
    namespace op
    {
        /// \brief Abstract base class for builtin (primitive) operations.
        class Builtin : public Node
        {
        public:
            virtual std::string description() const override { return "Builtin"; }
        protected:
            /// \brief Constructs a builtin operation.
            ///
            /// \param args The nodes producing this node's input tensors.
            Builtin(const std::vector<std::shared_ptr<Node>>& args)
                : Node(args)
            {
            }
        };

        /// \brief Abstract base class for elementwise unary operations, i.e., operations where the same
        ///        scalar operation is applied to each element.
        ///
        /// For example, if the underlying operation (determined by the subclass) is \f$\mathit{op}(x)\f$, the input tensor
        /// \f$[[x,y],[z,w]]\f$ will be mapped to \f$[[\mathit{op}(x),\mathit{op}(y)],[\mathit{op}(z),\mathit{op}(w)]]\f$.
        ///
        /// ## Inputs
        ///
        /// |       | Type                              | Description                                                                                                                                   |
        /// | ----- | --------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
        /// | `arg` | \f$E[d_1,\dots,d_n]~(n \geq 0)\f$ | A tensor of any shape. Subclasses may impose restrictions on the element type \f$E\f$ (see UnaryElementwiseBuiltin::propagate_element_types). |
        ///
        /// ## Output
        ///
        /// | Type                    | Description                                                                                                                                                                                                                                                            |
        /// | ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
        /// | \f$E'[d_1,\dots,d_n]\f$ | The tensor \f$T\f$, where \f$T[i_1,\dots,i_n] = \mathit{op}(\texttt{arg}[i_1,\dots,i_n])\f$. This will always have the same shape as the input tensor, but subclasses must determine the element type \f$E'\f$ (see UnaryElementwiseBuiltin::propagate_element_types). |
        class UnaryElementwiseBuiltin : public Builtin
        {
        protected:
            /// \brief Constructs a unary elementwise builtin operation.
            ///
            /// \param arg Node that produces the input tensor.
            UnaryElementwiseBuiltin(const std::shared_ptr<Node>& arg)
                : Builtin(Nodes{arg})
            {
            }

            /// \brief Propagate element type from input to output.
            ///
            /// Subclasses must override this method to:
            ///
            /// 1. Verify that `arg_element_type` is valid, throwing an ngraph_error message if not; and
            /// 2. Infer and return the element type for the return tensor.
            ///
            /// \param arg_element_type The element type of the input tensor.
            /// \return The inferred element type for the output tensor.
            virtual const element::Type&
                propagate_element_types(const element::Type& arg_element_type) const = 0;

        public:
            virtual void propagate_types() override;
        };

        /// \brief Abstract base class for elementwise unary arithmetic operations, i.e., operations where the same
        ///        scalar arithmetic operation is applied to each element.
        ///
        /// For example, if the underlying operation (determined by the subclass) is \f$\mathit{op}(x)\f$, the input tensor
        /// \f$[[x,y],[z,w]]\f$ will be mapped to \f$[[\mathit{op}(x),\mathit{op}(y)],[\mathit{op}(z),\mathit{op}(w)]]\f$.
        ///
        /// ## Inputs
        ///
        /// |       | Type                              | Description                                                              |
        /// | ----- | --------------------------------- | ------------------------------------------------------------------------ |
        /// | `arg` | \f$N[d_1,\dots,d_n]~(n \geq 0)\f$ | A tensor of any shape. The element type \f$N\f$ may be any numeric type. |
        ///
        /// ## Output
        ///
        /// | Type                   | Description                                                                                                                                                             |
        /// | ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
        /// | \f$N[d_1,\dots,d_n]\f$ | The tensor \f$T\f$, where \f$T[i_1,\dots,i_n] = \mathit{op}(\texttt{arg}[i_1,\dots,i_n])\f$. This will always have the same shape and element type as the input tensor. |
        class UnaryElementwiseArithmetic : public UnaryElementwiseBuiltin
        {
        protected:
            /// \brief Constructs a unary elementwise builtin arithmetic operation.
            ///
            /// \param arg Node that produces the input tensor.
            UnaryElementwiseArithmetic(const std::shared_ptr<Node>& arg)
                : UnaryElementwiseBuiltin({arg})
            {
            }

            /// \brief Propagate element type from input to output.
            ///
            /// Throws ngraph_error if the input type is not numeric, and returns the input type.
            ///
            /// \param arg_element_type The element type of the input tensor.
            /// \return The inferred element type (same as arg_element_type).
            virtual const element::Type&
                propagate_element_types(const element::Type& arg_element_type) const final override;
        };

        /// \brief Abstract base class for elementwise binary operations, i.e., operations where the same
        ///        scalar binary operation is applied to each corresponding pair of elements in two same-shaped
        ///        input tensors.
        ///
        /// For example, if the underlying operation (determined by the subclass) is \f$\mathit{op}(x,y)\f$, the input tensors
        /// \f$[[x_0,y_0],[z_0,w_0]]\f$ and \f$[[x_1,y_1],[z_1,w_1]]\f$ will be mapped to \f$[[\mathit{op}(x_0,x_1),\mathit{op}(y_0,y_1)],[\mathit{op}(z_0,z_1),\mathit{op}(w_0,w_1)]]\f$.
        ///
        /// ## Inputs
        ///
        /// |        | Type                                | Description                                                                                                                                                     |
        /// | ------ | ----------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
        /// | `arg0` | \f$E_0[d_1,\dots,d_n]~(n \geq 0)\f$ | A tensor of any shape. Subclasses may impose restrictions on the element type \f$E_0\f$ (see BinaryElementwiseBuiltin::propagate_element_types).                |
        /// | `arg1` | \f$E_1[d_1,\dots,d_n]~(n \geq 0)\f$ | A tensor of the same shape as `arg0`. Subclasses may impose restrictions on the element type \f$E_1\f$ (see BinaryElementwiseBuiltin::propagate_element_types). |
        ///
        /// ## Output
        ///
        /// | Type                     | Description                                                                                                                                                                                                                                                                                             |
        /// | ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
        /// | \f$E_2[d_1,\dots,d_n]\f$ | The tensor \f$T\f$, where \f$T[i_1,\dots,i_n] = \mathit{op}(\texttt{arg0}[i_1,\dots,i_n],\texttt{arg1}[i_1,\dots,i_n])\f$. This will always have the same shape as the input tensors, but subclasses must determine the element type \f$E_2\f$ (see BinaryElementwiseBuiltin::propagate_element_types). |
        class BinaryElementwiseBuiltin : public Builtin
        {
        protected:
            /// \brief Constructs a biary elementwise builtin operation.
            ///
            /// \param arg0 Node that produces the first input tensor.
            /// \param arg1 Node that produces the second input tensor.
            BinaryElementwiseBuiltin(const std::shared_ptr<Node>& arg0,
                                     const std::shared_ptr<Node>& arg1)
                : Builtin(Nodes{arg0, arg1})
            {
            }

            /// \brief Propagate element type from inputs to output.
            ///
            /// Subclasses must override this method to:
            ///
            /// 1. Verify that `arg0_element_type` and `arg1_element_type` are valid, throwing an ngraph_error message if not; and
            /// 2. Infer and return the element type for the return tensor.
            ///
            /// \param arg0_element_type The element type of the first input tensor.
            /// \param arg1_element_type The element type of the second input tensor.
            /// \return The inferred element type for the output tensor.
            virtual const element::Type&
                propagate_element_types(const element::Type& arg0_element_type,
                                        const element::Type& arg1_element_type) const = 0;

        public:
            virtual void propagate_types() override;
        };

        /// \brief Abstract base class for elementwise binary comparison operations, i.e., operations where the same
        ///        scalar binary comparison operation is applied to each corresponding pair of elements in two same-shaped
        ///        input tensors.
        ///
        /// For example, if the underlying comparison operation (determined by the subclass) is \f$\mathit{op}(x,y)\f$, the input tensors
        /// \f$[[x_0,y_0],[z_0,w_0]]\f$ and \f$[[x_1,y_1],[z_1,w_1]]\f$ will be mapped to \f$[[\mathit{op}(x_0,x_1),\mathit{op}(y_0,y_1)],[\mathit{op}(z_0,z_1),\mathit{op}(w_0,w_1)]]\f$.
        ///
        /// ## Inputs
        ///
        /// |        | Type                              | Description                                            |
        /// | ------ | --------------------------------- | ------------------------------------------------------ |
        /// | `arg0` | \f$E[d_1,\dots,d_n]~(n \geq 0)\f$ | A tensor of any shape and element type.                |
        /// | `arg1` | \f$E[d_1,\dots,d_n]~(n \geq 0)\f$ | A tensor of the same shape and element type as `arg0`. |
        ///
        /// ## Output
        ///
        /// | Type                               | Description                                                                                                                                                                                                        |
        /// | ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
        /// | \f$\texttt{bool}[d_1,\dots,d_n]\f$ | The tensor \f$T\f$, where \f$T[i_1,\dots,i_n] = \mathit{op}(\texttt{arg0}[i_1,\dots,i_n],\texttt{arg1}[i_1,\dots,i_n])\f$. This will always have the same shape as the input tensors, and the element type `bool`. |
        class BinaryElementwiseComparison : public BinaryElementwiseBuiltin
        {
        public:
            /// \brief Constructs a biary elementwise builtin comparison operation.
            ///
            /// \param arg0 Node that produces the first input tensor.
            /// \param arg1 Node that produces the second input tensor.
            BinaryElementwiseComparison(const std::shared_ptr<Node>& arg0,
                                        const std::shared_ptr<Node>& arg1)
                : BinaryElementwiseBuiltin(arg0, arg1)
            {
            }

            virtual std::string description() const override
            {
                return "BinaryElementwiseComparison";
            }

            /// \brief Propagate element type from inputs to output.
            ///
            /// Throws ngraph_error if the input types are not the same, and returns element::Bool::element_type().
            ///
            /// \param arg0_element_type The element type of the first input tensor.
            /// \param arg1_element_type The element type of the second input tensor.
            /// \return The inferred element type for the output tensor, which is always element::Bool::element_type().
            virtual const element::Type&
                propagate_element_types(const element::Type& arg0_element_type,
                                        const element::Type& arg1_element_type) const override;
        };

        /// \brief Abstract base class for elementwise binary arithmetic operations, i.e., operations where the same
        ///        scalar binary arithmetic operation is applied to each corresponding pair of elements in two same-shaped
        ///        input tensors.
        ///
        /// For example, if the underlying arithmetic operation (determined by the subclass) is \f$\mathit{op}(x,y)\f$, the input tensors
        /// \f$[[x_0,y_0],[z_0,w_0]]\f$ and \f$[[x_1,y_1],[z_1,w_1]]\f$ will be mapped to \f$[[\mathit{op}(x_0,x_1),\mathit{op}(y_0,y_1)],[\mathit{op}(z_0,z_1),\mathit{op}(w_0,w_1)]]\f$.
        ///
        /// ## Inputs
        ///
        /// |        | Type                              | Description                                                              |
        /// | ------ | --------------------------------- | ------------------------------------------------------------------------ |
        /// | `arg0` | \f$N[d_1,\dots,d_n]~(n \geq 0)\f$ | A tensor of any shape. The element type \f$N\f$ may be any numeric type. |
        /// | `arg1` | \f$N[d_1,\dots,d_n]~(n \geq 0)\f$ | A tensor of the same shape and element type as `arg0`.                   |
        ///
        /// ## Output
        ///
        /// | Type                   | Description                                                                                                                                                                                            |
        /// | ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
        /// | \f$N[d_1,\dots,d_n]\f$ | The tensor \f$T\f$, where \f$T[i_1,\dots,i_n] = \mathit{op}(\texttt{arg0}[i_1,\dots,i_n],\texttt{arg1}[i_1,\dots,i_n])\f$. This will always have the same shape and element type as the input tensors. |
        class BinaryElementwiseArithmetic : public BinaryElementwiseBuiltin
        {
        public:
            /// \brief Constructs a binary elementwise builtin arithmetic operation.
            ///
            /// \param arg0 Node that produces the first input tensor.
            /// \param arg1 Node that produces the second input tensor.
            BinaryElementwiseArithmetic(const std::shared_ptr<Node>& arg0,
                                        const std::shared_ptr<Node>& arg1)
                : BinaryElementwiseBuiltin(arg0, arg1)
            {
            }

            virtual std::string description() const override
            {
                return "BinaryElementwiseArithmetic";
            }

            /// \brief Propagate element type from inputs to output.
            ///
            /// Throws ngraph_error if the input types are not the same or are not numeric, and returns the same element type as the input types.
            ///
            /// \param arg0_element_type The element type of the first input tensor.
            /// \param arg1_element_type The element type of the second input tensor.
            /// \return The inferred element type for the output tensor, which is always the same as that of the input tensors.
            virtual const element::Type& propagate_element_types(
                const element::Type& arg0_element_type,
                const element::Type& arg1_element_type) const final override;
        };
    }
}
