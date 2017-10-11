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

#include "ngraph/ops/op.hpp"

namespace ngraph
{
    namespace op
    {
        /// \brief Concatenation operation.
        ///
        /// Given an axis index \f$a\f$ and a rank \f$r \geq 1\f$ where \f$0 \leq a \lt r\f$, and one or more \f$r\f$-tensors
        /// with the same element type whose shapes are the same except possibly at axis \f$a\f$, the tensors are
        /// concatenated along axis \f$a\f$.
        ///
        /// For example:
        ///   1. Concatenating matrices on axis 0 (the row axis) stacks the matrices from top to bottom.
        ///      The number of rows in the resulting matrix is the sum of the number of rows for each
        ///      input matrix.
        ///   2. Concatenating matrices on axis 1 (the column axis) concatenates them from left to right.
        ///      The number of columns in the resulting matrix is the sum of the number of columns for each
        ///      input matrix.
        ///   3. Concatenating 3-tensors on axis 2 (the depth axis) stacks them from front to back.
        ///      The depth of the resulting tensor is the sum of the depth of each input tensor.
        ///
        /// The resulting tensor will have the same rank as the input tensors.
        ///
        /// ## Parameters
        ///
        /// |                      | Description                                                    |
        /// | -------------------- | -------------------------------------------------------------- |
        /// | `concatenation_axis` | The axis \f$a\f$ along which to concatenate the input tensors. |
        ///
        /// ## Inputs
        ///
        /// |                 | Type                                              | Description                                                                                                              |
        /// | --------------- | ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
        /// | `args`[\f$i\f$] | \f$E[d_1,d_{a-1},d^i_a,d_{a+1},d_n]~(n \geq 1)\f$ | One or more input tensors, all of which have the same element type, and the same shape, except possibly at axis \f$a\f$. |
        ///
        /// ## Output
        ///
        /// | Type                                                          | Description                                                                                     |
        /// | ------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
        /// | \f$E'[d_1,\dots,d_{a-1},\Sigma_i(d^i_a),d_{a+1},\dots,d_n]\f$ | The tensor \f$T\f$, where \f$T\f$ is the concatenation of the input tensors along axis \f$a\f$. |
        ///
        /// ## Implementation Status
        ///
        /// | Backend | Status                                |
        /// | ------- | ------------------------------------- |
        /// | NGVM    | Implemented for vectors and matrices. |
        class Concat : public Builtin
        {
        public:
            /// \brief Constructs a concatenation operation.
            ///
            /// \param args               The nodes producing the input tensors.
            /// \param concatenation_axis The axis along which to concatenate the input tensors.
            Concat(const Nodes& args, size_t concatenation_axis)
                : Builtin(args)
                , m_concatenation_axis(concatenation_axis)
            {
            }

            virtual std::string description() const override { return "Concatenate"; }
            virtual void propagate_types() override;

            /// \return The concatenation axis.
            size_t get_concatenation_axis() const { return m_concatenation_axis; }
        protected:
            const size_t m_concatenation_axis;
        };
    }
}
