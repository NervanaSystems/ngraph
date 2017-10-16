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
        /// \brief Inner product/dot product/matrix product/tensor contraction operation.
        ///
        /// Takes two arguments `arg0` and `arg1`. There are three possible cases:
        ///
        /// 1. `arg0` or `arg1` is 0-dimensional. Then, treats that 0-dimensional argument as a scalars and computes a scalar-tensor product.
        ///     (Example: `arg0` has shape `{1,2,3}` and arg1 has shape `{}`; then the result will have shape `{1,2,3}`.)
        ///
        /// 2. `arg1` is a vector (1-dimensional tensor). Then, computes a dot product reducing on the innermost (rightmost) dimensions of `arg0` and `arg1`.
        ///         (Example: arg0 has shape `{1,2,3}` and arg1 has shape `{3}`; then the result will have shape `{1,2}`.)
        ///
        /// 3. `arg1` is more than 1-dimensional. Then, computes a dot product reducing on the innermost (rightmost) dimension of arg0, and the next-to-innermost dimension of arg1.
        ///         (Example: arg0 has shape {3,4} and arg1 has shape {4,3}; then the result will have shape {3,3}.)
        ///
        ///
        /// = Case 1: Scalar-tensor product =
        ///
        /// ## Inputs
        ///
        /// |        | Type                              | Description                                                  |
        /// | ------ | --------------------------------- | ------------------------------------------------------------ |
        /// | `arg0` | \f$E[]\f$                         | A scalar of any element type.                                |
        /// | `arg1` | \f$E[d_1,\dots,d_n]~(n \geq 0)\f$ | A tensor of any shape, with the same element type as `arg0`. |
        ///
        /// <i>(Note: the order of inputs may be reversed in this case, i.e., `arg1` can be the scalar and `arg0` the tensor.)</i>
        ///
        /// ## Output
        ///
        /// | Type                   | Description                                                                                          |
        /// | ---------------------- | ---------------------------------------------------------------------------------------------------- |
        /// | \f$E[d_1,\dots,d_n]\f$ | The tensor \f$T\f$, where \f$T[i_1,\dots,i_n] = \mathtt{arg0} \cdot \mathtt{arg1}[i_1,\dots,i_n]\f$. |
        ///
        /// ## Implementation Status
        ///
        /// | Backend | Status             |
        /// | ------- | ------------------ |
        /// | NGVM    | Fully implemented. |
        ///
        /// = Case 2: Vector-tensor product =
        ///
        /// ## Inputs
        ///
        /// |        | Type                                | Description                                                                                                  |
        /// | ------ | ----------------------------------- | ------------------------------------------------------------------------------------------------------------ |
        /// | `arg0` | \f$E[d]\f$                          | A vector of any element type.                                                                                |
        /// | `arg1` | \f$E[d_1,\dots,d_n,d]~(n \geq 0)\f$ | A tensor of any shape whose innermost dimension matches `arg0`'s size, with the same element type as `arg0`. |
        ///
        /// <i>(Note: in the particular case where \f$n = 0\f$, this is a vector dot product; when \f$n = 1\f$, this is a vector-matrix product.)</i>
        //
        /// ## Output
        ///
        /// | Type                                                          | Description                                                   |
        /// | ---------------------- | ---------------------------------------------------------------------------------------------------- |
        /// | \f$E[d_1,\dots,d_n]\f$ | The tensor \f$T\f$, where \f$T[i_1,\dots,i_n] = \mathtt{arg0} \cdot \mathtt{arg1}[i_1,\dots,i_n]\f$. |
        ///
        /// ## Implementation Status
        ///
        /// | Backend | Status                                      |
        /// | ------- | ------------------------------------------- |
        /// | NGVM    | Implemented for `arg1` with rank 2 or less. |
        ///
        /// = Case 3: Tensor-tensor product =
        ///
        /// ## Inputs
        ///
        /// |        | Type                                                        | Description                                                                                                                                                  |
        /// | ------ | ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
        /// | `arg0` | \f$E[d_1,\dots,d_n]~(n \geq 1)\f$                           | A tensor of any shape with rank of at least 1, and any element type.                                                                                         |
        /// | `arg1` | \f$E[d'_1,\dots,d'_m]~(m \geq 2\text{ and }d'_{m-1}=d_n)\f$ | A tensor with the same element type as `arg0`, and any shape with rank of at least 2 whose next-to-innermost dimension matches `arg0`'s innermost dimension. |
        ///
        /// <i>(Note: in the particular case where \f$n = m = 2\f$, this is a matrix product.)</i>
        ///
        /// ## Output
        ///
        /// | Type                                                  | Description                                                                                                                       |
        /// | ----------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------|
        /// | \f$E[d_1,\dots,d_{n-1},d'_1,\dots,d'_{m-2},d'_{m}]\f$ | The tensor \f$T\f$, where \f$T[i_1,\dots,i_{n-1},j_1,\dots,j_{m-2},j_m] = \dots\f$ TODO: FIXME: finish this; but it's like numpy. |
        ///
        /// ## Implementation Status
        ///
        /// | Backend | Status                                         |
        /// | ------- | ---------------------------------------------- |
        /// | NGVM    | Implemented for `arg1` with rank of exactly 2. |
        class Dot : public Builtin
        {
        public:
            /// \brief Constructs a dot product operation.
            ///
            /// \param arg0 The node producing the first argument.
            /// \param arg1 The node producing the second argument.
            Dot(const std::shared_ptr<Node>& arg0, const std::shared_ptr<Node>& arg1)
                : Builtin({arg0, arg1})
            {
            }

            virtual std::string description() const override { return "Dot"; }
            virtual void propagate_types() override;
        };
    }
}
