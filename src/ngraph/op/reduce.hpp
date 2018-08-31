//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

#include "ngraph/axis_set.hpp"
#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace op
    {
        /// \brief Tensor reduction operation.
        ///
        /// Element-wise reduces the input tensor, eliminating the specified reduction axes, given a reduction function that maps two scalars to a scalar.
        /// For example, if the reduction function \f$f(x,y) = x+y\f$:
        ///
        /// \f[
        ///     \mathit{reduce}\left(f,\{0\},
        ///         \left[ \begin{array}{ccc}
        ///                1 & 2 \\ 3 & 4 \\ 5 & 6 \end{array} \right]\right) =
        ///     \left[ (1 + 3 + 5), (2 + 4 + 6) \right] =
        ///     \left[ 9, 12 \right]~~~\text{(dimension 0 (rows) is eliminated)}
        /// \f]
        ///
        /// \f[
        ///     \mathit{reduce}\left(f,\{1\},
        ///         \left[ \begin{array}{ccc}
        ///                1 & 2 \\ 3 & 4 \\ 5 & 6 \end{array} \right]\right) =
        ///     \left[ (1 + 2), (3 + 4), (5 + 6) \right] =
        ///     \left[ 3, 7, 11 \right]~~~\text{(dimension 1 (columns) is eliminated)}
        /// \f]
        ///
        /// \f[
        ///     \mathit{reduce}\left(f,\{0,1\},
        ///         \left[ \begin{array}{ccc}
        ///                1 & 2 \\ 3 & 4 \\ 5 & 6 \end{array} \right]\right) =
        ///      (1 + 2) + (3 + 4) + (5 + 6) =
        ///      21~~~\text{(both dimensions (rows and columns) are eliminated)}
        /// \f]
        ///
        /// It is assumed that \f$f\f$ is associative. In other words, the order of operations is undefined. In the case where a collapsed dimension is 0,
        /// the value of `arg_init` will be substituted.
        ///
        /// Note that the parameter `reduction_axes` specifies which axes are to be <i>eliminated</i>, which can be a bit counterintuitive. For example,
        /// as seen above, eliminating the column dimension results in the <i>rows</i> being summed, not the columns.
        ///
        /// ## Parameters
        ///
        /// |                      | Description                                                                                                               |
        /// | -------------------- | ------------------------------------------------------------------------------------------------------------------------- |
        /// | `reduction_function` | The scalar function used to reduce the input tensor. Must take two arguments of type \f$E[]\f$ and return type \f$E[]\f$. |
        /// | `reduction_axes`     | The axes to eliminate through reduction.                                                                                  |
        ///
        /// ## Inputs
        ///
        /// |                | Type                              | Description                                                                                           |
        /// | -------------- | --------------------------------- | ----------------------------------------------------------------------------------------------------- |
        /// | `arg_reductee` | \f$E[d_1,\dots,d_n]~(n \geq 0)\f$ | An input tensor of any shape, with the element type matching that expected by the reduction function. |
        /// | `arg_init`     | \f$E[]\f$                         | An scalar to be used as a substitute output value on zero-sized axes.                                 |
        ///
        /// ## Output
        ///
        /// | Type                                      | Description                                                                                                      |
        /// | ----------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
        /// | \f$E[\textit{delete}(A,d_1,\dots,d_n)]\f$ | The tensor \f$T\f$, where \f$T\f$ is the input tensor with the `reduction_axes` \f$A\f$ eliminated by reduction. |
        class Reduce : public Op
        {
        public:
            /// \brief Constructs a reduction operation.
            ///
            /// \param arg_reductee The tensor view to be reduced.
            /// \param arg_init The initial value for reduction.
            /// \param reduction_function The reduction function to use.
            /// \param reduction_axes The axis positions (0-based) to be eliminated.
            Reduce(const std::shared_ptr<Node>& arg_reductee,
                   const std::shared_ptr<Node>& arg_init,
                   const std::shared_ptr<Function>& reduction_function,
                   const AxisSet& reduction_axes);

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

            /// \return A one-element vector containing the function to use for reduction.
            std::vector<std::shared_ptr<Function>> get_functions() const override
            {
                return std::vector<std::shared_ptr<Function>>{m_reduction_function};
            }
            /// \return The axis positions (0-based) to be eliminated through reduction.
            const AxisSet& get_reduction_axes() const { return m_reduction_axes; }
        protected:
            std::shared_ptr<Function> m_reduction_function;
            AxisSet m_reduction_axes;
        };
    }
}
