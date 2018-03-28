/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#pragma once

#include "ngraph/op/util/arithmetic_reduction.hpp"

namespace ngraph
{
    namespace op
    {
        /// \brief Max-reduction operation.
        ///
        /// Reduces the tensor, eliminating the specified reduction axes by taking the maximum element.
        ///
        /// This is equivalent to Reduce where `arg_init` = -inf and `reduction_function` is \f$f(x,y) = max(x,y)\f$.
        ///
        /// ## Parameters
        ///
        /// |                      | Description                                  |
        /// | -------------------- | -------------------------------------------- |
        /// | `reduction_axes`     | The axes to eliminate through max-reduction. |
        ///
        /// ## Inputs
        ///
        /// |       | Type                              | Description                                            |
        /// | ----- | --------------------------------- | ------------------------------------------------------ |
        /// | `arg` | \f$N[d_1,\dots,d_n]~(n \geq 0)\f$ | An input tensor of any shape and numeric element type. |
        ///
        /// ## Output
        ///
        /// | Type                                      | Description                                                                                                                       |
        /// | ----------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
        /// | \f$N[\textit{delete}(A,d_1,\dots,d_n)]\f$ | The tensor \f$T\f$, where \f$T\f$ is the input tensor with the `reduction_axes` \f$A\f$ eliminated by taking the maximum element. |
        class Max : public util::ArithmeticReduction
        {
        public:
            /// \brief Constructs a max-reduction operation.
            ///
            /// \param arg The tensor view to be reduced.
            /// \param reduction_axes The axis positions (0-based) to be eliminated.
            Max(const std::shared_ptr<Node>& arg, const AxisSet& reduction_axes)
                : ArithmeticReduction("Max", arg, reduction_axes)
            {
            }

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override
            {
                if (new_args.size() != 1)
                {
                    throw ngraph_error("Incorrect number of new arguments");
                }
                return std::make_shared<Max>(new_args.at(0), m_reduction_axes);
            }
        };
    }
}
