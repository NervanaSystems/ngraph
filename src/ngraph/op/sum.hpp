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

#include "ngraph/axis_set.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/op/util/arithmetic_reduction.hpp"
#include "ngraph/op/util/requires_tensor_view_args.hpp"

namespace ngraph
{
    namespace op
    {
        /// \brief Tensor sum operation.
        ///
        /// Element-wise sums the input tensor, eliminating the specified reduction axes.
        /// For example:
        ///
        /// \f[
        ///     \mathit{sum}\left(\{0\},
        ///         \left[ \begin{array}{ccc}
        ///                1 & 2 \\
        ///                3 & 4 \\
        ///                5 & 6 \end{array} \right]\right) =
        ///     \left[ (1 + 3 + 5), (2 + 4 + 6) \right] =
        ///     \left[ 9, 12 \right]~~~\text{(dimension 0 (rows) is eliminated)}
        /// \f]
        ///
        /// \f[
        ///     \mathit{sum}\left(\{1\},
        ///         \left[ \begin{array}{ccc}
        ///                1 & 2 \\
        ///                3 & 4 \\
        ///                5 & 6 \end{array} \right]\right) =
        ///     \left[ (1 + 2), (3 + 4), (5 + 6) \right] =
        ///     \left[ 3, 7, 11 \right]~~~\text{(dimension 1 (columns) is eliminated)}
        /// \f]
        ///
        /// \f[
        ///     \mathit{sum}\left(\{0,1\},
        ///         \left[ \begin{array}{ccc}
        ///                1 & 2 \\
        ///                3 & 4 \\
        ///                5 & 6 \end{array} \right]\right) =
        ///      (1 + 2) + (3 + 4) + (5 + 6) =
        ///      21~~~\text{(both dimensions (rows and columns) are eliminated)}
        /// \f]
        ///
        /// This is equivalent to Reduce where `arg_init` = 0 and `reduction_function` is \f$f(x,y) = x+y\f$.
        ///
        /// ## Parameters
        ///
        /// |                      | Description                              |
        /// | -------------------- | ---------------------------------------- |
        /// | `reduction_axes`     | The axes to eliminate through summation. |
        ///
        /// ## Inputs
        ///
        /// |       | Type                              | Description                                            |
        /// | ----- | --------------------------------- | ------------------------------------------------------ |
        /// | `arg` | \f$N[d_1,\dots,d_n]~(n \geq 0)\f$ | An input tensor of any shape and numeric element type. |
        ///
        /// ## Output
        ///
        /// | Type                                      | Description                                                                                                      |
        /// | ----------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
        /// | \f$N[\textit{delete}(A,d_1,\dots,d_n)]\f$ | The tensor \f$T\f$, where \f$T\f$ is the input tensor with the `reduction_axes` \f$A\f$ eliminated by summation. |
        class Sum : public util::ArithmeticReduction
        {
        public:
            /// \brief Constructs a summation operation.
            ///
            /// \param arg The tensor view to be summed.
            /// \param reduction_axes The axis positions (0-based) to be eliminated.
            Sum(const std::shared_ptr<Node>& arg, const AxisSet& reduction_axes);

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

            /// \return The default value for Sum.
            virtual std::shared_ptr<Node> get_default_value() const override
            {
                return ngraph::make_constant_from_string("0", get_element_type(), get_shape());
            }

        protected:
            virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                           const NodeVector& deltas) override;
        };
    }
}
