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
        /// \brief Tensor IndexReduction operation.
        ///
        /// Element-wise comparison (e.g. argmin, argmax) returning index for the input tensor, eliminating the specified reduction axis.
        /// For example:
        ///
        /// \f[
        ///     \mathit{ArgMin}\left(\{0\},
        ///         \left[ \begin{array}{ccc}
        ///                1 & 2 \\ 3 & 4 \\ 5 & 6 \end{array} \right]\right) =
        ///     \left[ (min_index(1, 3, 5), min_index(2, 4, 6) \right] =
        ///     \left[ 0, 0 \right]~~~\text{(dimension 0 (rows) is eliminated)}
        /// \f]
        ///
        /// \f[
        ///     \mathit{ArgMin}\left(\{1\},
        ///         \left[ \begin{array}{ccc}
        ///                1 & 2 \\ 3 & 4 \\ 5 & 6 \end{array} \right]\right) =
        ///     \left[ min_index(1, 2), min_index(3, 4), min_index(5, 6) \right] =
        ///     \left[ 0, 0, 0 \right]~~~\text{(dimension 1 (columns) is eliminated)}
        /// \f]
        ///
        /// This is equivalent to Reduce where `arg_init` = 0 and `reduction_function` is \f$f(x,y) = x+y\f$.
        ///
        /// ## Parameters
        ///
        /// |                      | Description                              |
        /// | -------------------- | ---------------------------------------- |
        /// | `axis`     | The axis along which to compute an index for minimum/maximum |
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
        /// | \f$T[\textit{delete}(axis)]\f$ | The tensor \f$T\f$, where \f$T\f$ is the tensor of minimum/maxmimum indices reduced along `axis` |
        class IndexReduction : public util::RequiresTensorViewArgs
        {
        public:
            size_t get_axis() const { return m_axis; }
            bool is_int64() const { return m_is_int64; }
            /// \brief Constructs a IndexReduction operation.
            ///
            /// \param arg The tensor view to be IndexReduction.
            /// \param axis The axis along which to compute an index for minimum/maximum
            /// \param keep_dimensions keep the axis dimension.
            /// \param is_int64 produce indices in int64 or int32.
            IndexReduction(const std::string& node_type,
                           const std::shared_ptr<Node>& arg,
                           size_t axis,
                           bool keep_dimensions,
                           bool is_int64);

        protected:
            size_t m_axis;
            bool m_is_int64;
            virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                           const NodeVector& deltas) override;
        };

        class ArgMin : public IndexReduction
        {
        public:
            ArgMin(const std::shared_ptr<Node>& arg,
                   size_t axis,
                   bool keep_dimensions,
                   bool is_int64)
                : IndexReduction("ArgMin", arg, axis, keep_dimensions, is_int64)
            {
            }

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;
        };
    }
}
