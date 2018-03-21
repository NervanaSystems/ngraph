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

#include "ngraph/coordinate.hpp"
#include "ngraph/op/util/requires_tensor_view_args.hpp"
#include "ngraph/strides.hpp"

namespace ngraph
{
    namespace op
    {
        /// \brief Takes two input tensors of identical rank, with the second tensor no larger than the first in any dimension, and returns a copy of
        ///        the first input tensor with the specified slice overwritten by the second input tensor.
        ///
        /// ## Parameters
        ///
        /// |                | Description                                                                                                                                                                                          |
        /// | -------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
        /// | `lower_bounds` | The (inclusive) lower-bound coordinates \f$l_i\f$ for the tensor slice to be overwritten. For example, a lower-bound of \f$(1,2)\f$ means to start the slice at row 1 and column 2.                  |
        /// | `upper_bounds` | The (non-inclusive) upper-bound coordinates \f$u_i\f$ for the tensor slice to be overwritten. For example, an upper-bound of \f$(5,4)\f$ means to end the slice before row 4 and column 3.           |
        /// | `strides`      | The strides \f$s_i\f$ for the tensor slice to be overwritten. For example, in the matrix case, strides of \f$(1,3)\f$ means to take every row, and every third column (starting at the lower bound). |
        ///
        /// ## Inputs
        ///
        /// |        | Type                                                                           | Description                                                                                                                            |
        /// | ------ | ------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------- |
        /// | `arg0` | \f$E[d_1,\dots,d_n]~(n \geq 0)\f$                                              | A tensor of any shape and element type.                                                                                                |
        /// | `arg1` | \f$E[d'_1,\dots,d'_n]\f$ where \f$(d'_i = \lceil(u_i - l_i)\, /\, s_i\rceil\f$ | A tensor of the same element type and rank as `arg0`, whose shape is determined by the lower and upper slice bounds and slice strides. |
        ///
        /// ## Output
        ///
        /// | Type                   | Description                                                                                                                                                                                                                 |
        /// | ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
        /// | \f$E[d_1,\dots,d_n]\f$ | The tensor \f$T\f$ where \f$T[i_1,\dots,i_n] = \texttt{arg1}[j_1,\dots,j_n]\f$ if \f$j_1,\dots,j_n\f$ is in bounds for `arg1` and for all \f$m\f$, \f$i_m = l_m + j_m s_m\f$, otherwise \f$\texttt{arg0}[i_1,\dots,i_n]\f$. |
        class ReplaceSlice : public util::RequiresTensorViewArgs
        {
        public:
            /// \brief Constructs a tensor slice replacement operation.
            ///
            /// \param arg0 The tensor view to overwrite into.
            /// \param arg1 The tensor view to write into `arg0`.
            /// \param lower_bounds The axiswise lower bounds of the slice (inclusive).
            /// \param upper_bounds The axiswise upper bounds of the slice (exclusive).
            /// \param strides The slicing strides; for example, strides of `{n,m}` means to take
            ///                every nth row and every mth column of `arg0` as part of the
            ///                slice to be replaced.
            ReplaceSlice(const std::shared_ptr<Node>& arg0,
                         const std::shared_ptr<Node>& arg1,
                         const Coordinate& lower_bounds,
                         const Coordinate& upper_bounds,
                         const Strides& strides);

            /// \brief Constructs a tensor slice replacement operation with unit strides; i.e., every element inside the bounding box will be overwritten.
            ///
            /// \param arg0 The tensor view to overwrite into.
            /// \param arg1 The tensor view to write into `arg0`.
            /// \param lower_bounds The axiswise lower bounds of the slice (inclusive).
            /// \param upper_bounds The axiswise upper bounds of the slice (exclusive).
            ReplaceSlice(const std::shared_ptr<Node>& arg0,
                         const std::shared_ptr<Node>& arg1,
                         const Coordinate& lower_bounds,
                         const Coordinate& upper_bounds);

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override
            {
                if (new_args.size() != 2)
                {
                    throw ngraph_error("Incorrect number of new arguments");
                }
                return std::make_shared<ReplaceSlice>(
                    new_args.at(0), new_args.at(1), m_lower_bounds, m_upper_bounds, m_strides);
            }

            /// \return The inclusive lower-bound coordinates.
            const Coordinate& get_lower_bounds() const { return m_lower_bounds; }
            /// \return The exclusive upper-bound coordinates.
            const Coordinate& get_upper_bounds() const { return m_upper_bounds; }
            /// \return The slicing strides.
            const Strides& get_strides() const { return m_strides; }
        protected:
            virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                           const std::shared_ptr<Node>& delta) override;
            void check_args();

            const Coordinate m_lower_bounds;
            const Coordinate m_upper_bounds;
            const Strides m_strides;
        };
    }
}
