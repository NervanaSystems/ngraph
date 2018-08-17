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
        /// \brief Takes a slice of an input tensor, i.e., the sub-tensor that resides within a bounding box, optionally with stride.
        ///
        /// Given an input tensor \f$T\f$ of shape \f$[d_1,\dots,d_n]\f$, lower bounds \f$[l_1,\dots,l_n]\f$, and upper bounds \f$[u_1,\dots,u_n]\f$,
        /// where \f$l_i \leq d_i \leq d_i\f$, and a stride \f$[s_1,\dots,s_n]\f$, returns a new tensor \f$T'\f$ of the same element type and shape
        /// \f$[d'_1,\dots,d'_n]\f$ where \f$d'_i = \lceil(u_i - l_i)\, /\, s_i\rceil\f$, where \f$T'[i_1,\dots,i_n] = T[i'_1,\dots,i'_n]\f$
        /// where \f$i'_j = i_j s_j + l_j\f$.
        ///
        /// ## Parameters
        ///
        /// |                | Description                                                                                                                                                                        |
        /// | -------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
        /// | `lower_bounds` | The (inclusive) lower-bound coordinates \f$l_i\f$ for the tensor slice. For example, a lower-bound of \f$(1,2)\f$ means to start the slice at row 1 and column 2.                  |
        /// | `upper_bounds` | The (non-inclusive) upper-bound coordinates \f$u_i\f$ for the tensor slice. For example, an upper-bound of \f$(5,4)\f$ means to end the slice before row 4 and column 3.           |
        /// | `strides`      | The strides \f$s_i\f$ for the tensor slice. For example, in the matrix case, strides of \f$(1,3)\f$ means to take every row, and every third column (starting at the lower bound). |
        ///
        /// ## Inputs
        ///
        /// |       | Type                                                | Description                             |
        /// | ----- | --------------------------------------------------- | --------------------------------------- |
        /// | `arg` | \f$E[\mathit{del}([d_1,\dots,d_n],A)]~(n \geq 0)\f$ | A tensor of any shape and element type. |
        ///
        /// ## Output
        ///
        /// | Type                                                                           | Description                       |
        /// | ------------------------------------------------------------------------------ | --------------------------------- |
        /// | \f$E[d'_1,\dots,d'_n]\f$ where \f$d'_i = \lceil(u_i - l_i)\, /\, s_i\rceil\f$. | The tensor sliced from the input. |
        class Slice : public util::RequiresTensorViewArgs
        {
        public:
            /// \brief Constructs a tensor slice operation.
            ///
            /// \param arg The tensor view to be sliced.
            /// \param lower_bounds The axiswise lower bounds of the slice (inclusive).
            /// \param upper_bounds The axiswise upper bounds of the slice (exclusive).
            /// \param strides The slicing strides; for example, strides of `{n,m}` means to take
            ///                every nth row and every mth column of the input matrix.
            Slice(const std::shared_ptr<Node>& arg,
                  const Coordinate& lower_bounds,
                  const Coordinate& upper_bounds,
                  const Strides& strides);

            /// \brief Constructs a tensor slice operation with unit strides; i.e., every element inside the bounding box will be copied to the output slice.
            ///
            /// \param arg The tensor view to be sliced.
            /// \param lower_bounds The axiswise lower bounds of the slice (inclusive).
            /// \param upper_bounds The axiswise upper bounds of the slice (exclusive).
            Slice(const std::shared_ptr<Node>& arg,
                  const Coordinate& lower_bounds,
                  const Coordinate& upper_bounds);

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

            /// \return The inclusive lower-bound coordinates.
            const Coordinate& get_lower_bounds() const { return m_lower_bounds; }
            /// \return The exclusive upper-bound coordinates.
            const Coordinate& get_upper_bounds() const { return m_upper_bounds; }
            /// \return The slicing strides.
            const Strides& get_strides() const { return m_strides; }
        protected:
            virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                           const NodeVector& deltas) override;
            void check_args();

            const Coordinate m_lower_bounds;
            const Coordinate m_upper_bounds;
            const Strides m_strides;
        };
    }
}
