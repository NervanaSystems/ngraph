//*****************************************************************************
// Copyright 2017-2019 Intel Corporation
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

#include "ngraph/coordinate.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/runtime/cpu/cpu_backend_visibility.h"
#include "ngraph/strides.hpp"

namespace ngraph
{
    namespace op
    {
        // clang-format off
        /// \brief Takes two input tensors of identical rank, with the second tensor no larger than the first in any dimension, and returns a copy of
        ///        the first input tensor with the specified slice incremented by the second input tensor.
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
        /// | \f$E[d_1,\dots,d_n]\f$ | The tensor \f$T\f$ where \f$T[i_1,\dots,i_n] += \texttt{arg1}[j_1,\dots,j_n]\f$ if \f$j_1,\dots,j_n\f$ is in bounds for `arg1` and for all \f$m\f$, \f$i_m = l_m + j_m s_m\f$, otherwise \f$\texttt{arg0}[i_1,\dots,i_n]\f$. |
        // clang-format on
        class UpdateSlice : public Op
        {
        public:
            CPU_BACKEND_API
            static constexpr NodeTypeInfo type_info{"UpdateSlice", 0};
            const NodeTypeInfo& get_type_info() const override { return type_info; }
            /// \brief Constructs a tensor slice update operation.
            ///
            /// \param arg0 The tensor to overwrite into.
            /// \param arg1 The tensor to increment into `arg0`.
            /// \param lower_bounds The axiswise lower bounds of the slice (inclusive).
            /// \param upper_bounds The axiswise upper bounds of the slice (exclusive).
            /// \param strides The slicing strides; for example, strides of `{n,m}` means to take
            ///                every nth row and every mth column of `arg0` as part of the
            ///                slice to be replaced.
            UpdateSlice(const Output<Node>& arg0,
                        const Output<Node>& arg1,
                        const Coordinate& lower_bounds,
                        const Coordinate& upper_bounds,
                        const Strides& strides);

            /// \brief Constructs a tensor slice replacement operation with unit strides; i.e.,
            /// every element inside the bounding box will be overwritten.
            ///
            /// \param arg0 The tensor to overwrite into.
            /// \param arg1 The tensor to increment into `arg0`.
            /// \param lower_bounds The axiswise lower bounds of the slice (inclusive).
            /// \param upper_bounds The axiswise upper bounds of the slice (exclusive).
            UpdateSlice(const Output<Node>& arg0,
                        const Output<Node>& arg1,
                        const Coordinate& lower_bounds,
                        const Coordinate& upper_bounds);

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;
            void validate_and_infer_types() override;

            /// \return The inclusive lower-bound coordinates.
            const Coordinate& get_lower_bounds() const { return m_lower_bounds; }
            /// \return The exclusive upper-bound coordinates.
            const Coordinate& get_upper_bounds() const { return m_upper_bounds; }
            /// \return The slicing strides.
            const Strides& get_strides() const { return m_strides; }
        protected:
            Coordinate m_lower_bounds;
            Coordinate m_upper_bounds;
            Strides m_strides;
        };
    }
}
