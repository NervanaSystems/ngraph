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

#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace op
    {
        /// \brief One-hot operator.
        ///
        /// ## Parameters
        ///
        /// |                | Description                                                |
        /// | -------------- | ---------------------------------------------------------- |
        /// | `shape`        | The desired output shape, including the new one-hot axis.  |
        /// | `one_hot_axis` | The index within the output shape of the new one-hot axis. |
        ///
        /// ## Inputs
        ///
        /// |       | Type                                                    | Description                                 |
        /// | ----- | ------------------------------------------------------- | ------------------------------------------- |
        /// | `arg` | \f$E[d_1,\dots,d_{m-1},d_{m+1},\dots,d_n]~(n \geq 0)\f$ | A tensor of any shape and any element type. |
        ///
        /// ## Output
        ///
        /// | Type                   | Description                                                                                                                                                                                                                                                                |
        /// | ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
        /// | \f$E[d_1,\dots,d_n]\f$ | The tensor \f$T'\f$, where \f$T'[i_1,\dots,i_{m-1},i_m,i_{m+1},\dots,i_n] = 1\f$ if \f$T[i_1,\dots,i_{m-1},i_{m+1},\dots,i_n] = i_m\f$, else \f$0\f$. However, \f$T'\f$ is undefined if any non-integral value or any out-of-bounds value is detected in the input tensor. |
        class OneHot : public Op
        {
        public:
            /// \brief Constructs a one-hot operation.
            ///
            /// \param arg          Node that produces the input tensor to be one-hot encoded.
            /// \param shape        The shape of the output tensor, including the new one-hot axis.
            /// \param one_hot_axis The index within the output shape of the new one-hot axis.
            OneHot(const std::shared_ptr<Node>& arg,
                   const PartialShape& shape,
                   size_t one_hot_axis);

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

            /// \return The index of the one-hot axis.
            size_t get_one_hot_axis() const { return m_one_hot_axis; }
        protected:
            void validate_and_infer_types() override;

            PartialShape m_shape;
            size_t m_one_hot_axis;
        };
    }
}
