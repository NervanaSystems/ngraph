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

#include "ngraph/axis_vector.hpp"
#include "ngraph/node_vector.hpp"
#include "ngraph/op/util/requires_tensor_view_args.hpp"

namespace ngraph
{
    namespace op
    {
        /// \brief Tensor reshape operation.
        ///
        /// "Converts" an input tensor into a new shape with the same number of elements.
        ///
        /// Given that the input tensor has shape \f$[d_1,\dots,d_n]\f$, the output may have any shape \f$[d'_1,\dots,d'_m]\f$ such that
        /// \f$\Pi_{0 \leq i \lt n}(d_i) = \Pi_{0 \leq i \lt m}(d'_i)\f$. For example, a \f$3\times{}4\f$ matrix can be reshaped into a
        /// 3-tensor of shape \f$3\times{}2\times{}2\f$, a matrix of shape \f$6\times{}2\f$, or a vector of size \f$12\f$, but not, for
        /// example, a matrix of size \f$4\times{}4\f$.
        ///
        /// The parameter `input_order` indicates the order in which to "walk" over the input axes. Given a tensor of shape \f$(d_1,\dots,d_n)\f$,
        /// an input order of \f$(a_0, a_1, \dots, a_{n-1})\f$ results in the coordinate for axis \f$a_{n-1}\f$ being varied most frequently,
        /// followed by axis \f$a-2\f$, and so on down to \f$a_0\f$.
        ///
        /// (TODO: example.)
        ///
        /// ## Parameters
        ///
        /// |                | Description                                                |
        /// | -------------- | ---------------------------------------------------------- |
        /// | `input_order`  | The order in which to walk over the input axes.            |
        /// | `output_shape` | The shape \f$[d'_1,\dots,d'_m]\f$ for the reshaped tensor. |
        ///
        /// ## Inputs
        ///
        /// |       | Type                              | Description                                                                                                  |
        /// | ----- | --------------------------------- | ------------------------------------------------------------------------------------------------------------ |
        /// | `arg` | \f$E[d_1,\dots,d_n]~(n \geq 0)\f$ | An input tensor of any type and shape, as long as the product of \f$d_i\f$ equals the product of \f$d'_i\f$. |
        ///
        /// ## Output
        ///
        /// | Type                     | Description                                                                                            |
        /// | ------------------------ | ------------------------------------------------------------------------------------------------------ |
        /// | \f$E[d'_1,\dots,d'_m]\f$ | The tensor \f$T\f$, where \f$T\f$ is the input tensor with its elements rearranged as described above. |
        class Reshape : public util::RequiresTensorViewArgs
        {
        public:
            /// \brief Constructs a reshape operation.
            ///
            /// \param arg The tensor view to be reshaped.
            /// \param input_order The order in which to iterate over input axes. This must be a permutation of the
            ///                    sequence \f$(0,\dots,n-1)\f$ where \f$n\f$ is the rank of the input tensor.
            /// \param output_shape The output shape. If the input shape is \f$(a_0,\dots,a_{k-1})\f$ then the output shape must
            ///        be of the form \f$(b_0,\dots,b_{j-1})\f$ where \f$\Pi(a_i) = \Pi(b_i)\f$.
            Reshape(const std::shared_ptr<Node>& arg,
                    const AxisVector& input_order,
                    const Shape& output_shape);

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

            /// \return The order in which to iterate over input axes.
            const AxisVector& get_input_order() const { return m_input_order; }
            /// \return The shape of the output tensor.
            const Shape& get_output_shape() const { return m_output_shape; }
        protected:
            virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                           const NodeVector& deltas) override;

            const AxisVector m_input_order;
            const Shape m_output_shape;
        };
    }
}
