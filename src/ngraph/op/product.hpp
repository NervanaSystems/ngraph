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

#include "ngraph/graph_util.hpp"
#include "ngraph/op/util/arithmetic_reduction.hpp"

namespace ngraph
{
    namespace op
    {
        /// \brief Product reduction operation.
        ///
        /// Reduces the tensor, eliminating the specified reduction axes by taking the product.
        class Product : public util::ArithmeticReduction
        {
        public:
            /// \brief Constructs a product reduction operation.
            ///
            /// \param arg The tensor view to be reduced.
            /// \param reduction_axes The axis positions (0-based) to be eliminated.
            Product(const std::shared_ptr<Node>& arg, const AxisSet& reduction_axes);

            /// \return The default value for Product.
            virtual std::shared_ptr<Node> get_default_value() const override
            {
                return ngraph::make_constant_from_string("1", get_element_type(), get_shape());
            }

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;
        };
    }
}
