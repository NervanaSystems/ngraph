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

#include <memory>

#include "ngraph/op/util/unary_elementwise_arithmetic.hpp"

namespace ngraph
{
    namespace op
    {
        /// \brief Elementwise absolute value operation.
        ///
        class Abs : public util::UnaryElementwiseArithmetic
        {
        public:
            /// \brief Constructs an absolute value operation.
            ///
            /// \param arg Node that produces the input tensor.<br>
            /// `[d1, ...]`
            ///
            /// Output `[d1, ...]`
            ///
            Abs(const std::shared_ptr<Node>& arg);
            Abs(const op::Abs& other, const NodeVector& new_args);

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

        protected:
            virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                           const NodeVector& deltas) override;
        };
    }
}
