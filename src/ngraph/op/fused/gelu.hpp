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

#include "ngraph/node.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/op/util/fused_op.hpp"

namespace ngraph
{
    namespace op
    {
        /// \brief Gaussian Error Linear Unit
        /// f(x) = 0.5 * x * (1 + erf( x / sqrt(2) )
        /// erf'(x) = 2 / sqrt(pi) * exp (-x^2)
        /// f'(x) = 0.5 * (1 + erf( x / sqrt(2)) + x * sqrt(2 / pi) * exp (-(x / sqrt(2))^2))
        ///
        class Gelu : public ngraph::op::util::FusedOp
        {
        public:
            NGRAPH_API
            static const std::string type_name;
            const std::string& description() const override { return type_name; }
            Gelu() = default;
            /// \brief Constructs an Gelu operation.
            ///
            /// \param data Input tensor
            Gelu(const Output<Node>& data);

            virtual NodeVector decompose_op() const override;

            void pre_validate_and_infer_types() override;

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;
        };
    }
}
