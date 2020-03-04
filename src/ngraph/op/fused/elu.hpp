//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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
        namespace v0
        {
            /// \brief Exponential Linear Unit
            /// x <  0 => f(x) = alpha * (exp(x) - 1.)
            /// x >= 0 => f(x) = x
            ///
            class NGRAPH_API Elu : public ngraph::op::util::FusedOp
            {
            public:
                static constexpr NodeTypeInfo type_info{"Elu", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                Elu() = default;
                /// \brief Constructs an Elu operation.
                ///
                /// \param data Input tensor
                /// \param alpha Multiplier for negative values
                Elu(const Output<Node>& data, const double alpha);

                bool visit_attributes(AttributeVisitor& visitor) override;
                virtual NodeVector decompose_op() const override;

                virtual std::shared_ptr<Node>
                    copy_with_new_args(const NodeVector& new_args) const override;

                double get_alpha() const { return m_alpha; }
            private:
                double m_alpha;
            };
        }
        using v0::Elu;
    } // namespace op
} // namespace ngraph
