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
        namespace v0
        {
            /// \brief Gaussian Error Linear Unit
            /// f(x) = 0.5 * x * (1 + erf( x / sqrt(2) )
            class NGRAPH_API Gelu : public ngraph::op::util::FusedOp
            {
            public:
                static constexpr NodeTypeInfo type_info{"Gelu", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                Gelu() = default;
                /// \brief Constructs an Gelu operation.
                ///
                /// \param data Input tensor
                Gelu(const Output<Node>& data);

                virtual NodeVector decompose_op() const override;

                void pre_validate_and_infer_types() override;

                virtual std::shared_ptr<Node>
                    copy_with_new_args(const NodeVector& new_args) const override;

            protected:
                virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                               const OutputVector& deltas) override;
            };

            /// \brief Backprop for Gelu(x) is GeluBackprop(x) * delta
            class NGRAPH_API GeluBackpropFactor : public util::FusedOp
            {
            public:
                static constexpr NodeTypeInfo type_info{"GeluBackpropFactor", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                GeluBackpropFactor() = default;

                GeluBackpropFactor(const Output<Node>& x);

                virtual NodeVector decompose_op() const override;

                void pre_validate_and_infer_types() override;

                virtual std::shared_ptr<Node>
                    copy_with_new_args(const NodeVector& new_args) const override;
            };
        }
        using v0::Gelu;
        using v0::GeluBackpropFactor;
    }
}
