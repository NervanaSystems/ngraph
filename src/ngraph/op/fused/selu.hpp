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
        namespace v1
        {
            /// \brief Performs a SELU activation function on all elements of the input node
            class Selu : public ngraph::op::util::FusedOp
            {
            public:
                NGRAPH_API
                static constexpr NodeTypeInfo type_info{"Selu", 1};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                /// \brief Constructs a Selu node.
                ///
                /// \param data - Node producing the input tensor
                /// \param alpha - Alpha coefficient of SELU operation
                /// \param lambda - Lambda coefficient of SELU operation
                Selu(const Output<Node>& data,
                     const Output<Node>& alpha,
                     const Output<Node>& lambda);

                virtual NodeVector decompose_op() const override;

                virtual std::shared_ptr<Node>
                    copy_with_new_args(const NodeVector& new_args) const override;
            };
        }
        using v1::Selu;
    } // namespace op
} // namespace ngraph
