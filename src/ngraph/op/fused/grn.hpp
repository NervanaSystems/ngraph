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

#include <memory>

#include "ngraph/node.hpp"
#include "ngraph/op/util/fused_op.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v0
        {
            /// \brief  Global Response Normalization with L2 norm (across channels only).
            ///
            class NGRAPH_API GRN : public ngraph::op::util::FusedOp
            {
            public:
                static constexpr NodeTypeInfo type_info{"GRN", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                GRN() = default;
                /// \brief      Constructs a GRN operation.
                ///
                /// \param      data  - Node producing the input tensor
                /// \param      bias  - The bias added to the variance.
                ///
                GRN(const Output<Node>& data, float bias);

                float get_bias() const { return m_bias; }
                virtual void pre_validate_and_infer_types() override;
                virtual NodeVector decompose_op() const override;

                virtual std::shared_ptr<Node>
                    copy_with_new_args(const NodeVector& new_args) const override;

            protected:
                float m_bias = 1.0f;
            };
        }
        using v0::GRN;
    }
}
