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
        /// \brief  Normalization input tensor with L2 norm.
        ///
        class NormalizeL2 : public ngraph::op::util::FusedOp
        {
        public:
            NGRAPH_API
            static const std::string type_name;
            const std::string& description() const override { return type_name; }
            ///
            /// \brief      Constructs a Normalize operation.
            ///
            /// \param      data            - Node producing the input tensor
            /// \param      scale           - Node producing the scale tensor
            /// \param      across_spatial  - Whether calculate norm across all channels.
            /// \param      channel_shared  - Whether scale is shared across channels.
            /// \param      eps             - The epsilon added to L2 norm.
            ///
            NormalizeL2(const std::shared_ptr<ngraph::Node>& data,
                      const std::shared_ptr<ngraph::Node>& scale,
                      float eps);

            float get_eps() const { return m_eps; }
            virtual NodeVector decompose_op() const override;
            virtual void pre_validate_and_infer_types() override;

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

        protected:
            float m_eps{1.f};
        };
    }
}
