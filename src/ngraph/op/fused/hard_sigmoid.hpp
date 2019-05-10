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

#include "ngraph/autodiff/adjoints.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/util/fused_op.hpp"

namespace ngraph
{
    namespace op
    {
        /// \brief      Parameterized, bounded sigmoid-like, piecewise linear
        ///             function. min(max(alpha*x + beta, 0), 1)
        ///
        class HardSigmoid : public ngraph::op::util::FusedOp
        {
        public:
            /// \brief      Constructs a HardSigmoid operation.
            ///
            /// \param      data   Input tensor.
            /// \param[in]  alpha  The alpha parameter.
            /// \param[in]  beta   The beta parameter.
            ///
            HardSigmoid(const std::shared_ptr<ngraph::Node>& data, float alpha, float beta);

            virtual NodeVector decompose_op() const override;
            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

            float get_alpha() const { return m_alpha; }
            float get_beta() const { return m_beta; }
        private:
            float m_alpha;
            float m_beta;
        };
    }
}
