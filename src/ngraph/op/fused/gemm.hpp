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
        /// \brief General Matrix multiplication: https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3
        ///
        /// A' = transpose(A) if transA else A
        /// B' = transpose(B) if transB else B
        ///
        /// Compute Y = alpha * A ' * B' + beta * C
        ///
        class Gemm : public ngraph::op::util::FusedOp
        {
        public:
            /// \brief Constructs an Gemm operation.
            ///
            /// \param A, B, C Input tensors
            /// \param alpha Multiplier for negative values
            Gemm(const std::shared_ptr<ngraph::Node>& A,
                 const std::shared_ptr<ngraph::Node>& B,
                 const std::shared_ptr<ngraph::Node>& C,
                 double alpha,
                 double beta,
                 bool transA,
                 bool transB);

            virtual NodeVector decompose_op() const override;

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

        private:
            double m_alpha;
            double m_beta;
            bool m_transA;
            bool m_transB;
        };
    }
}
