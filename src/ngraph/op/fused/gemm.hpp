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
        /// \brief Operator performing General Matrix multiplication.
        ///
        /// \note More information: https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3
        ///
        /// A' = transpose(A) if transA else A
        /// B' = transpose(B) if transB else B
        ///
        /// Compute Y = alpha * A' * B' + beta * C
        ///
        class Gemm : public ngraph::op::util::FusedOp
        {
        public:
            NGRAPH_API
            static const std::string type_name;
            const std::string& description() const override { return type_name; }
            Gemm() = default;
            /// \brief Constructs an Gemm operation.
            ///
            /// \param A Input tensor A
            /// \param B Input tensor B
            /// \param C Input tensor C
            /// \param alpha Scalar multiplier for the product of input tensors A * B
            /// \param beta Scalar multiplier for input tensor C
            /// \param transA Whether A should be transposed
            /// \param transB Whether B should be transposed
            Gemm(const Output<Node>& A,
                 const Output<Node>& B,
                 const Output<Node>& C,
                 double alpha = 1.0,
                 double beta = 1.0,
                 bool transA = false,
                 bool transB = false);

            virtual NodeVector decompose_op() const override;

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

            double get_alpha() const { return m_alpha; }
            double get_beta() const { return m_beta; }
            bool get_transA() const { return m_transA; }
            bool get_transB() const { return m_transB; }
        private:
            double m_alpha;
            double m_beta;
            bool m_transA;
            bool m_transB;
        };
    }
}
