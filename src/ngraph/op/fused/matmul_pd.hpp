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
        /// \brief Operator performing Matrix Multiplication.
        class MatMulPd : public ngraph::op::util::FusedOp
        {
        public:
            NGRAPH_API
            static constexpr NodeTypeInfo type_info{"MatMulPd", 0};
            const NodeTypeInfo& get_type_info() const override { return type_info; }
            MatMulPd() = default;
            /// \brief Constructs an ScaleShift operation.
            ///
            /// \param A Matrix A
            /// \param B Matrix B
            /// \param transpose_a If matrix A should be transposed.
            /// \param transpose_b If matrix B should be transposed.
            MatMulPd(const Output<Node>& A,
                   const Output<Node>& B,
                   const bool& transpose_a = 0,
                   const bool& transpose_b = 0);

            virtual NodeVector decompose_op() const override;

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

            bool get_transpose_a() const { return m_transpose_a; }
            bool get_transpose_b() const { return m_transpose_b; }
           
        private:
            
            bool m_transpose_a;
            bool m_transpose_b;
        };




     class MatMulPdBackward : public ngraph::op::util::FusedOp
        {
        public:
        
            NGRAPH_API
            static constexpr NodeTypeInfo type_info{"MatMulPdBackward", 0};
            const NodeTypeInfo& get_type_info() const override { return type_info; }
            MatMulPdBackward() = default;
            /// \brief Constructs an ScaleShift operation.
            ///
            /// \param A Matrix A
            /// \param B Matrix B
            /// \param transpose_a If matrix A should be transposed.
            /// \param transpose_b If matrix B should be transposed.
            MatMulPdBackward(std::shared_ptr<ngraph::Node> A,
                   std::shared_ptr<ngraph::Node> B,
                   std::shared_ptr<ngraph::Node> OutGrad,
                   bool is_X, bool is_Y,
                   bool transpose_a,
                   bool transpose_b );

            virtual NodeVector decompose_op() const override;

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

             bool get_transpose_a() const { return m_transpose_a; }
             bool get_transpose_b() const { return m_transpose_b; }
             bool get_is_Y() { return m_is_Y; }
             bool get_is_X() { return m_is_X; }
        private:
            std::shared_ptr<ngraph::Node> m_A;
            std::shared_ptr<ngraph::Node> m_B;
            std::shared_ptr<ngraph::Node> OutGrad;
            bool m_is_X;
            bool m_is_Y;
            bool m_transpose_a;
            bool m_transpose_b;

            std::shared_ptr<ngraph::Node> helper_dotOp(const std::shared_ptr<ngraph::Node>& a,
                                    const std::shared_ptr<ngraph::Node>& b) const;

           std::shared_ptr<ngraph::Node> helper_reshapeToOriginal(
           std::shared_ptr<ngraph::Node> input, const ngraph::Shape& shape) const;
           std::shared_ptr<ngraph::Node> helper_transposeAndFlat3D(
           const std::shared_ptr<ngraph::Node>& input, const bool transpose,
           bool x = true) const;
           std::shared_ptr<ngraph::Node> helper_broadcast3D(
           const std::shared_ptr<ngraph::Node>& input, size_t axis0) const;
                                      

        };



    } // namespace op
} // namespace ngraph
