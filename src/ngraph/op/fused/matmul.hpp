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
        class MatMul : public ngraph::op::util::FusedOp
        {
        public:
            NGRAPH_API
            static constexpr NodeTypeInfo type_info{"MatMul", 0};
            const NodeTypeInfo& get_type_info() const override { return type_info; }
            MatMul() = default;
            /// \brief Constructs an ScaleShift operation.
            ///
            /// \param A Matrix A
            /// \param B Matrix B
            /// \param transpose_a If matrix A should be transposed.
            /// \param transpose_b If matrix B should be transposed.
            MatMul(const Output<Node>& A,
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
    } // namespace op
} // namespace ngraph
