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

namespace ngraph
{
    namespace builder
    {
        /// \brief  Factory class which generates an nGraph sub-graph performing MatMul operation.
        ///
        /// This default implementation `MatmulFactory` creates a `MatMul` operation for floating-point data.
        /// Subclasses: `QLinearMatmulFactory` and `MatmulIntegerFactory` implement quantized versions.
        class MatmulFactory
        {
        public:
            explicit MatmulFactory(const OutputVector& inputs)
                : m_inputs(inputs)
            {
            }

            virtual ~MatmulFactory() = default;

            /// \brief Create a sub-graph representing an ONNX MatMul operation.
            ///
            /// \return NodeVector containing the sub-graph output node.
            virtual NodeVector make_matmul_op();

        protected:
            /// \return Node representing the left operand.
            virtual std::shared_ptr<Node> get_left();

            /// \return Node representing the right operand.
            virtual std::shared_ptr<Node> get_right();

            /// \return Node representing the nGraph Dot operation used to construct MatMul.
            virtual std::shared_ptr<Node> make_dot(const std::shared_ptr<Node>& left,
                                                   const std::shared_ptr<Node>& right);

            const OutputVector m_inputs;
        };

        /// \brief  Factory class which generates an nGraph sub-graph based on an ONNX QLinearMatMul operation.
        class QLinearMatmulFactory : public MatmulFactory
        {
        public:
            explicit QLinearMatmulFactory(const OutputVector& inputs)
                : MatmulFactory(inputs)
            {
            }

        protected:
            std::shared_ptr<Node> get_right() override;
            std::shared_ptr<Node> make_dot(const std::shared_ptr<Node>& left,
                                           const std::shared_ptr<Node>& right) override;
        };

        /// \brief  Factory class which generates an nGraph sub-graph based on an ONNX MatMulInteger operation.
        class MatmulIntegerFactory : public MatmulFactory
        {
        public:
            explicit MatmulIntegerFactory(const OutputVector& inputs)
                : MatmulFactory(inputs)
            {
            }

        protected:
            std::shared_ptr<Node> make_dot(const std::shared_ptr<Node>& left,
                                           const std::shared_ptr<Node>& right) override;
        };
    } // namespace builder
} // namespace ngraph
