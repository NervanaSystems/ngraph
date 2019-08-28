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

#include "core/node.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace matmul
        {
            /// \brief  Factory class which generates an nGraph sub-graph based on an ONNX MatMul
            ///         operation.
            ///
            /// \note
            /// The sub-graph is needed to adjust nGraph's Dot operation semantics to semantics
            /// expected by ONNX, which are modeled on NumPy's "stacks of arrays" approach.
            /// Differences are apparent with matrices of rank > 2.
            ///
            /// This default implementation `MatmulFactory` creates a `MatMul` operation for
            /// floating-point data. Subclasses: `QLinearMatmulFactory` and `MatmulIntegerFactory`
            /// implement quantized versions.
            class MatmulFactory
            {
            public:
                explicit MatmulFactory(const Node& node)
                    : m_onnx_node(node)
                    , m_inputs(node.get_ng_inputs())
                {
                }

                virtual ~MatmulFactory() = default;

                /// \brief Create a sub-graph representing an ONNX MatMul operation.
                ///
                /// \return NodeVector containing the sub-graph output node.
                virtual NodeVector make_matmul_op();

                /// \return Node representing the left operand.
                virtual std::shared_ptr<ngraph::Node> get_left();

                /// \return Node representing the right operand.
                virtual std::shared_ptr<ngraph::Node> get_right();

                /// \return Node representing the nGraph Dot operation used to construct MatMul.
                virtual std::shared_ptr<ngraph::Node>
                    make_dot(const std::shared_ptr<ngraph::Node>& left,
                             const std::shared_ptr<ngraph::Node>& right);

            protected:
                const Node& m_onnx_node;
                const NodeVector m_inputs;
            };

            /// \brief  Factory class which generates an nGraph sub-graph based on an ONNX
            ///         QLinearMatMul operation.
            class QLinearMatmulFactory : public MatmulFactory
            {
            public:
                explicit QLinearMatmulFactory(const Node& node)
                    : MatmulFactory(node)
                {
                }

                std::shared_ptr<ngraph::Node> get_right() override;
                std::shared_ptr<ngraph::Node>
                    make_dot(const std::shared_ptr<ngraph::Node>& left,
                             const std::shared_ptr<ngraph::Node>& right) override;
            };

            /// \brief  Factory class which generates an nGraph sub-graph based on an ONNX
            ///         MatMulInteger operation.
            class MatmulIntegerFactory : public MatmulFactory
            {
            public:
                explicit MatmulIntegerFactory(const Node& node)
                    : MatmulFactory(node)
                {
                }

                std::shared_ptr<ngraph::Node>
                    make_dot(const std::shared_ptr<ngraph::Node>& left,
                             const std::shared_ptr<ngraph::Node>& right) override;
            };
        }
    }
}
