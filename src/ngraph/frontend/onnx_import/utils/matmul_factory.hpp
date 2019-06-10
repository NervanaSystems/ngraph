//*****************************************************************************
// Copyright 2018-2019 Intel Corporation
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
            class MatmulFactory
            {
            public:
                explicit MatmulFactory(const Node& node)
                    : m_onnx_node(node)
                    , m_inputs(node.get_ng_inputs())
                {
                }

                virtual ~MatmulFactory() = default;

                virtual NodeVector make_matmul_op();

                virtual std::shared_ptr<ngraph::Node> get_left();
                virtual std::shared_ptr<ngraph::Node> get_right();
                virtual std::shared_ptr<ngraph::Node>
                    make_dot(const std::shared_ptr<ngraph::Node>& left,
                             const std::shared_ptr<ngraph::Node>& right);

            protected:
                const Node& m_onnx_node;
                const NodeVector m_inputs;
            };

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
