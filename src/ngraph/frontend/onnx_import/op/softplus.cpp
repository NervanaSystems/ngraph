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

#include <memory>

#include "default_opset.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/greater.hpp"
#include "ngraph/op/select.hpp"
#include "ngraph/opsets/opset0.hpp"
#include "softplus.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                NodeVector softplus(const Node& node)
                {
                    auto data = node.get_ng_inputs().at(0);

                    std::shared_ptr<ngraph::Node> zero_node =
                        std::make_shared<default_opset::Constant>(
                            data->get_element_type(), data->get_shape(), std::vector<float>{0.f});
                    std::shared_ptr<ngraph::Node> one_node =
                        std::make_shared<default_opset::Constant>(
                            data->get_element_type(), data->get_shape(), std::vector<float>{1.f});

                    std::shared_ptr<ngraph::Node> positive_val_node =
                        data + std::make_shared<default_opset::Log>(
                                   std::make_shared<default_opset::Exp>(
                                       std::make_shared<default_opset::Negative>(data)) +
                                   one_node);

                    std::shared_ptr<ngraph::Node> negative_val_node =
                        std::make_shared<default_opset::Log>(
                            std::make_shared<default_opset::Exp>(data) + one_node);

                    std::shared_ptr<ngraph::Node> condition_node =
                        std::make_shared<ngraph::opset0::Greater>(data, zero_node);

                    //
                    // This equation represents:
                    //     x + log(exp(-x) + 1) - for x > 0; to manage exponent overflow,
                    //     log(exp(x) + 1)      - elsewhere.
                    //
                    return {std::make_shared<ngraph::opset0::Select>(
                        condition_node, positive_val_node, negative_val_node)};
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
