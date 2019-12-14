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
#include <vector>

#include "ngraph/op/constant.hpp"
#include "ngraph/op/fused/selu.hpp"
#include "selu.hpp"

using namespace ngraph::op;

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                NodeVector selu(const Node& node)
                {
                    auto data = node.get_ng_inputs().at(0);
                    auto alpha =
                        node.get_attribute_value<double>("alpha", 1.67326319217681884765625);
                    auto gamma =
                        node.get_attribute_value<double>("gamma", 1.05070102214813232421875);

                    auto alpha_node = std::make_shared<ngraph::op::Constant>(
                        data->get_element_type(), data->get_shape(), std::vector<double>{alpha});

                    auto gamma_node = std::make_shared<ngraph::op::Constant>(
                        data->get_element_type(), data->get_shape(), std::vector<double>{gamma});

                    return {std::make_shared<ngraph::op::v0::Selu>(data, alpha_node, gamma_node)};
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
