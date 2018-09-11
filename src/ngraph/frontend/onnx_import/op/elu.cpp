//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
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

#include "ngraph/node.hpp"

#include "ngraph/op/add.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/exp.hpp"
#include "ngraph/op/maximum.hpp"
#include "ngraph/op/minimum.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/subtract.hpp"

#include "utils/broadcasting.hpp"

#include "elu.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            NodeVector elu(const Node& node)
            {
                auto data = node.get_ng_inputs().at(0);
                double alpha = node.get_attribute_value<double>("alpha", 1);

                std::shared_ptr<ngraph::Node> alpha_node = std::make_shared<ngraph::op::Constant>(
                    data->get_element_type(), Shape{}, std::vector<double>{alpha});
                alpha_node = make_broadcast_node(alpha_node, data->get_shape());

                std::shared_ptr<ngraph::Node> zero_node = std::make_shared<ngraph::op::Constant>(
                    data->get_element_type(), Shape{}, std::vector<double>{0});
                zero_node = make_broadcast_node(zero_node, data->get_shape());

                return {std::make_shared<ngraph::op::Maximum>(data, zero_node) +
                        alpha_node * std::make_shared<ngraph::op::Exp>(
                                         std::make_shared<ngraph::op::Minimum>(data, zero_node)) -
                        alpha_node};
            }

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
