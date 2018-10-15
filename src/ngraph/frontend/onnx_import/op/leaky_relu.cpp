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

#include "ngraph/node.hpp"
#include "ngraph/node_vector.hpp"
#include "ngraph/shape.hpp"

#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/maximum.hpp"
#include "ngraph/op/multiply.hpp"

#include "exceptions.hpp"

#include "core/node.hpp"
#include "utils/broadcasting.hpp"

#include "leaky_relu.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                NodeVector leaky_relu(const Node& node)
                {
                    auto data = node.get_ng_inputs().at(0);
                    double alpha = node.get_attribute_value<double>("alpha", 0.01);

                    ASSERT_VALID_ARGUMENT(node, ((alpha >= 0) && (alpha <= 1)))
                        << " alpha value should be in range (0,1)";

                    std::shared_ptr<ngraph::Node> alpha_node =
                        std::make_shared<ngraph::op::Constant>(
                            data->get_element_type(), Shape{}, std::vector<double>{alpha});
                    alpha_node = make_broadcast_node(alpha_node, data->get_shape());
                    return {std::make_shared<ngraph::op::Maximum>(data * alpha_node, data)};
                }

            } // namespace set_1

        } //namespace op

    } // namespace onnx_import

} // namespace ngraph
