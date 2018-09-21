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

#include "ngraph/node_vector.hpp"
#include "ngraph/op/batch_norm.hpp"

#include "ngraph/frontend/onnx_import/exceptions.hpp"
#include "ngraph/frontend/onnx_import/op/batch_norm.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            NodeVector batch_norm(const Node& node)
            {
                NodeVector inputs{node.get_ng_inputs()};
                auto x = inputs.at(0);
                auto scale = inputs.at(1);
                auto bias = inputs.at(2);
                std::shared_ptr<ngraph::Node> mean{nullptr};
                std::shared_ptr<ngraph::Node> var{nullptr};

                int is_test{node.get_attribute_value<int>("is_test", 1)};
                int spatial{node.get_attribute_value<int>("spatial", 1)};
                double epsilon{node.get_attribute_value<double>("epsilon", 1e-5)};
                // TODO: Implement learning mode support
                // float momentum{node.get_attribute_value<float>("momentum", 0.9f)};
                bool training = false;

                ASSERT_IS_SUPPORTED(node, is_test) << "only 'is_test' mode is supported.";
                ASSERT_IS_SUPPORTED(node, spatial) << "only 'spatial' mode is supported.";

                if (inputs.size() >= 5)
                {
                    mean = inputs.at(3);
                    var = inputs.at(4);
                    return {std::make_shared<ngraph::op::BatchNorm>(
                        epsilon, scale, bias, x, mean, var, training)};
                }

                return {std::make_shared<ngraph::op::BatchNorm>(epsilon, scale, bias, x)};
            }

        } // namespace  op

    } // namespace  onnx_import

} // namespace  ngraph
