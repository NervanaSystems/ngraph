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

#include <numeric>

#include "exceptions.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/fused/softmax_crossentropy.hpp"
#include "ngraph/op/multiply.hpp"
#include "softmax.hpp"
#include "utils/common.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                static ngraph::op::SoftmaxCrossEntropy::ReductionType
                    get_reduction_type(const Node& node)
                {
                    auto reduction_type = node.get_attribute_value<std::string>("type", "mean");

                    if (reduction_type == "none")
                        return ngraph::op::SoftmaxCrossEntropy::ReductionType::NONE;
                    else if (reduction_type == "sum")
                        return ngraph::op::SoftmaxCrossEntropy::ReductionType::SUM;
                    else // "mean"
                        return ngraph::op::SoftmaxCrossEntropy::ReductionType::MEAN;
                }

                static void apply_weights(std::shared_ptr<ngraph::Node> labels,
                                          std::shared_ptr<ngraph::Node> weights)
                {
                    const auto class_axis = 1;
                    const auto classes_number = labels->get_shape().at(class_axis);
                    ASSERT_VALID_ARGUMENT(weights,
                                          weights->get_shape().size() == 1 &&
                                              weights->get_shape().at(0) == classes_number)
                        << "Weights must be 1D tensor and have size equals number of classes";

                    const auto target_shape = std::make_shared<ngraph::op::Constant>(
                        element::i64,
                        Shape{weights->get_shape().size()},
                        std::vector<int64_t>(weights->get_shape().begin(),
                                             weights->get_shape().end()));
                    const auto broadcast_axis = std::make_shared<ngraph::op::Constant>(
                        element::i64, Shape{}, std::vector<int64_t>{class_axis});
                    auto broadcasted_weights = std::make_shared<ngraph::op::v1::Broadcast>(
                        weights, target_shape, broadcast_axis);
                    auto scaled_labels =
                        std::make_shared<ngraph::op::Multiply>(labels, broadcasted_weights);
                    labels = scaled_labels;
                }

                NodeVector softmax_crossentropy(const Node& node)
                {
                    NodeVector inputs{node.get_ng_inputs()};
                    auto data = inputs.at(0);
                    auto labels = inputs.at(1);
                    if (inputs.size() > 2) // optional weights input is provided
                    {
                        auto weights = inputs.at(2);
                        apply_weights(labels, weights);
                    }
                    const auto soft_label = true;
                    int64_t ignore_index = -100;
                    const auto reduction_type = get_reduction_type(node);
                    return {std::make_shared<ngraph::op::SoftmaxCrossEntropy>(
                        data, labels, soft_label, ignore_index, reduction_type)};
                }
            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
