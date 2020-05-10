//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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
#include "ngraph/builder/reshape.hpp"
#include "ngraph/validation_util.hpp"
#include "softmax.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace
        {
            std::shared_ptr<ngraph::Node> softmax_2D(const std::shared_ptr<ngraph::Node> data)
            {
                NGRAPH_CHECK(data->get_output_partial_shape(0).rank().same_scheme(2),
                             "The Softmax input data needs to be coerced to 2D");

                const auto axis_1 = default_opset::Constant::create(element::i32, Shape{1}, {1});
                const auto max = std::make_shared<default_opset::ReduceMax>(data, axis_1);

                // equivalent to numpy's max.reshape((-1,1))
                const auto reshape_pattern =
                    default_opset::Constant::create(element::i32, Shape{2}, {0, 1});
                const auto reshaped_max =
                    std::make_shared<default_opset::Reshape>(max, reshape_pattern, true);

                const auto data_minus_max =
                    std::make_shared<default_opset::Subtract>(data, reshaped_max);

                const auto exp = std::make_shared<default_opset::Exp>(data_minus_max);
                const auto sum_exp = std::make_shared<default_opset::ReduceSum>(exp, axis_1);
                const auto reshaped_sum_exp =
                    std::make_shared<default_opset::Reshape>(sum_exp, reshape_pattern, true);

                return std::make_shared<default_opset::Divide>(exp, reshaped_sum_exp);
            }
        }

        namespace op
        {
            namespace set_1
            {
                NodeVector softmax(const Node& node)
                {
                    const auto data = node.get_ng_inputs().at(0);
                    const auto axis = node.get_attribute_value<int64_t>("axis", 1);
                    const auto normalized_axis = ngraph::normalize_axis(
                        node.get_description(), axis, data->get_output_partial_shape(0).rank());

                    const auto coerced_data =
                        ngraph::builder::opset1::flatten(data, normalized_axis);
                    const auto coerced_softmax = softmax_2D(coerced_data);

                    if (data->get_output_partial_shape(0).is_static())
                    {
                        return {ngraph::builder::opset1::reshape(coerced_softmax,
                                                                 data->get_output_shape(0))};
                    }
                    else
                    {
                        const auto data_shape = std::make_shared<default_opset::ShapeOf>(data);
                        return {std::make_shared<default_opset::Reshape>(
                            coerced_softmax, data_shape, false)};
                    }
                }
            }
        }
    }
}
