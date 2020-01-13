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

#include "argmax.hpp"
#include "core/node.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/topk.hpp"
#include "ngraph/opsets/opset0.hpp"
#include "default_opset.hpp"
#include "utils/reduction.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                NodeVector argmax(const Node& node)
                {
                    const auto axis = node.get_attribute_value<std::int64_t>("axis", 0);
                    const auto keepdims = node.get_attribute_value<std::int64_t>("keepdims", 1);

                    auto input_node = node.get_ng_inputs().at(0);
                    const auto normalized_axis = ngraph::normalize_axis(
                        node.get_description(), axis, input_node->get_shape().size());

                    //std::int64_t k = input_node->get_shape().at(normalized_axis);
                    const auto k_node = default_opset::Constant::create(ngraph::element::i64, Shape{}, { 1 });

                    const auto topk = std::make_shared<default_opset::TopK>(input_node,
                        k_node,
                        normalized_axis,
                        default_opset::TopK::Mode::MAX,
                        default_opset::TopK::SortType::NONE);

                    const auto indices = std::make_shared<ngraph::opset0::GetOutputElement>(topk, 1);

                    if (keepdims == 0)
                    {
                        auto output_shape = input_node->get_shape();
                        output_shape.erase(output_shape.begin() + normalized_axis);
                        auto reshape_node = builder::opset1::reshape(indices, output_shape);
                        auto reconvert_node =
                            std::make_shared<ngraph::op::Convert>(reshape_node, element::i64);
                        return { reconvert_node };
                    }
                    //auto convert_node = std::make_shared<ngraph::op::Convert>(indices, element::f32);

                    //auto output_shape = input_node->get_shape();
                    //output_shape.at(normalized_axis) = 1;
                    //auto reshape_node = builder::opset1::reshape(indices, output_shape);

                    // WORKAROUND FOR PROBLEMS WITH RESHAPE ON i64 @TODO: remove
                    auto reconvert_node =
                        std::make_shared<ngraph::op::Convert>(/*reshape_node*/indices, element::i64);

                    return { reconvert_node };
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
