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

#include "hardmax.hpp"
#include "exceptions.hpp"
#include "ngraph/builder/reshape.hpp"
#include "ngraph/op/one_hot.hpp"
#include "ngraph/op/topk.hpp"
#include "ngraph/opsets/opset0.hpp"
#include "ngraph/validation_util.hpp"
#include "utils/common.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                NodeVector hardmax(const Node& node)
                {
                    const auto input = node.get_ng_inputs().at(0);
                    const auto& input_shape = input->get_shape();
                    const auto axis = node.get_attribute_value<std::int64_t>("axis", 1);

                    const auto normalized_axis =
                        ngraph::normalize_axis(node.get_description(), axis, input_shape.size());

                    // reshape to 2D - "batch size" x "input feature dimensions" (NxD)
                    const auto coerced_tensor =
                        ngraph::builder::opset1::flatten(input, normalized_axis);
                    const auto& coerced_shape = coerced_tensor->get_shape();
                    const auto row_size = static_cast<int64_t>(coerced_shape.at(1));

                    const auto indices_axis = 1;
                    const auto max_indices = std::make_shared<opset0::GetOutputElement>(
                        std::make_shared<default_opset::TopK>(
                            coerced_tensor,
                            default_opset::Constant::create(ngraph::element::i64, Shape{}, {1}),
                            indices_axis,
                            default_opset::TopK::Mode::MAX,
                            default_opset::TopK::SortType::NONE),
                        1);

                    const auto depth =
                        ngraph::op::Constant::create(ngraph::element::i64, Shape{}, {row_size});
                    const auto on_value =
                        ngraph::op::Constant::create(ngraph::element::i64, Shape{}, {1});
                    const auto off_value =
                        ngraph::op::Constant::create(ngraph::element::i64, Shape{}, {0});

                    const auto results = std::make_shared<default_opset::OneHot>(
                        max_indices, depth, on_value, off_value, indices_axis);
                    const auto converted_results = std::make_shared<default_opset::Convert>(
                        results, input->get_element_type());

                    return {ngraph::builder::opset1::reshape(converted_results, input_shape)};
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
