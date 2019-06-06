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

#include "hardmax.hpp"
#include "exceptions.hpp"
#include "ngraph/builder/reshape.hpp"
#include "ngraph/frontend/onnx_import/utils/common.hpp"
#include "ngraph/op/argmax.hpp"
#include "ngraph/op/embedding_lookup.hpp"

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
                    auto axis = node.get_attribute_value<std::int64_t>("axis", 1);

                    ASSERT_VALID_ARGUMENT(node, axis >= 0 && axis < input_shape.size())
                        << "The provided axis value " << axis
                        << " does not match the input tensor dimensions";

                    // reshape to 2D - "batch size" x "input feature dimensions" (NxD)
                    const auto coerced_tensor = ngraph::builder::flatten(input, axis);
                    const auto& coerced_shape = coerced_tensor->get_shape();

                    const std::shared_ptr<ngraph::Node> argmax_2d =
                        std::make_shared<ngraph::op::ArgMax>(coerced_tensor, 1, element::i64);

                    std::shared_ptr<ngraph::Node> eye_matrix =
                        common::square_identity(coerced_shape.at(1), input->get_element_type());

                    // the results are elements of the eye_matrix indexed by argmax_2d values
                    // in other words: eye_matrix[argmax_2d]
                    auto results =
                        std::make_shared<ngraph::op::EmbeddingLookup>(argmax_2d, eye_matrix);

                    return {ngraph::builder::reshape(results, input_shape)};
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
