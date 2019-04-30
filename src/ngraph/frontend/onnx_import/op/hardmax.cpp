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

#include "exceptions.hpp"
#include "hardmax.hpp"
#include "ngraph/op/argmax.hpp"
#include "ngraph/op/util/reshape.hpp"
#include "ngraph/frontend/onnx_import/utils/reshape.hpp"
#include "ngraph/frontend/onnx_import/utils/eye.hpp"
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
                    const auto& input = node.get_ng_inputs().at(0);
                    const auto& input_shape = input->get_shape();

                    const auto axis = node.get_attribute_value<int>("axis", 1);

                    // reshape to 2D (NxD)
                    const auto coerced_tensor = reshape::flatten(input, axis);
                    const auto& coerced_shape = coerced_tensor->get_shape();

                    const std::shared_ptr<ngraph::Node> argmax_2d = std::make_shared<ngraph::op::ArgMax>(coerced_tensor, 1, element::i64);

                    std::shared_ptr<ngraph::Node> eye_matrix;
                    if (input->get_element_type() == element::f32)
                    {
                        eye_matrix = std::dynamic_pointer_cast<ngraph::Node>(eye::square_identity<float>(coerced_shape.at(1), input->get_element_type()));
                    }
                    else if (input->get_element_type() == element::f64)
                    {
                        eye_matrix = std::dynamic_pointer_cast<ngraph::Node>(eye::square_identity<double>(coerced_shape.at(1), input->get_element_type()));
                    }
                    else
                    {
                        ASSERT_IS_SUPPORTED(input, false) << "The input tensor contains unsupported data type " << input->get_element_type();
                    }

                    auto results = std::make_shared<ngraph::op::EmbeddingLookup>(argmax_2d, eye_matrix);

                    return {ngraph::op::util::reshape(results, input_shape)};
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
