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

#include "exceptions.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/convert.hpp"
#include "ngraph/op/greater.hpp"
#include "ngraph/op/less.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/subtract.hpp"
#include "shrink.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                NodeVector shrink(const Node& node)
                {
                    const auto input = node.get_ng_inputs().at(0);
                    const float bias = node.get_attribute_value<float>("bias", 0.0f);
                    const float lambd = node.get_attribute_value<float>("lambd", 0.5f);

                    ASSERT_VALID_ARGUMENT(node, !(lambd < 0.0f))
                        << " The provided 'lambd' value:" << lambd << " must not be negative.";

                    std::shared_ptr<ngraph::op::Constant> negative_lambd;
                    const auto input_element_type = input->get_element_type();
                    if (input_element_type.is_signed())
                    {
                        negative_lambd = ngraph::op::Constant::create(
                            input_element_type, input->get_shape(), {-lambd});
                    }
                    else
                    {
                        // Passing -lambd to unsigned type constant will cause an overflow.
                        // For unsigned types the lowest possible value is 0.
                        negative_lambd = ngraph::op::Constant::create(
                            input_element_type, input->get_shape(), {0});
                    }

                    const auto positive_lambd = ngraph::op::Constant::create(
                        input_element_type, input->get_shape(), {lambd});

                    const auto bias_tensor = ngraph::op::Constant::create(
                        input_element_type, input->get_shape(), {bias});

                    // Create a mask indicating locations of values that need to be adjusted
                    // by adding and subtracting bias
                    // All other values indicated by 'false' in the masks need to be zeroed out
                    std::shared_ptr<ngraph::Node> values_below_neg_lambd =
                        std::make_shared<ngraph::op::Less>(input, negative_lambd);
                    std::shared_ptr<ngraph::Node> values_above_pos_lambd =
                        std::make_shared<ngraph::op::Greater>(input, positive_lambd);

                    // Convert from bool to the input type to be able to multiply adjusted inputs
                    // by the created masks
                    values_below_neg_lambd = std::make_shared<ngraph::op::Convert>(
                        values_below_neg_lambd, input_element_type);
                    values_above_pos_lambd = std::make_shared<ngraph::op::Convert>(
                        values_above_pos_lambd, input_element_type);

                    std::shared_ptr<ngraph::Node> input_minus_bias = input - bias_tensor;
                    std::shared_ptr<ngraph::Node> input_plus_bias = input + bias_tensor;

                    // multiply by the corresponding mask to zero-out the values within
                    // the <-lambd;lambd> range and keep the bias-adjusted values from outside of it
                    input_minus_bias = values_above_pos_lambd * input_minus_bias;
                    input_plus_bias = values_below_neg_lambd * input_plus_bias;

                    return {input_plus_bias + input_minus_bias};
                }

            } // namespace set_1

        } //namespace op

    } // namespace onnx_import

} // namespace ngraph
