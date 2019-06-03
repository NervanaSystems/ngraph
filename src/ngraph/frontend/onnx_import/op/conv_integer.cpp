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

#include "op/conv_integer.hpp"
#include "ngraph/builder/quantization/quantized_linear_convolution.hpp"
#include "ngraph/frontend/onnx_import/exceptions.hpp"
#include "ngraph/frontend/onnx_import/utils/convpool.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                NodeVector conv_integer(const Node& node)
                {
                    const NodeVector& inputs = node.get_ng_inputs();
                    auto input = inputs.at(0);
                    auto filters = inputs.at(1);

                    int64_t groups{node.get_attribute_value<int64_t>("group", 1)};
                    ASSERT_VALID_ARGUMENT(node, (groups == 1))
                        << "Only value of 1 for 'group' supported for ConvInteger. Given: "
                        << groups;

                    auto window_movement_strides = convpool::get_strides(node);
                    auto window_dilation_strides = convpool::get_dilations(node);
                    auto paddings = convpool::get_pads(node);
                    const auto& padding_below = paddings.first;
                    const auto& padding_above = paddings.second;

                    const Strides default_data_dilation_strides(input->get_shape().size() - 2, 1);

                    return {ngraph::builder::quantization::QuantizedConvInteger(
                        input,
                        filters,
                        window_movement_strides,
                        window_dilation_strides,
                        padding_below,
                        padding_above,
                        default_data_dilation_strides)};
                }
            } // namespace set_1

        } //namespace op

    } // namespace onnx_import

} // namespace ngraph
