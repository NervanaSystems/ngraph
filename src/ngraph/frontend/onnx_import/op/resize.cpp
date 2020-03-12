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

#include "resize.hpp"
#include "default_opset.hpp"
#include "exceptions.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                NodeVector resize(const onnx_import::Node& node)
                {
                    const auto inputs = node.get_ng_inputs();
                    const auto data = inputs.at(0);
                    const auto scales = inputs.at(1);

                    const auto data_shape = data->get_output_partial_shape(0);
                    const auto scales_shape = scales->get_output_partial_shape(0);

                    auto mode = node.get_attribute_value<std::string>("mode", "nearest");

                    auto attrs = ngraph::op::InterpolateAttrs();
                    attrs.mode = mode;

                    if (scales_shape.rank().is_static())
                    {
                        AxisSet axes;
                        for (int ax = 0; ax < scales_shape.rank().get_length(); ++ax)
                        {
                            axes.insert(ax);
                        }
                        attrs.axes = axes;
                    }
                    else
                    {
                        throw error::NotSupported(
                            "ResizeOp: Dynamic rank of Scales input is not supported");
                    }

                    auto shape_of_data = std::make_shared<default_opset::Convert>(
                        std::make_shared<default_opset::ShapeOf>(data), ngraph::element::f32);
                    auto multiply =
                        std::make_shared<default_opset::Multiply>(shape_of_data, scales);
                    auto output_shape = std::make_shared<default_opset::Convert>(
                        std::make_shared<default_opset::Floor>(multiply), ngraph::element::i64);
                    return {
                        std::make_shared<default_opset::Interpolate>(data, output_shape, attrs)};
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
