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

#include "default_opset.hpp"
#include "resize.hpp"


namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_10
            {
                NodeVector resize(const onnx_import::Node& node)
                {
                    const std::shared_ptr<ngraph::Node> data{node.get_ng_inputs().at(0)};
                    const auto data_shape = data->get_output_partial_shape(0);
                    const auto data_rank = data_shape.rank();
                    auto mode = node.get_attribute_value<std::string>("mode", "nearest");
                    auto scales = node.get_attribute_value<std::vector<float>>("scales");

                    AxisSet axes;
                    for (auto ax = 0; ax < scales.size(); ++ax)
                    {
                        axes.insert(ax);
                    }

                    auto attrs = ngraph::op::InterpolateAttrs();
                    attrs.axes = axes;
                    attrs.mode = mode;
                    
                    if (data_shape.is_static())
                    {
                        auto data_static_shape = data_shape.to_shape();
                        Shape output_shape;
                        for (size_t i = 0; i < data_static_shape.size(); ++i)
                        {
                            output_shape.push_back(std::floor(data_static_shape.at(i) * scales.at(i)));
                        }
                        auto output_shape_const = default_opset::Constant::create(element::u64, 
                            Shape(output_shape.size()), output_shape);
                        return {std::make_shared<default_opset::Interpolate>(data, output_shape_const, attrs)};
                    }
                    else
                    {
                        auto shape_of_data = std::make_shared<default_opset::ShapeOf>(data);
                        auto scales_const = default_opset::Constant::create(element::f32, Shape(scales.size()), scales);
                        auto multiply = std::make_shared<default_opset::Multiply>(shape_of_data, scales_const);
                        auto output_shape = std::make_shared<default_opset::Floor>(multiply);
                        return {std::make_shared<default_opset::Interpolate>(data, output_shape, attrs)};
                    }
                }

            } // namespace set_10

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
