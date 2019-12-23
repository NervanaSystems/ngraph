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

#include "default_opset.hpp"
#include "exceptions.hpp"
#include "ngraph/builder/split.hpp"
#include "ngraph/coordinate_diff.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/convert.hpp"
#include "ngraph/op/pad.hpp"
#include "ngraph/shape.hpp"
#include "pad.hpp"
#include "utils/convpool.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                NodeVector pad(const Node& node)
                {
                    auto data = node.get_ng_inputs().at(0);
                    const Shape& data_shape = data->get_shape();

                    double value = node.get_attribute_value<double>("value", 0);
                    std::string mode = node.get_attribute_value<std::string>("mode", "constant");
                    ngraph::op::PadMode pad_mode;
                    if (mode == "constant")
                    {
                        pad_mode = ngraph::op::PadMode::CONSTANT;
                    }
                    else if (mode == "reflect")
                    {
                        pad_mode = ngraph::op::PadMode::REFLECT;
                    }
                    else if (mode == "edge")
                    {
                        pad_mode = ngraph::op::PadMode::EDGE;
                    }
                    else
                    {
                        throw error::InvalidArgument("Unsupported padding mode: [" + mode + "]");
                    }
                    auto paddings = convpool::get_pads(node, data_shape);
                    ngraph::CoordinateDiff padding_below = paddings.first;
                    ngraph::CoordinateDiff padding_above = paddings.second;

                    return {std::make_shared<default_opset::Pad>(
                        data,
                        std::make_shared<default_opset::Constant>(
                            element::i64, ngraph::Shape{padding_below.size()}, padding_below),
                        std::make_shared<default_opset::Constant>(
                            element::i64, ngraph::Shape{padding_above.size()}, padding_above),
                        std::make_shared<default_opset::Constant>(
                            data->get_element_type(), ngraph::Shape{}, std::vector<double>{value}),
                        pad_mode)};
                }

            } // namespace set_1
            namespace set_11
            {
                NodeVector pad(const Node& node)
                {
                    auto data = node.get_ng_inputs().at(0);
                    const Shape& data_shape = data->get_shape();

                    auto pads = node.get_ng_inputs().at(1);
                    std::shared_ptr<ngraph::Node> values;
                    if (node.get_ng_inputs().size() == 3)
                    {
                        values = node.get_ng_inputs().at(2);
                    }
                    else
                    {
                        values = ngraph::op::Constant::create(element::i64, ngraph::Shape{}, {0});
                    }
                    auto axis = ngraph::op::Constant::create(element::i64, ngraph::Shape{}, {0});
                    NodeVector padding = builder::split(pads, 2, 0);
                    auto padding_begin =
                        std::make_shared<default_opset::Convert>(padding.at(0), element::i64);
                    auto padding_end =
                        std::make_shared<default_opset::Convert>(padding.at(1), element::i64);

                    std::string mode = node.get_attribute_value<std::string>("mode", "constant");
                    ngraph::op::PadMode pad_mode;
                    if (mode == "constant")
                    {
                        pad_mode = ngraph::op::PadMode::CONSTANT;
                    }
                    else if (mode == "reflect")
                    {
                        pad_mode = ngraph::op::PadMode::REFLECT;
                    }
                    else if (mode == "edge")
                    {
                        pad_mode = ngraph::op::PadMode::EDGE;
                    }
                    else
                    {
                        throw error::InvalidArgument("Unsupported padding mode: [" + mode + "]");
                    }
                    return {std::make_shared<default_opset::Pad>(
                        data, padding_begin, padding_end, values, pad_mode)};
                }

            } // namespace set_11

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
