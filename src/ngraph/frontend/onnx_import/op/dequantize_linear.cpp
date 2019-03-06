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

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>

#include "exceptions.hpp"
#include "ngraph/axis_set.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/convert.hpp"
#include "ngraph/op/dequantize.hpp"
#include "ngraph/shape.hpp"
#include "quantize_linear.hpp"
#include "utils/common.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                NodeVector dequantize_linear(const Node& node)
                {
                    NodeVector inputs{node.get_ng_inputs()};
                    std::shared_ptr<ngraph::Node> x = inputs.at(0);
                    std::shared_ptr<ngraph::Node> x_scale = inputs.at(1);
                    std::shared_ptr<ngraph::Node> zero_point;
                    if (inputs.size() == 3)
                    {
                        zero_point = inputs.at(2);
                    }
                    else
                    {
                        zero_point = common::make_constant_node(
                            x->get_element_type(), Shape{}, std::vector<std::uint8_t>{0});
                    }

                    Shape y_scale_shape = x_scale->get_shape();
                    Shape y_zero_point_shape = zero_point->get_shape();

                    ASSERT_VALID_ARGUMENT(node, y_scale_shape.size() == 0)
                        << "x_scale must be a scalar.";
                    ASSERT_VALID_ARGUMENT(node, y_zero_point_shape.size() == 0)
                        << "zero_point must be a scalar.";

                    if (x->get_element_type() != zero_point->get_element_type())
                    {
                        zero_point = std::make_shared<ngraph::op::Convert>(zero_point,
                                                                           x->get_element_type());
                    }

                    return {std::make_shared<ngraph::op::Dequantize>(
                        x, x_scale, zero_point, x_scale->get_element_type(), AxisSet{})};
                }

            } // namespace set_1

        } //namespace op

    } // namespace onnx_import

} // namespace ngraph
