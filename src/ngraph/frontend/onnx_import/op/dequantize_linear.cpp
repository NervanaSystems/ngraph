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
                    std::shared_ptr<ngraph::Node> y_scale = inputs.at(1);
                    std::shared_ptr<ngraph::Node> y_zero_point = inputs.at(2);

                    bool has_axis = false;
                    Shape x_shape = x->get_shape();
                    Shape y_scale_shape = y_scale->get_shape();
                    Shape y_zero_point_shape = y_zero_point->get_shape();
                    AxisSet axis_set{};

                    try
                    {
                        std::int64_t axis = node.get_attribute_value<std::int64_t>("axis");
                        axis_set = AxisSet{static_cast<std::size_t>(axis)};
                        has_axis = true;
                    }
                    catch (const error::node::UnknownAttribute&)
                    {
                    }

                    // Per `axis` quantization. Input ‘scale’ and ‘zero_point’ must be 1-D tensors
                    if (has_axis)
                    {
                        std::size_t axis_dim_value = x_shape.at(*std::begin(axis_set));

                        ASSERT_VALID_ARGUMENT(
                            node,
                            (y_scale_shape.size() == 1 && y_scale_shape.at(0) == axis_dim_value))
                            << "y_scale must be 1D tensor with size equal to: " << axis_dim_value;
                        ASSERT_VALID_ARGUMENT(node,
                                              (y_zero_point_shape.size() == 1 &&
                                               y_zero_point_shape.at(0) == axis_dim_value))
                            << "y_zero_point must be 1D tensor with size equal to: "
                            << axis_dim_value;
                    }
                    // Per tensor quantization. Input ‘scale’ and ‘zero_point’ must be scalars
                    else
                    {
                        ASSERT_VALID_ARGUMENT(node, y_scale_shape.size() == 0)
                            << "y_scale must be a scalar if no axis is provided.";
                        ASSERT_VALID_ARGUMENT(node, y_zero_point_shape.size() == 0)
                            << "y_zero_point must be a scalar if no axis is provided.";
                    }

                    if (x->get_element_type() != y_zero_point->get_element_type())
                    {
                        y_zero_point = std::make_shared<ngraph::op::Convert>(y_zero_point,
                                                                             x->get_element_type());
                    }

                    return {std::make_shared<ngraph::op::Dequantize>(
                        x, y_scale, y_zero_point, y_scale->get_element_type(), axis_set)};
                }

            } // namespace set_1

        } //namespace op

    } // namespace onnx_import

} // namespace ngraph
