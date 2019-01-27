//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
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
#include "ngraph/op/dequantize.hpp"
#include "ngraph/shape.hpp"
#include "quantize_linear.hpp"

namespace onnxruntime
{
    namespace ngraph_ep
    {
        ngraph::NodeVector dequantize_linear(const ngraph::onnx_import::Node& node)
        {
            ngraph::NodeVector inputs{node.get_ng_inputs()};
            std::shared_ptr<ngraph::Node> x = inputs.at(0);
            std::shared_ptr<ngraph::Node> y_scale = inputs.at(1);
            std::shared_ptr<ngraph::Node> y_zero_point = inputs.at(2);

            bool has_axis = false;
            ngraph::Shape x_shape = x->get_shape();
            ngraph::Shape y_scale_shape = y_scale->get_shape();
            ngraph::Shape y_zero_point_shape = y_zero_point->get_shape();
            ngraph::AxisSet axis_set{};

            try
            {
                std::int64_t axis = node.get_attribute_value<std::int64_t>("axis");
                axis_set = ngraph::AxisSet{static_cast<std::size_t>(axis)};
                has_axis = true;
            }
            catch (const ngraph::onnx_import::error::node::UnknownAttribute&)
            {
            }

            // Per `axis` quantization. Input ‘scale’ and ‘zero_point’ must be 1-D tensors
            if (has_axis)
            {
                std::size_t axis_dim_value = x_shape.at(*std::begin(axis_set));

                ASSERT_VALID_ARGUMENT(
                    node, (y_scale_shape.size() == 1 && y_scale_shape.at(0) == axis_dim_value))
                    << "y_scale must be 1D tensor with size equal to: " << axis_dim_value;
                ASSERT_VALID_ARGUMENT(
                    node,
                    (y_zero_point_shape.size() == 1 && y_zero_point_shape.at(0) == axis_dim_value))
                    << "y_zero_point must be 1D tensor with size equal to: " << axis_dim_value;
            }
            // Per tensor quantization. Input ‘scale’ and ‘zero_point’ must be scalars
            else
            {
                ASSERT_VALID_ARGUMENT(node, y_scale_shape.size() == 0)
                    << "y_scale must be a scalar if no axis is provided.";
                ASSERT_VALID_ARGUMENT(node, y_zero_point_shape.size() == 0)
                    << "y_zero_point must be a scalar if no axis is provided.";
            }

            // TODO:  THIS IS A WORKAROUND WHICH SHOULD BE REMOVED
            // nGraph requires same data type for input and offset nodes.
            if (x->get_element_type() != y_zero_point->get_element_type())
            {
                // Currently only support Constant node.
                ASSERT_IS_SUPPORTED(node, y_zero_point->is_constant())
                    << "doesn't support zero point input of other type than Constant.";

                auto y_zp_constant = std::static_pointer_cast<ngraph::op::Constant>(y_zero_point);

                y_zero_point = ngraph::op::Constant::create<std::uint8_t>(
                    x->get_element_type(),
                    y_zero_point->get_shape(),
                    y_zp_constant->get_vector<std::uint8_t>());
            }
            // END TODO

            return {std::make_shared<ngraph::op::Dequantize>(
                x, y_scale, y_zero_point, y_scale->get_element_type(), axis_set)};
        }

    } // namespace ngraph_ep

} // namespace onnxruntime
