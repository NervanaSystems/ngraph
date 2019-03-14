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
#include "ngraph/op/quantize.hpp"
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
                NodeVector quantize_linear(const Node& node)
                {
                    NodeVector inputs{node.get_ng_inputs()};
                    std::shared_ptr<ngraph::Node> x = inputs.at(0);
                    std::shared_ptr<ngraph::Node> y_scale = inputs.at(1);
                    std::shared_ptr<ngraph::Node> y_zero_point = inputs.at(2);

                    Shape y_scale_shape = y_scale->get_shape();
                    Shape y_zero_point_shape = y_zero_point->get_shape();

                    ASSERT_VALID_ARGUMENT(node, y_scale_shape.size() == 0)
                        << "y_scale must be a scalar.";
                    ASSERT_VALID_ARGUMENT(node, y_zero_point_shape.size() == 0)
                        << "y_zero_point must be a scalar.";

                    return {std::make_shared<ngraph::op::Quantize>(
                        x,
                        y_scale,
                        y_zero_point,
                        y_zero_point->get_element_type(),
                        AxisSet{},
                        ngraph::op::Quantize::RoundMode::ROUND_NEAREST_TOWARD_EVEN)};
                }

            } // namespace set_1

        } //namespace op

    } // namespace onnx_import

} // namespace ngraph
