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

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "exceptions.hpp"
#include "lp_norm.hpp"
#include "ngraph/axis_set.hpp"
#include "ngraph/builder/norm.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/divide.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                NodeVector lp_norm(const Node& node)
                {
                    const std::shared_ptr<ngraph::Node> data{node.get_ng_inputs().at(0)};
                    std::int64_t axis{node.get_attribute_value<std::int64_t>("axis", -1)};
                    const std::int64_t p_norm{node.get_attribute_value<std::int64_t>("p", 2)};

                    if (axis < 0)
                    {
                        axis += data->get_shape().size();
                    }

                    ASSERT_VALID_ARGUMENT(node, p_norm == 1 || p_norm == 2)
                        << "Invalid `p` attribute value: " << p_norm
                        << "Only normalization of 1st or 2nd order is supported.";

                    const AxisSet reduction_axes{static_cast<std::size_t>(axis)};
                    std::shared_ptr<ngraph::Node> norm = ngraph::builder::lp_norm(
                        data, reduction_axes, static_cast<std::size_t>(p_norm));
                    norm = std::make_shared<ngraph::op::Broadcast>(
                        norm, data->get_shape(), reduction_axes);

                    return {data / norm};
                }

            } // namespace set_1

        } //namespace op

    } // namespace onnx_import

} // namespace ngraph
