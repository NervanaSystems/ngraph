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
#include <memory>

#include "exceptions.hpp"
#include "lp_pool.hpp"
#include "ngraph/axis_set.hpp"
#include "ngraph/builder/norm.hpp"
#include "ngraph/builder/split.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/util.hpp"
#include "utils/common.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                NodeVector global_lp_pool(const Node& node)
                {
                    std::shared_ptr<ngraph::Node> data{node.get_ng_inputs().at(0)};
                    std::size_t channel_axis{1};
                    std::size_t channels_count = data->get_shape().at(channel_axis);
                    std::int64_t p_norm{node.get_attribute_value<std::int64_t>("p", 2)};

                    ASSERT_VALID_ARGUMENT(node, p_norm >= 0)
                        << "Only positive (including zero) values are supported for 'p' attribute.";

                    NodeVector slices = ngraph::builder::split(data, channels_count, channel_axis);

                    for (auto& slice : slices)
                    {
                        const Shape& orig_shape = data->get_shape();
                        // all dimensions except spatial/feature
                        AxisSet reduction_axes{
                            common::get_monotonic_range<std::size_t>(orig_shape.size(), 2)};

                        slice = ngraph::builder::lp_norm(
                            slice, reduction_axes, static_cast<std::size_t>(p_norm));

                        // output shape is all ones except N channel
                        Shape output_shape(orig_shape.size(), 1);
                        output_shape.at(0) = orig_shape.at(0);
                        slice = std::make_shared<ngraph::op::Reshape>(
                            slice,
                            ngraph::get_default_order(slice->get_shape().size()),
                            output_shape);
                    }

                    return {std::make_shared<ngraph::op::Concat>(slices, channel_axis)};
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
