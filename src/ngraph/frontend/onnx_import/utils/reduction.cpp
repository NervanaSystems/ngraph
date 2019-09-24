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

#include <cstddef> // std::size_t
#include <vector>

#include "exceptions.hpp"
#include "reduction.hpp"
#include "utils/common.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace reduction
        {
            namespace detail
            {
                AxisSet get_reduction_axes(const Node& node)
                {
                    auto reduction_axes =
                        node.get_attribute_value<std::vector<std::int64_t>>("axes", {});
                    std::vector<std::size_t> valid_reduction_axes = common::validate_axes(
                        node, reduction_axes, node.get_ng_inputs().at(0)->get_shape().size());

                    if (reduction_axes.empty())
                    {
                        valid_reduction_axes =
                            onnx_import::common::get_monotonic_range<std::size_t>(
                                node.get_ng_inputs().at(0)->get_shape().size());
                    }
                    return AxisSet{valid_reduction_axes};
                }
            } // namespace  detail

            std::shared_ptr<ngraph::Node>
                make_ng_reduction_op(const Node& node,
                                     const std::shared_ptr<ngraph::Node>& ng_input,
                                     ReductionFunction reduction_function)
            {
                auto data_shape = ng_input->get_shape();

                auto reduction_axes = detail::get_reduction_axes(node);

                ASSERT_VALID_ARGUMENT(node, reduction_axes.size() <= data_shape.size())
                    << "provided reduction axes count (" << reduction_axes.size()
                    << ") is larger than input tensor rank (" << data_shape.size() << ")";

                std::shared_ptr<ngraph::Node> op_node =
                    reduction_function(ng_input, reduction_axes);

                std::int64_t keepdims = node.get_attribute_value<std::int64_t>("keepdims", 1);
                if (keepdims == 0)
                {
                    return op_node;
                }

                auto output_shape = data_shape;
                // flatten reduced axes and preserve original dimensions count.
                for (const auto& idx : reduction_axes)
                {
                    output_shape.at(idx) = 1;
                }
                return std::make_shared<ngraph::op::Reshape>(
                    op_node,
                    ngraph::get_default_order(op_node->get_shape().size()),
                    Shape{output_shape});
            }

        } // namespace  reduction
    }     // namespace onnx_import
} // namespace ngraph
