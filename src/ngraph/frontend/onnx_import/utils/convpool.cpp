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

#include "convpool.hpp"
#include "core/attribute.hpp"
#include "core/node.hpp"
#include "exceptions.hpp"
#include "ngraph/coordinate_diff.hpp"
#include "ngraph/shape.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace convpool
        {
            Shape get_kernel_shape(const Node& node)
            {
                std::size_t input_spatial_dims = node.get_ng_inputs().at(0)->get_shape().size() - 2;
                return node.get_attribute_value<std::vector<std::size_t>>(
                    "kernel_shape", std::vector<std::size_t>(input_spatial_dims, 1UL));
            }

            namespace detail
            {
                Strides get_strides_helper(const Node& node,
                                           const std::string& name,
                                           const Shape& kernel_shape)
                {
                    return node.get_attribute_value<std::vector<std::size_t>>(
                        name, std::vector<std::size_t>(kernel_shape.size(), 1UL));
                }
            } // namespace detail

            Strides get_strides(const Node& node, const Shape& kernel_shape)
            {
                return detail::get_strides_helper(node, "strides", kernel_shape);
            }

            Strides get_strides(const Node& node)
            {
                return get_strides(node, get_kernel_shape(node));
            }

            Strides get_dilations(const Node& node)
            {
                return detail::get_strides_helper(node, "dilations", get_kernel_shape(node));
            }

            namespace
            {
                ngraph::op::PadType get_ng_pad_type(const std::string& pad_name)
                {
                    if (pad_name == "VALID")
                    {
                        return ngraph::op::PadType::VALID;
                    }
                    if (pad_name == "SAME_UPPER")
                    {
                        return ngraph::op::PadType::SAME_UPPER;
                    }
                    if (pad_name == "SAME_LOWER")
                    {
                        return ngraph::op::PadType::SAME_LOWER;
                    }
                    if (pad_name == "NOTSET")
                    {
                        return ngraph::op::PadType::NOTSET;
                    }
                    return ngraph::op::PadType::INVALID;
                }
            } // namespace

            ngraph::op::PadType get_auto_pad(const Node& node)
            {
                // Default value means use explicitly provided padding values.
                ngraph::op::PadType pad_type{ngraph::op::PadType::NOTSET};
                if (node.has_attribute("auto_pad"))
                {
                    const std::string& pad_str{node.get_attribute_value<std::string>("auto_pad")};
                    pad_type = get_ng_pad_type(pad_str);
                    CHECK_VALID_NODE(node,
                                     pad_type != ngraph::op::PadType::INVALID,
                                     "Provided `auto_pad` attribute value: '",
                                     pad_str,
                                     "' is invalid.");
                }
                return pad_type;
            }

            std::pair<CoordinateDiff, CoordinateDiff> get_pads(const Node& node,
                                                               const Shape& kernel_shape)
            {
                CoordinateDiff pads;
                if (node.has_attribute("pads"))
                {
                    auto pads_int64 = node.get_attribute_value<std::vector<int64_t>>("pads");
                    pads = CoordinateDiff{std::begin(pads_int64), std::end(pads_int64)};
                }

                if (pads.size() == kernel_shape.size() * 2)
                {
                    return {{std::begin(pads), std::begin(pads) + pads.size() / 2},
                            {std::begin(pads) + pads.size() / 2, std::end(pads)}};
                }
                else
                {
                    // No paddings provided or only one side values provided, which means same
                    // padding at both begin and end of axis.
                    return {pads, pads};
                }
            }

        } // namespace convpool
    }     // namespace onnx_import
} // namespace ngraph
