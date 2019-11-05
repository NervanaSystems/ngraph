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

#include <unordered_map>

#include "convpool.hpp"
#include "exceptions.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/strides.hpp"
#include "ngraph/validation_util.hpp"

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

            ngraph::op::PadType get_auto_pad(const Node& node)
            {
                // Default value means use explicitly provided padding values.
                ngraph::op::PadType pad_type{ngraph::op::PadType::NOTSET};
                if (node.has_attribute("auto_pad"))
                {
                    static std::unordered_multimap<std::string, ngraph::op::PadType>
                        auto_pad_values{
                            {"VALID", ngraph::op::PadType::VALID},
                            {"SAME_UPPER", ngraph::op::PadType::SAME_UPPER},
                            {"SAME_LOWER", ngraph::op::PadType::SAME_LOWER},
                            {"NOTSET", ngraph::op::PadType::NOTSET},
                            {"", ngraph::op::PadType::NOTSET},
                        };

                    const std::string& pad_str{node.get_attribute_value<std::string>("auto_pad")};
                    const auto pad_val_it = auto_pad_values.find(pad_str);
                    CHECK_VALID_NODE(node,
                                     pad_val_it != auto_pad_values.end(),
                                     "Provided `auto_pad` attribute value: '",
                                     pad_str,
                                     "' is invalid.");
                    pad_type = pad_val_it->second;
                }
                return pad_type;
            }

            std::pair<CoordinateDiff, CoordinateDiff> get_pads(const Node& node,
                                                               const Shape& kernel_shape)
            {
                CoordinateDiff pads(kernel_shape.size(), 0);
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

            void calculate_auto_pads(const Shape& data_shape,
                                     const Shape& filter_shape,
                                     const Strides& strides,
                                     const Strides& dilations,
                                     const ngraph::op::PadType& pad_type,
                                     CoordinateDiff& padding_below,
                                     CoordinateDiff& padding_above)
            {
                if (pad_type == ngraph::op::PadType::SAME_UPPER ||
                    pad_type == ngraph::op::PadType::SAME_LOWER)
                {
                    padding_below.clear();
                    padding_above.clear();
                    // Extract kernel shape - remove (N,C) channels
                    Shape kernel_shape(std::next(std::begin(filter_shape), 2),
                                       std::end(filter_shape));
                    ngraph::infer_auto_padding(data_shape,
                                               kernel_shape,
                                               strides,
                                               dilations,
                                               pad_type,
                                               padding_above,
                                               padding_below);
                }
            }

        } // namespace convpool
    }     // namespace onnx_import
} // namespace ngraph
