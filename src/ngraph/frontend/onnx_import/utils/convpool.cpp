//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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
                const auto& data_shape = node.get_ng_inputs().at(0)->get_output_partial_shape(0);
                const size_t input_spatial_dims = data_shape.rank().get_length() - 2;
                return node.get_attribute_value<std::vector<size_t>>(
                    "kernel_shape", std::vector<size_t>(input_spatial_dims, 1UL));
            }

            namespace detail
            {
                /// \brief              Helper method used to read vector attribute
                /// \note               Default value is vector of size spatial dims filled with
                ///                     ones
                ///
                /// \param   node       Node from which attribute is read
                /// \param   attr_name  Attribute name (such as `strides`, `dilations`)
                ///
                /// \return             Read vector attribute if available or default value
                std::vector<std::size_t> get_attribute_value(const Node& node,
                                                             const std::string& attr_name)
                {
                    if (node.has_attribute(attr_name))
                    {
                        return node.get_attribute_value<std::vector<std::size_t>>(attr_name);
                    }
                    const auto data_rank =
                        node.get_ng_inputs().at(0)->get_output_partial_shape(0).rank();
                    CHECK_VALID_NODE(node,
                                     data_rank.is_static(),
                                     "If '",
                                     attr_name,
                                     "' is not provided data rank must be static");
                    const auto data_spatial_dims = data_rank.get_length() - 2;
                    return std::vector<std::size_t>(data_spatial_dims, 1UL);
                }
            } // namespace detail

            Strides get_strides(const Node& node)
            {
                return detail::get_attribute_value(node, "strides");
            }

            Strides get_dilations(const Node& node)
            {
                return detail::get_attribute_value(node, "dilations");
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
                                                               const size_t kernel_rank)
            {
                CoordinateDiff pads(kernel_rank, 0);
                if (node.has_attribute("pads"))
                {
                    auto pads_int64 = node.get_attribute_value<std::vector<int64_t>>("pads");
                    pads = CoordinateDiff{std::begin(pads_int64), std::end(pads_int64)};
                }

                if (pads.size() == kernel_rank * 2)
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

            std::pair<CoordinateDiff, CoordinateDiff> get_pads(const Node& node)
            {
                const auto data_rank =
                    node.get_ng_inputs().at(0)->get_output_partial_shape(0).rank();
                CHECK_VALID_NODE(node,
                                 data_rank.is_static(),
                                 "The rank of node must be static in order to calculate pads");
                const auto data_spatial_dims = data_rank.get_length() - 2;

                return get_pads(node, data_spatial_dims);
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
