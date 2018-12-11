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

#include <cmath>

#include "convpool.hpp"
#include "core/attribute.hpp"
#include "core/node.hpp"
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
                Shape get_output_data_shape(const Shape& input, const Strides& strides)
                {
                    Shape output;
                    for (std::size_t idx = 0; idx < input.size(); ++idx)
                    {
                        output.emplace_back(std::ceil(static_cast<float>(input.at(idx)) /
                                                      static_cast<float>(strides.at(idx))));
                    }
                    return output;
                }

                Shape get_pad_shape(const Shape& input,
                                    const Shape& kernel,
                                    const Shape& strides,
                                    const Shape& output)
                {
                    Shape pad_shape;
                    for (std::size_t idx = 0; idx < input.size(); ++idx)
                    {
                        pad_shape.emplace_back((output.at(idx) - 1) * strides.at(idx) +
                                               kernel.at(idx) - input.at(idx));
                    }
                    return pad_shape;
                }

                CoordinateDiff get_auto_pads(const Shape& input_shape,
                                             const Shape& kernel_shape,
                                             const Strides& strides,
                                             const std::string& auto_pad)
                {
                    CoordinateDiff pads_begin;
                    CoordinateDiff pads_end;
                    // Omit {N,C} axes
                    Shape input_spatial_shape{std::next(std::begin(input_shape), 2),
                                              std::end(input_shape)};
                    // Assume that all {input_spatial_shape,kernel_shape,strides}.size()
                    // is the same.
                    const Shape& output_spatial_shape =
                        get_output_data_shape(input_spatial_shape, strides);
                    const Shape& pad_shape = get_pad_shape(
                        input_spatial_shape, kernel_shape, strides, output_spatial_shape);
                    if (auto_pad == "SAME_UPPER")
                    {
                        for (size_t pad : pad_shape)
                        {
                            // Integer division
                            pads_begin.emplace_back(pad / 2);
                            pads_end.emplace_back(pad - pads_begin.back());
                        }
                    }
                    else if (auto_pad == "SAME_LOWER")
                    {
                        for (size_t pad : pad_shape)
                        {
                            // Integer division
                            pads_end.emplace_back(pad / 2);
                            pads_begin.emplace_back(pad - pads_end.back());
                        }
                    }
                    CoordinateDiff pads{pads_begin};
                    pads.insert(std::end(pads), std::begin(pads_end), std::end(pads_end));
                    return pads;
                }

            } // namespace

            std::pair<CoordinateDiff, CoordinateDiff> get_pads(const Node& node,
                                                               const Shape& kernel_shape)
            {
                CoordinateDiff pads;
                try
                {
                    auto pads_int64 = node.get_attribute_value<std::vector<int64_t>>("pads");
                    pads = CoordinateDiff{std::begin(pads_int64), std::end(pads_int64)};
                }
                catch (const error::node::UnknownAttribute&)
                {
                    std::string auto_pad{node.get_attribute_value<std::string>("auto_pad", "")};
                    if (!auto_pad.empty())
                    {
                        pads = get_auto_pads(node.get_ng_inputs().at(0)->get_shape(),
                                             kernel_shape,
                                             get_strides(node),
                                             auto_pad);
                    }
                }
                if (pads.empty())
                {
                    pads = CoordinateDiff(static_cast<std::ptrdiff_t>(kernel_shape.size()), 0UL);
                }

                if (pads.size() != kernel_shape.size() * 2)
                {
                    // Paddings specified in (H, W, C) format.
                    return {pads, pads};
                }
                else
                {
                    return {{std::begin(pads), std::begin(pads) + pads.size() / 2},
                            {std::begin(pads) + pads.size() / 2, std::end(pads)}};
                }
            }

        } // namespace convpool
    }     // namespace onnx_import
} // namespace ngraph
