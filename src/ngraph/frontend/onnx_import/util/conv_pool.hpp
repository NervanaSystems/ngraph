/*******************************************************************************
 * Copyright 2018 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

#pragma once

#include <cmath>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "ngraph/coordinate_diff.hpp"
#include "ngraph/node.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/strides.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace util
        {
            namespace
            {
                namespace detail
                {
                    inline std::vector<size_t>
                        get_auto_pads(const std::shared_ptr<ngraph::Node>& node,
                                      const Shape& kernel_shape,
                                      const std::string& auto_pad)
                    {
                        std::vector<size_t> pads;

                        /*
                   * SAME_UPPER or SAME_LOWER mean pad the input so that the output size match
                   * the input. In case of odd number add the extra padding at the end for 
                   * SAME_UPPER and at the beginning for SAME_LOWER.
                   */
                        auto pad_value = [](size_t dim) {
                            return (std::reinterpret_cast<float>(dim) - 1.) / 2.;
                        };

                        if (auto_pad == "SAME_UPPER")
                        {
                            for (size_t dim : kernel_shape)
                            {
                                pads.emplace_back{std::floor(pad_value(dim))};
                            }
                            for (size_t dim : kernel_shape)
                            {
                                pads.emplace_back{std::ceil(pad_value(dim))};
                            }
                        }
                        else if (auto_pad == "SAME_LOWER")
                        {
                            for (size_t dim : kernel_shape)
                            {
                                pads.emplace_back{std::ceil(pad_value(dim))};
                            }
                            for (size_t dim : kernel_shape)
                            {
                                pads.emplace_back{std::floor(pad_value(dim))};
                            }
                        }

                        /*
                   * If auto_pad == 'VALID' we return empty vector. Similary with any other
                   * value of auto_pad.
                   */

                        return pads;
                    }
                }
            }

            /**
         * @brief Get shape of kernel (filter) in pixels.
         *
         * @param node The Node ptr representing Conv or Pool operation.
         * @return The kernel Shape object representing its dimensions (height, width, depth).
         */
            inline ngraph::Shape get_kernel_shape(const std::shared_ptr<ngraph::Node>& node)
            {
                return ngraph::Shape{
                    node.get_attribute_value<std::vector<std::size_t>>("kernel_shape", {1, 1})};
            }

            /**
         * @brief  Get number of pixels to stride operation by in each direction.
         *
         * @param node The Node ptr representing Conv or Pool operation.
         * @param kernel_shape The shape of the kernel which we retrieve strides for.
         * @return The kernel Shape object representing its dimensions (height, width, depth).
         */
            inline ngraph::Strides get_strides(const std::shared_ptr<ngraph::Node>& node,
                                               const Shape& kernel_shape)
            {
                return ngraph::Strides{node.get_attribute_value<std::vector<std::size_t>>(
                    "strides", std::vector<std::size_t>(kernel_shape.size(), (std::size_t)1))};
            }

            /**
         * @brief  Get number of pixels to stride operation by in each direction.
         *
         * @param node The Node ptr representing Conv or Pool operation.
         * @return The kernel Shape object representing its dimensions (height, width, depth).
         */
            inline ngraph::Strides get_strides(const std::shared_ptr<ngraph::Node>& node)
            {
                return get_strides(node, get_kernel_shape(node));
            }

            /**
         * @brief Get number of pixels for filter dilation in each direction.
         * 
         * @param node The Node ptr representing ONNX operation.
         * @return The Strides object containing number of pixels for filter dilation 
         *         (height, width, depth).
         */
            inline ngraph::Strides get_dilations(const std::shared_ptr<ngraph::Node>& node)
            {
                const ngraph::Shape& kernel_shape = get_kernel_shape(node);
                return ngraph::Strides{node.get_attribute_value<std::vector<std::size_t>>(
                    "dilations", std::vector<std::size_t>(kernel_shape.size(), (std::size_t)1))};
            }

            /**
         * @brief Get padding values for the operation described by an ONNX node.
         * @details If `auto_pad` attribute is specified as SAME_UPPER or SAME_LOWER, or VALID
         *          values are calculated. Otherwise values are taken from the `pads` attribute.
         *          
         *          `pads` value should follow [x1_begin, x2_begin..., x1_end, x2_end,...].
         * 
         * @param node The Node ptr representing ONNX operation.
         * @param kernel_shape The shape of the kernel which we retrieve pads for.
         * 
         * @return A pair of (padding_above, padding_below), which elements contains number of
         *         pixels to pad in respective dimensions (height, width, depth).
         */
            inline std::pair<ngraph::CoordinateDiff, ngraph::CoordinateDiff>
                get_pads(const std::shared_ptr<ngraph::Node>& node, const Shape& kernel_shape)
            {
                std::vector<std::size_t> pads;
                try
                {
                    pads = node.get_attribute_value<std::vector<std::size_t>>("pads");
                }
                catch (const error::node::UnknownAttribute&)
                {
                    std::string auto_pad{node.get_attribute_value<std::string>("auto_pad", "")};
                    if (!auto_pad.empty())
                    {
                        pads = detail::get_auto_pads(node, kernel_shape, auto_pad);
                    }
                }

                if (pads.empty())
                {
                    pads = std::vector<std::size_t>(kernel_shape.size(), (std::size_t)0);
                }

                ngraph::CoordinateDiff padding_above;
                ngraph::CoordinateDiff padding_below;

                // Pads may be specified in (H, W, C) format.
                if (pads.size() <= 3)
                {
                    padding_above = ngraph::CoordinateDiff{pads.begin(), pads.end()};
                    padding_below = ngraph::CoordinateDiff{pads.begin(), pads.end()};
                }
                else
                {
                    padding_above =
                        ngraph::CoordinateDiff{pads.begin() + pads.size() / 2, pads.end()};
                    padding_below =
                        ngraph::CoordinateDiff{pads.begin(), pads.begin() + pads.size() / 2};
                }

                return std::make_pair<ngraph::CoordinateDiff, ngraph::CoordinateDiff>(
                    padding_above, padding_below);
            }

            /**
         * @brief Get padding values for the operation described by an ONNX node.
         * @details If `auto_pad` attribute is specified as SAME_UPPER or SAME_LOWER, or VALID
         *          values are calculated. Otherwise values are taken from the `pads` attribute.
         *          
         *          `pads` value should follow [x1_begin, x2_begin..., x1_end, x2_end,...].
         * 
         * @param node The Node ptr representing ONNX operation.
         * 
         * @return A pair of (padding_above, padding_below), which elements contains number of
         *         pixels to pad in respective dimensions (height, width, depth).
         */
            inline std::pair<ngraph::CoordinateDiff, ngraph::CoordinateDiff>
                get_pads(const std::shared_ptr<ngraph::Node>& node)
            {
                return get_pads(node, get_kernel_shape(node));
            }

        } // namespace  util

    } // namespace  onnx_import

} // namespace  ngraph
