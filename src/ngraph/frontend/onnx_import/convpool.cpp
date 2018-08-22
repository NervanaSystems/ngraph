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

#include "convpool.hpp"
#include <cmath>

namespace ngraph
{
    namespace onnx_import
    {
        namespace attribute
        {
            namespace
            {
                CoordinateDiff get_auto_pads(const Shape& kernel_shape, const std::string& auto_pad)
                {
                    CoordinateDiff pads;

                    // Add padding to the input to match the size of output size.
                    auto pad_value = [](size_t dim) {
                        return (static_cast<float>(dim) - 1.f) / 2.f;
                    };

                    if (auto_pad == "SAME_UPPER")
                    {
                        for (size_t dim : kernel_shape)
                        {
                            pads.emplace_back(std::floor(pad_value(dim)));
                        }
                        for (size_t dim : kernel_shape)
                        {
                            pads.emplace_back(std::ceil(pad_value(dim)));
                        }
                    }
                    else if (auto_pad == "SAME_LOWER")
                    {
                        for (size_t dim : kernel_shape)
                        {
                            pads.emplace_back(std::ceil(pad_value(dim)));
                        }
                        for (size_t dim : kernel_shape)
                        {
                            pads.emplace_back(std::floor(pad_value(dim)));
                        }
                    }

                    return pads;
                }

            } // namespace

            std::pair<CoordinateDiff, CoordinateDiff> get_pads(const Node& node,
                                                               const Shape& kernel_shape)
            {
                CoordinateDiff pads;
                try
                {
                    pads = CoordinateDiff{node.get_attribute_value<std::vector<int64_t>>("pads")};
                }
                catch (const error::node::UnknownAttribute&)
                {
                    std::string auto_pad{node.get_attribute_value<std::string>("auto_pad", "")};
                    if (!auto_pad.empty())
                    {
                        pads = get_auto_pads(kernel_shape, auto_pad);
                    }
                }
                if (pads.empty())
                {
                    pads = {static_cast<std::ptrdiff_t>(kernel_shape.size()), 0UL};
                }

                if (pads.size() <= 3)
                {
                    // Paddings specified in (H, W, C) format.
                    return {pads, pads};
                }
                else
                {
                    return {{std::begin(pads) + pads.size() / 2, std::end(pads)},
                            {std::begin(pads), std::begin(pads) + pads.size() / 2}};
                }
            }

        } // namespace attribute
    }     // namespace onnx_import
} // namespace ngraph
