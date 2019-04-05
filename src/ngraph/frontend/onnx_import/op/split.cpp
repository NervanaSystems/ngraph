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

#include <cstdint>
#include <vector>

#include "exceptions.hpp"
#include "op/split.hpp"
#include "utils/reshape.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace error
        {
            namespace op
            {
                namespace split
                {
                    namespace detail
                    {
                        struct Error : ngraph_error
                        {
                            explicit Error(const std::string& name, const std::string& message)
                                : ngraph_error{"Split node (" + name + "): " + message}
                            {
                            }
                        };
                    } // namespace detail

                    struct OutOfRange : detail::Error
                    {
                        explicit OutOfRange(const std::string& name)
                            : Error{name,
                                    "provided split axis is out of input tensor dimensions range."}
                        {
                        }
                    };

                    struct Parts : detail::Error
                    {
                        explicit Parts(const std::string& name,
                                       std::size_t parts,
                                       std::size_t axis_length)
                            : Error{name,
                                    "tensor cannot be split into " + std::to_string(parts) +
                                        " equal parts, along axis of length " +
                                        std::to_string(axis_length)}
                        {
                        }
                    };

                    struct Sum : detail::Error
                    {
                        explicit Sum(const std::string& name, std::size_t parts, std::size_t axis)
                            : Error{name,
                                    "provided lengths of split parts does not sum up to "
                                    "length of axis we split on: " +
                                        std::to_string(parts) + " != " + std::to_string(axis)}
                        {
                        }
                    };

                } // namespace split

            } // namespace op

        } // namespace error

        namespace op
        {
            namespace set_1
            {
                NodeVector split(const Node& node)
                {
                    std::shared_ptr<ngraph::Node> input = node.get_ng_inputs().at(0);
                    auto input_shape = input->get_shape();
                    std::size_t count_outputs{node.get_output_names().size()};
                    int64_t axis{node.get_attribute_value<int64_t>("axis", 0)};
                    std::size_t axis_to_split{static_cast<std::size_t>(axis)};
                    if (axis < 0)
                    {
                        axis_to_split = input_shape.size() + axis;
                    }
                    else if (axis_to_split >= input_shape.size())
                    {
                        throw error::op::split::OutOfRange{node.get_name()};
                    }
                    std::size_t length_axis_to_split{input_shape.at(axis_to_split)};
                    std::vector<std::size_t> length_parts;
                    try
                    {
                        length_parts = node.get_attribute_value<std::vector<std::size_t>>("split");
                    }
                    catch (const std::exception&)
                    {
                        if (length_axis_to_split % count_outputs)
                        {
                            throw error::op::split::Parts{
                                node.get_name(), count_outputs, length_axis_to_split};
                        }
                        length_parts.assign(count_outputs, length_axis_to_split / count_outputs);
                    }

                    std::size_t total_parts_length = 0;
                    for (auto length : length_parts)
                    {
                        ASSERT_VALID_ARGUMENT(node, length > 0)
                            << "Invalid value in 'split' attribute";
                        total_parts_length += length;
                    }
                    ASSERT_VALID_ARGUMENT(node, total_parts_length == input_shape.at(axis_to_split))
                        << "Cannot split using values in 'split' attribute";
                    return reshape::split(input, length_parts, axis_to_split);
                }

            } // namespace set_1

        } //namespace op

    } // namespace onnx_import

} // namespace ngraph
