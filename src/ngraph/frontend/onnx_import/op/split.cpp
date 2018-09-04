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

#include "ngraph/op/slice.hpp"

#include "op/split.hpp"

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
                    }

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
            namespace detail
            {
                template <typename T>
                inline T get_valid_array_index(T left, T right)
                {
                    return (left >= 0) ? std::min(left, right)
                                       : std::max(static_cast<T>(0), right + left);
                }

                inline std::shared_ptr<ngraph::op::Slice>
                    make_ng_slice(const std::shared_ptr<ngraph::Node>& node,
                                  std::vector<std::size_t> axes,
                                  std::vector<std::size_t> starts,
                                  std::vector<std::size_t> ends)
                {
                    std::vector<std::size_t> upper_bounds{node->get_shape()};
                    std::vector<std::size_t> lower_bounds(upper_bounds.size());
                    for (std::size_t index{0}; index < axes.size(); ++index)
                    {
                        std::size_t axis{axes.at(index)};
                        lower_bounds.at(axis) =
                            get_valid_array_index(starts.at(index), node->get_shape().at(axis));
                        upper_bounds.at(axis) =
                            get_valid_array_index(ends.at(index), node->get_shape().at(axis));
                    }
                    return std::make_shared<ngraph::op::Slice>(node, lower_bounds, upper_bounds);
                }

            } // namespace detail

            NodeVector split(const Node& node)
            {
                std::shared_ptr<ngraph::Node> input = node.get_ng_inputs().at(0);
                std::size_t count_outputs{node.get_output_names().size()};
                int64_t axis{node.get_attribute_value<int64_t>("axis", 0)};
                std::size_t axis_to_split{static_cast<std::size_t>(axis)};
                if (axis < 0)
                {
                    axis_to_split = input->get_shape().size() + axis;
                }
                else if (axis_to_split >= input->get_shape().size())
                {
                    throw error::op::split::OutOfRange{node.get_name()};
                }
                std::size_t length_axis_to_split{input->get_shape().at(axis_to_split)};
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

                std::size_t start_index{0};
                NodeVector outputs;
                for (const auto& length_part : length_parts)
                {
                    std::size_t end_index{start_index + length_part};
                    outputs.push_back(
                        detail::make_ng_slice(input, {axis_to_split}, {start_index}, {end_index}));
                    start_index = end_index;
                }
                return outputs;
            }

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
