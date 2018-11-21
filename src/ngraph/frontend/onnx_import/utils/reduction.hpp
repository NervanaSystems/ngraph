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

#pragma once

#include <cstddef>  // std::size_t
#include <cstdint>  // std::int64_t
#include <iterator> // std::begin, std::end
#include <memory>   // std::make_shared
#include <numeric>  // std::iota
#include <string>
#include <type_traits> // std::enable_if, std::is_base_of
#include <vector>

#include "ngraph/axis_set.hpp"
#include "ngraph/node_vector.hpp"
#include "ngraph/op/convert.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/util/arithmetic_reduction.hpp"
#include "ngraph/shape.hpp"

#include "core/node.hpp"
#include "exceptions.hpp"
#include "utils/common.hpp"
#include "utils/reshape.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace reduction
        {
            namespace detail
            {
                inline AxisSet get_reduction_axes(const Node& node)
                {
                    auto reduction_axes =
                        node.get_attribute_value<std::vector<std::size_t>>("axes", {});
                    if (reduction_axes.empty())
                    {
                        reduction_axes = onnx_import::common::get_monotonic_range<std::size_t>(
                            node.get_ng_inputs().at(0)->get_shape().size());
                    }
                    return AxisSet{reduction_axes};
                }
            } // namespace  detail

            /// \brief      Create an nGraph version of an ONNX reduction operation.
            ///
            /// \param[in]  node          The node representing incoming ONNX operation.
            ///
            /// \tparam     OnnxOperator  Class of an nGraph ArithmeticReduction operation
            ///                           (e.g. Min, Max, SUm, Product).
            ///
            /// \return     nGraph node equivalent of the ONNX operation.
            ///
            template <class OnnxOperator,
                      typename std::enable_if<std::is_base_of<ngraph::op::util::ArithmeticReduction,
                                                              OnnxOperator>::value,
                                              int>::type = 0>
            std::shared_ptr<ngraph::Node>
                make_ng_reduction_op(const Node& node,
                                     const std::shared_ptr<ngraph::Node>& ng_input)
            {
                auto data_shape = ng_input->get_shape();

                auto reduction_axes = detail::get_reduction_axes(node);

                ASSERT_VALID_ARGUMENT(node, reduction_axes.size() <= data_shape.size())
                    << "provided reduction axes count (" << reduction_axes.size()
                    << ") is larger than input tensor rank (" << data_shape.size() << ")";

                auto op_node = std::make_shared<OnnxOperator>(ng_input, reduction_axes);

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
                    reshape::get_default_axis_vector(op_node->get_shape().size()),
                    Shape{output_shape});
            }

            template <class IndexReduction>
            std::shared_ptr<ngraph::Node> make_ng_index_reduction_op(const Node& node)
            {
                auto axis = node.get_attribute_value<int64_t>("axis", 0);
                auto keepdims = node.get_attribute_value<int64_t>("keepdims", 1);
                auto input_node = node.get_ng_inputs().at(0);

                auto op_node = std::make_shared<IndexReduction>(input_node, axis, element::i64);

                if (keepdims == 0)
                {
                    return op_node;
                }

                // WORKAROUND FOR PROBLEMS WITH RESHAPE ON i64 @TODO: remove
                auto convert_node = std::make_shared<ngraph::op::Convert>(op_node, element::f32);

                auto output_shape = input_node->get_shape();
                output_shape.at(axis) = 1;
                auto reshape_node = std::make_shared<ngraph::op::Reshape>(
                    convert_node,
                    reshape::get_default_axis_vector(op_node->get_shape().size()),
                    Shape{output_shape});

                // WORKAROUND FOR PROBLEMS WITH RESHAPE ON i64 @TODO: remove
                auto reconvert_node =
                    std::make_shared<ngraph::op::Convert>(reshape_node, element::i64);

                return reconvert_node;
            }

        } // namespace  reduction
    }     // namespace onnx_import
} // namespace ngraph
