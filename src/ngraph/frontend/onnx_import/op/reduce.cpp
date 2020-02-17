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

#include <cstddef>  // std::size_t
#include <iterator> // std::begin, std::end
#include <numeric>  // std::accumulate

#include "default_opset.hpp"
#include "ngraph/shape.hpp"
#include "reduce.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                NodeVector reduce_mean(const Node& node)
                {
                    const auto data = node.get_ng_inputs().at(0);
                    const auto& data_shape = data->get_output_partial_shape(0);

                    // sum up the input data along the reduction axes
                    const auto sum_node = reduction::make_ng_reduction_op(
                        node,
                        data,
                        std::make_shared<default_opset::ReduceSum,
                                         const std::shared_ptr<ngraph::Node>&,
                                         const std::shared_ptr<ngraph::Node>&,
                                         bool>);

                    // calculate the product of dimensions pointed to by reduction axes
                    size_t reduced_elems_count = 1U;

                    if (data_shape.is_static())
                    {
                        const auto input_shape = data_shape.to_shape();

                        // calculate the product of dimensions pointed to by reduction axes
                        // this value represents the number of input tensor values that were reduced
                        for (const auto axis : reduction::detail::get_reduction_axes(node))
                        {
                            reduced_elems_count *= input_shape.at(axis);
                        }
                    }
                    else
                    {
                        for (const auto axis : reduction::detail::get_reduction_axes(node))
                        {
                            const auto dim_to_reduce = data_shape[axis];
                            NGRAPH_CHECK(dim_to_reduce.is_static(),
                                         "Axis ",
                                         axis,
                                         " in the input data tensor needs to be statically "
                                         "specified to create a ReduceMean operation");

                            reduced_elems_count *= static_cast<size_t>(dim_to_reduce);
                        }
                    }

                    const auto const_node = default_opset::Constant::create(
                        sum_node->get_element_type(), {}, {reduced_elems_count});

                    // divide the sum node containing reduced values by the number
                    // of those values to obtain the mean
                    return {std::make_shared<default_opset::Divide>(sum_node, const_node)};
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
