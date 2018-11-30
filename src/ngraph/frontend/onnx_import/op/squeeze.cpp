//*****************************************************************************
// Copyright 2018 Intel Corporation
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

#include <cstddef>
#include <functional>
#include <iterator>
#include <numeric>
#include <set>
#include <vector>

#include "exceptions.hpp"
#include "ngraph/axis_vector.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/shape.hpp"
#include "squeeze.hpp"
#include "utils/reshape.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                NodeVector squeeze(const Node& node)
                {
                    NodeVector inputs{node.get_ng_inputs()};
                    auto data = inputs.at(0);
                    auto data_shape = data->get_shape();
                    auto axes = node.get_attribute_value<std::vector<std::size_t>>("axes", {});
                    AxisVector input_order{reshape::get_default_axis_vector(data_shape.size())};

                    // Prepare set of unique axes marked to be removed from input data.
                    if (axes.empty())
                    {
                        // Default behaviour is to remove all single dimension axes.
                        for (std::size_t idx = 0; idx < data_shape.size(); ++idx)
                        {
                            if (data_shape.at(idx) == 1)
                            {
                                // Mark with zero elements to remove;
                                data_shape.at(idx) = 0;
                            }
                        }
                    }
                    else
                    {
                        std::set<std::size_t, std::greater<std::size_t>> unique_axes(
                            std::begin(axes), std::end(axes));
                        for (uint64_t axis : unique_axes)
                        {
                            ASSERT_VALID_ARGUMENT(node, data_shape.at(axis) == 1)
                                << "provided axis value is invalid. Only single dimension axes may "
                                   "be removed.";
                            // Mark with zero elements to remove;
                            data_shape.at(axis) = 0;
                        }
                    }

                    Shape output_data_shape;
                    for (std::size_t idx = 0; idx < data_shape.size(); ++idx)
                    {
                        if (data_shape.at(idx) != 0)
                        {
                            output_data_shape.push_back(data_shape.at(idx));
                        }
                    }
                    return {std::make_shared<ngraph::op::Reshape>(
                        data, input_order, output_data_shape)};
                }

            } // namespace set_1
        }     //namespace op
    }         // namespace onnx_import
} // namespace ngraph
