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

#include "ngraph/axis_vector.hpp"
#include "ngraph/op/reshape.hpp"
#include "utils/reshape.hpp"

#include "exceptions.hpp"
#include "squeeze.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            NodeVector squeeze(const Node& node)
            {
                NodeVector inputs{node.get_ng_inputs()};
                auto data = inputs.at(0);
                auto data_shape = data->get_shape();
                auto axes = node.get_attribute_value<std::vector<std::size_t>>("axes", {});
                AxisVector input_order{reshape::get_default_axis_vector(data_shape.size())};

                // Default behaviour is to squeeze all single dimension axes.
                if (axes.empty())
                {
                    auto it = std::begin(data_shape);
                    while (it != std::end(data_shape))
                    {
                        if (*it == 1)
                        {
                            data_shape.erase(it);
                            continue;
                        }
                        ++it;
                    }
                }
                else
                {
                    std::set<std::size_t, std::greater<std::size_t>> unique_axes(std::begin(axes),
                                                                                 std::end(axes));
                    for (uint64_t axis : unique_axes)
                    {
                        ASSERT_VALID_ARGUMENT(node, data_shape.at(axis) == 1)
                            << "provided axis value is invalid. Only single dimension axes may "
                               "be removed.";
                        data_shape.erase(std::next(std::begin(data_shape), axis));
                    }
                }

                return {std::make_shared<ngraph::op::Reshape>(data, input_order, data_shape)};
            }

        } // namespace  op

    } // namespace  onnx_import

} // namespace  ngraph
