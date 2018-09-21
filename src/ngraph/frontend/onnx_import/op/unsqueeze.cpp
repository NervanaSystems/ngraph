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

#include <numeric>

#include "ngraph/op/reshape.hpp"
#include "utils/reshape.hpp"

#include "exceptions.hpp"
#include "unsqueeze.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            NodeVector unsqueeze(const Node& node)
            {
                NodeVector inputs{node.get_ng_inputs()};
                auto data = inputs.at(0);
                auto data_shape = data->get_shape();
                auto axes = node.get_attribute_value<std::vector<int64_t>>("axes");

                ASSERT_VALID_ARGUMENT(node, !axes.empty()) << "'axes' attribute is mandatory.";

                std::sort(std::begin(axes), std::end(axes), std::less<int64_t>());

                AxisVector input_order{reshape::get_default_axis_vector(data_shape.size())};

                for (auto axis : axes)
                {
                    ASSERT_VALID_ARGUMENT(node, axis >= 0 && axis <= data_shape.size())
                        << "provided 'axes' attribute is not valid.";

                    data_shape.insert(std::next(std::begin(data_shape), axis), 1);
                }

                return {std::make_shared<ngraph::op::Reshape>(data, input_order, data_shape)};
            }

        } // namespace  op

    } // namespace  onnx_import

} // namespace  ngraph
