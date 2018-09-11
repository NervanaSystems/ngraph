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

#include <algorithm>
#include <iterator>
#include <memory>

#include "ngraph/node.hpp"
#include "ngraph/node_vector.hpp"

#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/maximum.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/reshape.hpp"

#include "core/node.hpp"
#include "utils/broadcasting.hpp"

#include "prelu.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            NodeVector prelu(const Node& node)
            {
                NodeVector ng_inputs{node.get_ng_inputs()};
                auto data = ng_inputs.at(0);
                auto data_shape = data->get_shape();
                std::shared_ptr<ngraph::Node> slope = ng_inputs.at(1);
                auto slope_shape = slope->get_shape();

                if ((slope_shape.size() == 1) && (slope_shape.at(0) != 1))
                {
                    auto it =
                        std::find(std::begin(data_shape), std::end(data_shape), slope_shape.at(0));
                    auto index = std::distance(std::begin(data_shape), it);
                    slope = make_broadcast_node(slope, data->get_shape(), index);
                }
                else
                {
                    auto params = numpy_style_broadcast_for_binary_operation(slope, data);
                    slope = params.at(0);
                }

                return {std::make_shared<ngraph::op::Maximum>(data * slope, data)};
            }

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
