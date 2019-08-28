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

#pragma once

#include "core/node.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/util/broadcasting.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                inline NodeVector add(const Node& node)
                {
                    auto left_rank = node.get_ng_inputs().at(0)->get_shape().size();
                    auto right_rank = node.get_ng_inputs().at(1)->get_shape().size();
                    auto axis =
                        node.get_attribute_value<std::int64_t>("axis", left_rank - right_rank);
                    NodeVector ng_inputs{ngraph::op::legacy_style_broadcast_for_binary_operation(
                        node.get_ng_inputs().at(0), node.get_ng_inputs().at(1), axis)};

                    return {std::make_shared<ngraph::op::Add>(ng_inputs.at(0), ng_inputs.at(1))};
                }

            } // namespace set_1

            namespace set_7
            {
                inline NodeVector add(const Node& node)
                {
                    NodeVector ng_inputs{ngraph::op::numpy_style_broadcast(node.get_ng_inputs())};
                    return {std::make_shared<ngraph::op::Add>(ng_inputs.at(0), ng_inputs.at(1))};
                }

            } // namespace set_7

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
