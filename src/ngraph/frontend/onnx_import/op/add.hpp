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

#include "ngraph/node_vector.hpp"
#include "ngraph/op/add.hpp"

#include "core/node.hpp"
#include "utils/broadcasting.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            inline NodeVector add(const Node& node)
            {
                NodeVector ng_inputs{
                    numpy_style_broadcast_for_binary_operation(node.get_ng_inputs())};
                return {std::make_shared<ngraph::op::Add>(ng_inputs.at(0), ng_inputs.at(1))};
            }

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
