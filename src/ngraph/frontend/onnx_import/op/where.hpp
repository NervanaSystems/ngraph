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

#include <memory>

#include "core/node.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/util/broadcasting.hpp"
#include "ngraph/opsets/opset0.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                inline NodeVector where(const Node& node)
                {
                    NodeVector ng_inputs{ngraph::op::numpy_style_broadcast(node.get_ng_inputs())};

                    return {std::make_shared<ngraph::opset0::Select>(
                        ng_inputs.at(0), ng_inputs.at(1), ng_inputs.at(2))};
                }
            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
