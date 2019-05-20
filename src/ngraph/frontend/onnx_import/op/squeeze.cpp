//*****************************************************************************
// Copyright 2018-2019 Intel Corporation
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

#include <vector>

#include "exceptions.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/fused/squeeze.hpp"
#include "squeeze.hpp"

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
                    auto data = node.get_ng_inputs().at(0);
                    auto axes = node.get_attribute_value<std::vector<std::int64_t>>("axes", {});

                    for (auto axis : axes)
                    {
                        ASSERT_VALID_ARGUMENT(node, axis >= 0)
                            << "provided axes attribute is invalid. Only non-negative "
                            << "integers are allowed, got " << axis << ".";
                    }

                    auto axes_node = std::make_shared<ngraph::op::Constant>(
                        element::u64, Shape{axes.size()}, axes);
                    return {std::make_shared<ngraph::op::Squeeze>(data, axes_node)};
                }

            } // namespace set_1
        }     //namespace op
    }         // namespace onnx_import
} // namespace ngraph
