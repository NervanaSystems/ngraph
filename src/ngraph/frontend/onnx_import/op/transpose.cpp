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

#include <memory>
#include <vector>

#include "ngraph/node.hpp"

#include "transpose.hpp"
#include "utils/reshape.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            NodeVector transpose(const Node& node)
            {
                std::shared_ptr<ngraph::Node> data = node.get_ng_inputs().at(0);

                auto permute_axes = node.get_attribute_value<std::vector<std::size_t>>("perm", {});

                return {(permute_axes.empty()) ? reshape::transpose(data)
                                               : reshape::reorder_axes(data, permute_axes)};
            }

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
