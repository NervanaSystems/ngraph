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

#include "concat.hpp"

#include "ngraph/op/concat.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            NodeVector concat(const Node& node)
            {
                NodeVector inputs{node.get_ng_inputs()};
                auto axis = node.get_attribute_value<int64_t>("axis");

                return {std::make_shared<ngraph::op::Concat>(inputs, axis)};
            }

        } // namespace  op

    } // namespace  onnx_import

} // namespace  ngraph
