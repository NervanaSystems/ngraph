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

#include "flatten.hpp"

#include "ngraph/frontend/onnx_import/exceptions.hpp"
#include "ngraph/frontend/onnx_import/utils/reshape.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            NodeVector flatten(const Node& node)
            {
                NodeVector inputs{node.get_ng_inputs()};

                auto data = inputs.at(0);

                auto axis = node.get_attribute_value<int64_t>("axis", 1);

                if (axis < 0 || axis > data->get_shape().size())
                {
                    throw error::parameter::Value("Flatten node (",
                                                  node.get_name(),
                                                  "): provided axis attribute is not valid.");
                }

                return {utils::flatten(data, axis)};
            }

        } // namespace  op

    } // namespace  onnx_import

} // namespace  ngraph
