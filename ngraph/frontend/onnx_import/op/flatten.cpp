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

#include <cinttypes>

#include "exceptions.hpp"
#include "flatten.hpp"
#include "ngraph/builder/reshape.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                NodeVector flatten(const Node& node)
                {
                    NodeVector inputs{node.get_ng_inputs()};
                    auto data = inputs.at(0);
                    auto axis = node.get_attribute_value<std::int64_t>("axis", 1);

                    ASSERT_VALID_ARGUMENT(node, (axis >= 0) && (axis <= data->get_shape().size()))
                        << "provided 'axis' attribute is not valid.";

                    return {ngraph::builder::flatten(data, axis)};
                }

            } // namespace set_1

        } //namespace op

    } // namespace  onnx_import

} // namespace  ngraph
