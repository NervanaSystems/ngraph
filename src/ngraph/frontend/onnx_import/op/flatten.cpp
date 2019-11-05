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

#include <cinttypes>

#include "exceptions.hpp"
#include "flatten.hpp"
#include "ngraph/builder/reshape.hpp"
#include "utils/common.hpp"

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
                    auto data_rank = data->get_shape().size();
                    // Accepted range is [-r, r] where r = rank(input).
                    auto valid_axis =
                        common::validate_axis(node, axis, data_rank, -data_rank, data_rank);

                    return {ngraph::builder::flatten(data, valid_axis)};
                }

            } // namespace set_1

        } // namespace op

    } // namespace  onnx_import

} // namespace  ngraph
