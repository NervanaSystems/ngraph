//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#include "default_opset.hpp"
#include "ngraph/validation_util.hpp"
#include "softmax.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                NodeVector softmax(const Node& node)
                {
                    const auto data = node.get_ng_inputs().at(0);
                    const auto data_rank = data->get_output_partial_shape(0).rank();

                    NGRAPH_CHECK(
                        data_rank.is_static(),
                        "The input data tensor's rank of the Softmax op must be known(static).");

                    const auto axis = node.get_attribute_value<int64_t>("axis", 1);
                    const auto normalized_axis = ngraph::normalize_axis(
                        node.get_description(), axis, static_cast<size_t>(data_rank));

                    return {std::make_shared<default_opset::Softmax>(data, normalized_axis)};
                }
            }
        }
    }
}
