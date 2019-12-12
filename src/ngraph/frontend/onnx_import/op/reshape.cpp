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

#include <cstddef>
#include <memory>
#include <vector>

#include "exceptions.hpp"
#include "ngraph/axis_vector.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/shape.hpp"
#include "reshape.hpp"
#include "utils/reshape.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                NodeVector reshape(const Node& node)
                {
                    NodeVector ng_inputs{node.get_ng_inputs()};
                    const auto data = ng_inputs.at(0);

                    std::shared_ptr<ngraph::Node> pattern;

                    // Since opset 5 the target shape is provided as input
                    if (ng_inputs.size() == 2)
                    {
                        NGRAPH_CHECK(ng_inputs.at(1)->is_constant(),
                                     "The target shape input has to be a Constant.");

                        pattern = ng_inputs.at(1);
                    }
                    else
                    {
                        const auto output_shape =
                            node.get_attribute_value<std::vector<int64_t>>("shape", {});

                        pattern = ngraph::op::Constant::create(
                            element::i64, Shape{output_shape.size()}, output_shape);
                    }

                    return {std::make_shared<ngraph::op::v1::Reshape>(data, pattern, true)};
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
