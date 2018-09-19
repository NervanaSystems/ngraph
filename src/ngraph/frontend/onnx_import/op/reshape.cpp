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

#include <cstddef>
#include <memory>
#include <vector>

#include "ngraph/axis_vector.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/shape.hpp"

#include "exceptions.hpp"
#include "reshape.hpp"
#include "utils/reshape.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            NodeVector reshape(const Node& node)
            {
                NodeVector ng_inputs{node.get_ng_inputs()};
                auto data = ng_inputs.at(0);
                auto data_shape = data->get_shape();

                auto output_shape = node.get_attribute_value<std::vector<std::size_t>>("shape", {});

                // If no shape argument (opset >= 5) and there is second input.
                if (output_shape.empty() && ng_inputs.size() == 2)
                {
                    // Currently only support Constant node.
                    ASSERT_IS_SUPPORTED(node, ng_inputs.at(1)->description() == "Constant")
                        << "doesn't support shape input of other type than Constant.";

                    auto output_shape_node =
                        std::dynamic_pointer_cast<ngraph::op::Constant>(ng_inputs.at(1));
                    output_shape = output_shape_node->get_vector<std::size_t>();
                }
                // Do nothing if there is no shape argument nor second node input.
                else if (output_shape.empty())
                {
                    return {data};
                }

                output_shape = reshape::infer_dimensions(node.get_name(), data_shape, output_shape);
                return {std::make_shared<ngraph::op::Reshape>(
                    data,
                    reshape::get_default_axis_vector(data_shape.size()),
                    Shape{output_shape})};
            }

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
