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

#include "asinh.hpp"
#include "default_opset.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/shape.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                NodeVector asinh(const Node& node)
                {
                    std::shared_ptr<ngraph::Node> data{node.get_ng_inputs().at(0)};

                    // Define inverse hyperbolic sine in terms of natural logarithm:
                    //
                    // asinh(x) = ln(x + sqrt(x^2 + 1))
                    //

                    if (data->get_output_partial_shape(0).is_static())

                    {
                        std::shared_ptr<ngraph::Node> one_node{default_opset::Constant::create(
                        data->get_element_type(),
                        data->get_shape(),
                        std::vector<float>(ngraph::shape_size(data->get_shape()), 1.f))};

                        std::shared_ptr<ngraph::Node> sqrt_node{
                        std::make_shared<default_opset::Sqrt>(data * data + one_node)};

                        return {std::make_shared<default_opset::Log>(data + sqrt_node)};
                    }
                    else
                    {
                        const auto one_node = default_opset::Constant::create(
                            data->get_element_type(), {}, {1.f});
                        
                        // const auto shape_of_data =
                        //         std::make_shared<default_opset::ShapeOf>(data->get_output_partial_shape(0));
                                

                        // const auto broadcasted_ones = std::make_shared<default_opset::Broadcast>(
                        //         one_node, shape_of_data);
                                                        
                        const auto data_power = std::make_shared<default_opset::Multiply>(data, data);
                        const auto sqrt_args = std::make_shared<default_opset::Add>(data_power, one_node);
                        const auto sqrt_node = std::make_shared<default_opset::Sqrt>(sqrt_args);
                        const auto log_args = std::make_shared<default_opset::Add>(data, sqrt_node);

                        return {std::make_shared<default_opset::Log>(log_args)};
                    }


                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
