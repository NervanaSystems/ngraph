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

#include "op/gemm.hpp"

#include "ngraph/op/add.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/dot.hpp"
#include "ngraph/op/multiply.hpp"

#include "ngraph/frontend/onnx_import/exceptions.hpp"
#include "ngraph/frontend/onnx_import/utils/broadcasting.hpp"
#include "ngraph/frontend/onnx_import/utils/reshape.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                NodeVector gemm(const Node& node)
                {
                    NodeVector inputs{node.get_ng_inputs()};
                    auto input_a = inputs.at(0);
                    auto input_b = inputs.at(1);
                    auto input_c = inputs.at(2);

                    double alpha = node.get_attribute_value<double>("alpha", 1);
                    double beta = node.get_attribute_value<double>("beta", 1);

                    auto trans_a = node.get_attribute_value<int64_t>("transA", 0);
                    auto trans_b = node.get_attribute_value<int64_t>("transB", 0);

                    if (trans_a != 0)
                    {
                        input_a = reshape::transpose(input_a);
                    }
                    if (trans_b != 0)
                    {
                        input_b = reshape::transpose(input_b);
                    }

                    // code from python not implemented in c++ yet.
                    // reshape_for_matmul(node, input_a, input_b);

                    // A' * B'
                    std::shared_ptr<ngraph::Node> a_dot_b =
                        std::make_shared<ngraph::op::Dot>(input_a, input_b);

                    // alpha
                    std::shared_ptr<ngraph::Node> alpha_node =
                        std::make_shared<ngraph::op::Constant>(a_dot_b->get_element_type(),
                                                               ngraph::Shape{},
                                                               std::vector<double>{alpha});
                    alpha_node = make_broadcast_node(alpha_node, a_dot_b->get_shape());
                    // alpha * A' * B'
                    a_dot_b = std::make_shared<ngraph::op::Multiply>(alpha_node, a_dot_b);

                    // beta * C
                    std::shared_ptr<ngraph::Node> beta_node =
                        std::make_shared<ngraph::op::Constant>(input_c->get_element_type(),
                                                               ngraph::Shape{},
                                                               std::vector<double>{beta});
                    beta_node = make_broadcast_node(beta_node, input_c->get_shape());
                    input_c = std::make_shared<ngraph::op::Multiply>(beta_node, input_c);

                    // alpha * A' * B' + beta * C
                    NodeVector broadcasted_nodes =
                        numpy_style_broadcast_for_binary_operation(a_dot_b, input_c);
                    // The ONNX documentation says that `input_c` should be "unidirectional broadcastable"
                    // to the `a_dot_b` tensor. Since numpy style broadcasting is bidirectional, below we
                    // only use the second output from above broadcasting. In other words we want to
                    // preserve the shape of original `a_dot_b` tensor.
                    return {std::make_shared<ngraph::op::Add>(a_dot_b, broadcasted_nodes.at(1))};
                }

            } // namespace set_1

        } //namespace op

    } // namespace  onnx_import

} // namespace  ngraph
