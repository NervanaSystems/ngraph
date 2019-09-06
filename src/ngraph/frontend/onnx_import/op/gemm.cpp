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

#include <memory>

#include "gemm.hpp"
#include "ngraph/op/fused/gemm.hpp"

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

                    bool trans_a = node.get_attribute_value<int64_t>("transA", 0);
                    bool trans_b = node.get_attribute_value<int64_t>("transB", 0);

                    return NodeVector{std::make_shared<ngraph::op::Gemm>(
                        input_a, input_b, input_c, alpha, beta, trans_a, trans_b)};
                }

            } // namespace set_1

        } // namespace op

    } // namespace  onnx_import

} // namespace  ngraph
