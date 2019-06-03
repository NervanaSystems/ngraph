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

#include "op/matmul_integer.hpp"
#include "ngraph/builder/quantization/quantized_linear_matmul.hpp"
#include "ngraph/frontend/onnx_import/exceptions.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                NodeVector matmul_integer(const Node& node)
                {
                    const NodeVector& ng_inputs{node.get_ng_inputs()};
                    auto num_inputs = ng_inputs.size();
                    auto input_a = ng_inputs.at(0);
                    auto input_b = ng_inputs.at(1);

                    if (num_inputs == 2)
                    {
                        return {
                            builder::quantization::QuantizedLinearMatmulInteger(input_a, input_b)};
                    }

                    throw error::NotSupported(
                        "MatMulInteger operation with customized zero-points not supported.");
                }
            } // namespace set_1

        } //namespace op

    } // namespace onnx_import

} // namespace ngraph
