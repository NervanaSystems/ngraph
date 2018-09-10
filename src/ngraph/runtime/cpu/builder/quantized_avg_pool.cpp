/*******************************************************************************
* Copyright 2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "ngraph/runtime/cpu/op/quantized_avg_pool.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/runtime/cpu/cpu_builder.hpp"
#include "ngraph/runtime/cpu/mkldnn_invoke.hpp"
#include "ngraph/runtime/cpu/mkldnn_utils.hpp"

using namespace std;
using namespace ngraph;

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            template <>
            void Builder::BUILDER_DECL(ngraph::op::QuantizedAvgPool)
            {
                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    auto& functors = external_function->get_functors();
                    auto& arg_tensor = external_function->get_tensor_data(args[0].get_name());
                    auto& out_tensor = external_function->get_tensor_data(out[0].get_name());
                    auto& out1_tensor = external_function->get_tensor_data(out[1].get_name());
                    auto& out2_tensor = external_function->get_tensor_data(out[2].get_name());
                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();

                    vector<float> quant_util;
                    mkldnn_emitter->build_quantized_avg_pool(node, quant_util);
                    auto& deps = mkldnn_emitter->get_primitive_deps(quant_util[2]);

                    auto functor = [&, quant_util](CPURuntimeContext* ctx) {
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[0], arg_tensor);
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[1], out_tensor);
                        *(static_cast<float*>(out1_tensor)) = quant_util[0];
                        *(static_cast<float*>(out2_tensor)) = quant_util[1];
                        cpu::mkldnn_utils::mkldnn_invoke_primitive(ctx, quant_util[2]);
                    };
                    functors.emplace_back(functor);
                }
                else
                {
                    throw ngraph_error("unsupported parameters for QuantizedAvgPool via DEX");
                }
            }
            REGISTER_OP_BUILDER(QuantizedAvgPool);
        }
    }
}
