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

#include "ngraph/runtime/cpu/op/dequantize.hpp"
#include <vector>
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
            void Builder::BUILDER_DECL(ngraph::op::Dequantize)
            {
                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    auto& functors = external_function->get_functors();
                    auto& arg_tensor = external_function->get_tensor_data(args[0].get_name());
                    auto& out_tensor = external_function->get_tensor_data(out[0].get_name());
                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();

                    auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);

                    size_t dequantize_index =
                        mkldnn_emitter->build_dequantization(node, input_desc, result_desc);

                    auto& deps = mkldnn_emitter->get_primitive_deps(dequantize_index);
                    auto functor = [&, dequantize_index](CPURuntimeContext* ctx) {
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[0], arg_tensor);
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[1], out_tensor);
                        cpu::mkldnn_utils::mkldnn_invoke_primitive(ctx, dequantize_index);
                    };
                    functors.emplace_back(functor);
                }
                else
                {
                    throw ngraph_error("unsupported parameters for DequantizeOp via DEX");
                }
            }
            REGISTER_OP_BUILDER(Dequantize);
        }
    }
}
