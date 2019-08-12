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

#include "ngraph/op/experimental/quantized_max_pool.hpp"
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
            void Builder::BUILDER_DECL(ngraph::op::QuantizedMaxPool)
            {
                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    auto& functors = external_function->get_functors();

                    auto arg_buffer_index = external_function->get_buffer_index(args[0].get_name());
                    auto out_buffer_index = external_function->get_buffer_index(out[0].get_name());

                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto qmax_pool_desc =
                        mkldnn_emitter->get_max_pooling_forward_desc<ngraph::op::QuantizedMaxPool>(
                            node, false);
                    // QuantizedMaxPool needs 3 primitives: input, result, and pooling_forward.
                    size_t qmax_pool_index = mkldnn_emitter->reserve_primitive_space(3);
                    auto& deps = mkldnn_emitter->get_primitive_deps(qmax_pool_index);

                    auto functor =
                        [&, qmax_pool_desc, qmax_pool_index, arg_buffer_index, out_buffer_index](
                            CPURuntimeContext* ctx, CPUExecutionContext* ectx) {
                            if (ctx->first_iteration)
                            {
                                mkldnn_emitter->build_pooling_forward(
                                    ctx->mkldnn_primitives, qmax_pool_desc, deps, qmax_pool_index);
                            }
                            cpu::mkldnn_utils::set_memory_ptr(
                                ctx, deps[0], ctx->buffer_data[arg_buffer_index]);
                            cpu::mkldnn_utils::set_memory_ptr(
                                ctx, deps[1], ctx->buffer_data[out_buffer_index]);
                            cpu::mkldnn_utils::mkldnn_invoke_primitive(ctx, qmax_pool_index);
                        };
                    functors.emplace_back(functor);
                }
                else
                {
                    throw ngraph_error("unsupported parameters for QuantizedMaxPool via DEX");
                }
            }
            REGISTER_OP_BUILDER(QuantizedMaxPool);
#ifdef NGRAPH_CPU_STATIC_LIB_ENABLE
            void register_builders_quantized_max_pool_cpp() {}
#endif
        }
    }
}
