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

#include "ngraph/runtime/cpu/op/quantized_matmul.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/runtime/cpu/cpu_builder.hpp"
#include "ngraph/runtime/cpu/cpu_executor.hpp"
#include "ngraph/runtime/cpu/kernel/dot.hpp"
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
            void Builder::BUILDER_DECL(ngraph::op::QuantizedMatmul)
            {
                auto& functors = external_function->get_functors();

                auto arg0_shape = args[0].get_shape();
                auto arg1_shape = args[1].get_shape();
                auto result_shape = out[0].get_shape();

                auto arg0_buffer_index = external_function->get_buffer_index(args[0].get_name());
                auto arg1_buffer_index = external_function->get_buffer_index(args[1].get_name());
                auto arg2_buffer_index =
                    external_function->get_buffer_index(args[2].get_name()); // scale
                auto out0_buffer_index = external_function->get_buffer_index(out[0].get_name());

                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();

                    auto ip_desc =
                        mkldnn_emitter->get_inner_product_forward_desc<ngraph::op::QuantizedMatmul>(
                            node);
                    auto ip_attr =
                        mkldnn_emitter->get_inner_product_forward_attr<ngraph::op::QuantizedMatmul>(
                            node);
                    size_t scratchpad_size = QUERY_SCRATCHPAD_2ARGS(ip_forward, ip_desc, ip_attr);

                    size_t ip_index = mkldnn_emitter->inner_product_forward_init(false);
                    auto& deps = mkldnn_emitter->get_primitive_deps(ip_index);

                    auto functor = [&,
                                    ip_desc,
                                    ip_attr,
                                    deps,
                                    ip_index,
                                    scratchpad_size,
                                    arg0_buffer_index,
                                    arg1_buffer_index,
                                    arg2_buffer_index,
                                    out0_buffer_index](CPURuntimeContext* ctx,
                                                       CPUExecutionContext* /* ectx */) mutable {
                        if (ctx->first_iteration)
                        {
                            vector<float> dyn_scales;
                            dyn_scales.push_back(
                                *(static_cast<float*>(ctx->buffer_data[arg2_buffer_index])));
                            ip_attr.set_output_scales(0, dyn_scales);
                            mkldnn_emitter->build_inner_product_forward<false>(
                                ctx->mkldnn_memories,
                                ctx->mkldnn_primitives,
                                ctx->mkldnn_scratchpad_mds,
                                ip_desc,
                                ip_attr,
                                executor::global_cpu_engine,
                                deps,
                                ip_index);
                        }
                        cpu::mkldnn_utils::set_memory_ptr(
                            ctx, deps[0], ctx->buffer_data[arg0_buffer_index]);
                        cpu::mkldnn_utils::set_memory_ptr(
                            ctx, deps[1], ctx->buffer_data[arg1_buffer_index]);
                        cpu::mkldnn_utils::set_memory_ptr(
                            ctx, deps[2], ctx->buffer_data[out0_buffer_index]);

                        cpu::mkldnn_utils::mkldnn_invoke_primitive(
                            ctx,
                            ip_index,
                            deps,
                            cpu::mkldnn_utils::OpType::QUANTIZEDMATMUL,
                            scratchpad_size);
                    };
                    functors.emplace_back(functor);
                }
                else
                {
                    throw ngraph_error("Unsupported QuantizedMatmul");
                }
            }

            void register_builders_quantized_matmul_cpp() { REGISTER_OP_BUILDER(QuantizedMatmul); }
        }
    }
}
