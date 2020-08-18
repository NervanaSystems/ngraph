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

#include "ngraph/op/gelu.hpp"
#include "ngraph/runtime/cpu/cpu_builder.hpp"
#include "ngraph/runtime/cpu/dnnl_emitter.hpp"
#include "ngraph/runtime/cpu/dnnl_invoke.hpp"
#include "ngraph/runtime/cpu/dnnl_utils.hpp"
#include "ngraph/runtime/cpu/op/gelu_backprop.hpp"

using namespace std;
using namespace ngraph;

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            template <>
            void Builder::BUILDER_DECL(ngraph::op::v0::Gelu)
            {
                auto& functors = external_function->get_functors();

                auto input_buffer_index = external_function->get_buffer_index(args[0].get_name());
                auto out_buffer_index = external_function->get_buffer_index(out[0].get_name());

                if (runtime::cpu::dnnl_utils::use_dnnl_kernel(node))
                {
                    auto& dnnl_emitter = external_function->get_dnnl_emitter();
                    auto gelu_desc = dnnl_emitter->get_gelu_forward_desc(node);
                    size_t scratchpad_size = QUERY_SCRATCHPAD(eltwise_forward, gelu_desc);

                    // Gelu needs 3 primitives: input, result, and eltwise_forward
                    auto gelu_index = dnnl_emitter->reserve_primitive_space(3);
                    auto& deps = dnnl_emitter->get_primitive_deps(gelu_index);

                    auto functor = [&,
                                    gelu_desc,
                                    gelu_index,
                                    scratchpad_size,
                                    input_buffer_index,
                                    out_buffer_index](CPURuntimeContext* ctx,
                                                      CPUExecutionContext* /* ectx */) {
                        if (ctx->first_iteration)
                        {
                            dnnl_emitter->build_gelu(ctx->dnnl_memories,
                                                     ctx->dnnl_primitives,
                                                     ctx->dnnl_scratchpad_mds,
                                                     gelu_desc,
                                                     deps,
                                                     gelu_index);
                        }
                        cpu::dnnl_utils::set_memory_ptr(
                            ctx, deps[0], ctx->buffer_data[input_buffer_index]);
                        cpu::dnnl_utils::set_memory_ptr(
                            ctx, deps[1], ctx->buffer_data[out_buffer_index]);

                        cpu::dnnl_utils::dnnl_invoke_primitive(
                            ctx, gelu_index, deps, cpu::dnnl_utils::OpType::GELU, scratchpad_size);
                    };
                    functors.emplace_back(functor);
                }
                else
                {
                    throw ngraph_error("Gelu is supported with DNNL kernel only for f32.");
                }
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::GeluBackprop)
            {
                auto& functors = external_function->get_functors();

                auto arg_fwd_buffer_index = external_function->get_buffer_index(args[0].get_name());
                auto delta_buffer_index = external_function->get_buffer_index(args[1].get_name());
                auto out_buffer_index = external_function->get_buffer_index(out[0].get_name());

                if (runtime::cpu::dnnl_utils::use_dnnl_kernel(node))
                {
                    auto& dnnl_emitter = external_function->get_dnnl_emitter();
                    auto bwd_desc = dnnl_emitter->get_gelu_backward_desc(node);
                    auto fwd_desc = dnnl_emitter->get_gelu_forward_desc(node);
                    size_t scratchpad_size =
                        QUERY_SCRATCHPAD_2ARGS(eltwise_backward, fwd_desc, bwd_desc);

                    // geluBackprop needs 4 primitives: input, delta, result, and eltwise_backward.
                    size_t gelu_b_index = dnnl_emitter->reserve_primitive_space(4);
                    auto& deps = dnnl_emitter->get_primitive_deps(gelu_b_index);

                    auto functor = [&,
                                    bwd_desc,
                                    fwd_desc,
                                    gelu_b_index,
                                    scratchpad_size,
                                    arg_fwd_buffer_index,
                                    delta_buffer_index,
                                    out_buffer_index](CPURuntimeContext* ctx,
                                                      CPUExecutionContext* /* ectx */) {
                        if (ctx->first_iteration)
                        {
                            dnnl_emitter->build_gelu_backward(ctx->dnnl_memories,
                                                              ctx->dnnl_primitives,
                                                              ctx->dnnl_scratchpad_mds,
                                                              bwd_desc,
                                                              fwd_desc,
                                                              deps,
                                                              gelu_b_index);
                        }
                        cpu::dnnl_utils::set_memory_ptr(
                            ctx, deps[0], ctx->buffer_data[arg_fwd_buffer_index]);
                        cpu::dnnl_utils::set_memory_ptr(
                            ctx, deps[1], ctx->buffer_data[delta_buffer_index]);
                        cpu::dnnl_utils::set_memory_ptr(
                            ctx, deps[2], ctx->buffer_data[out_buffer_index]);

                        cpu::dnnl_utils::dnnl_invoke_primitive(
                            ctx,
                            gelu_b_index,
                            deps,
                            cpu::dnnl_utils::OpType::GELUBACKPROP,
                            scratchpad_size);
                    };
                    functors.emplace_back(functor);
                }
                else
                {
                    throw ngraph_error("GeluBackprop is supported only for f32 with dnnl.");
                }
            }

            void register_builders_gelu_cpp()
            {
                REGISTER_OP_BUILDER(ngraph::op::v0::Gelu);
                REGISTER_OP_BUILDER(ngraph::op::GeluBackprop);
            }
        }
    }
}
