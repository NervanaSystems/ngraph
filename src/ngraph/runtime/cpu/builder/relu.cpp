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

#include "ngraph/runtime/cpu/kernel/relu.hpp"
#include "ngraph/op/relu.hpp"
#include "ngraph/runtime/cpu/cpu_builder.hpp"
#include "ngraph/runtime/cpu/dnnl_invoke.hpp"
#include "ngraph/runtime/cpu/dnnl_utils.hpp"

using namespace std;
using namespace ngraph;

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            template <>
            void Builder::BUILDER_DECL(ngraph::op::v0::Relu)
            {
                if (runtime::cpu::dnnl_utils::use_dnnl_kernel(node))
                {
                    auto& functors = external_function->get_functors();

                    auto arg_buffer_index = external_function->get_buffer_index(args[0].get_name());
                    auto out_buffer_index = external_function->get_buffer_index(out[0].get_name());

                    auto& dnnl_emitter = external_function->get_dnnl_emitter();
                    auto relu_desc = dnnl_emitter->get_relu_forward_desc(node);
                    size_t scratchpad_size = QUERY_SCRATCHPAD(eltwise_forward, relu_desc);

                    // Relu needs 3 primitives: input, result, and eltwise_forward.
                    size_t relu_index = dnnl_emitter->reserve_primitive_space(3);
                    auto& deps = dnnl_emitter->get_primitive_deps(relu_index);

                    auto functor = [&,
                                    relu_desc,
                                    relu_index,
                                    scratchpad_size,
                                    arg_buffer_index,
                                    out_buffer_index](CPURuntimeContext* ctx,
                                                      CPUExecutionContext* /* ectx */) {
                        if (ctx->first_iteration)
                        {
                            dnnl_emitter->build_relu_forward(ctx->dnnl_memories,
                                                             ctx->dnnl_primitives,
                                                             ctx->dnnl_scratchpad_mds,
                                                             relu_desc,
                                                             deps,
                                                             relu_index);
                        }
                        cpu::dnnl_utils::set_memory_ptr(
                            ctx, deps[0], ctx->buffer_data[arg_buffer_index]);
                        cpu::dnnl_utils::set_memory_ptr(
                            ctx, deps[1], ctx->buffer_data[out_buffer_index]);

                        cpu::dnnl_utils::dnnl_invoke_primitive(
                            ctx, relu_index, deps, cpu::dnnl_utils::OpType::RELU, scratchpad_size);
                    };
                    functors.emplace_back(functor);
                }
                else
                {
                    BUILD_UNARY_ELEMWISE_FUNCTOR(runtime::cpu::kernel::relu);
                }
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::v0::ReluBackprop)
            {
                auto& functors = external_function->get_functors();

                auto arg_fwd_buffer_index = external_function->get_buffer_index(args[0].get_name());
                auto delta_buffer_index = external_function->get_buffer_index(args[1].get_name());
                auto out_buffer_index = external_function->get_buffer_index(out[0].get_name());
                size_t count = out[0].get_size();

                if (runtime::cpu::dnnl_utils::use_dnnl_kernel(node))
                {
                    auto& dnnl_emitter = external_function->get_dnnl_emitter();
                    auto bwd_desc = dnnl_emitter->get_relu_backward_desc(node);
                    auto fwd_desc = dnnl_emitter->get_relu_forward_desc(node);
                    size_t scratchpad_size =
                        QUERY_SCRATCHPAD_2ARGS(eltwise_backward, fwd_desc, bwd_desc);

                    // ReluBackprop needs 4 primitives: input, delta, result, and eltwise_backward.
                    size_t relu_index = dnnl_emitter->reserve_primitive_space(4);
                    auto& deps = dnnl_emitter->get_primitive_deps(relu_index);

                    auto functor = [&,
                                    bwd_desc,
                                    fwd_desc,
                                    relu_index,
                                    scratchpad_size,
                                    arg_fwd_buffer_index,
                                    delta_buffer_index,
                                    out_buffer_index](CPURuntimeContext* ctx,
                                                      CPUExecutionContext* /* ectx */) {
                        if (ctx->first_iteration)
                        {
                            dnnl_emitter->build_relu_backward(ctx->dnnl_memories,
                                                              ctx->dnnl_primitives,
                                                              ctx->dnnl_scratchpad_mds,
                                                              bwd_desc,
                                                              fwd_desc,
                                                              deps,
                                                              relu_index);
                        }
                        cpu::dnnl_utils::set_memory_ptr(
                            ctx, deps[0], ctx->buffer_data[arg_fwd_buffer_index]);
                        cpu::dnnl_utils::set_memory_ptr(
                            ctx, deps[1], ctx->buffer_data[delta_buffer_index]);
                        cpu::dnnl_utils::set_memory_ptr(
                            ctx, deps[2], ctx->buffer_data[out_buffer_index]);

                        cpu::dnnl_utils::dnnl_invoke_primitive(
                            ctx,
                            relu_index,
                            deps,
                            cpu::dnnl_utils::OpType::RELUBACKPROP,
                            scratchpad_size);
                    };
                    functors.emplace_back(functor);
                }
                else
                {
                    std::function<decltype(runtime::cpu::kernel::relu_backprop<float>)> kernel;

                    SELECT_KERNEL(
                        kernel, out[0].get_element_type(), runtime::cpu::kernel::relu_backprop)

                    auto functor = [&,
                                    kernel,
                                    count,
                                    arg_fwd_buffer_index,
                                    delta_buffer_index,
                                    out_buffer_index](CPURuntimeContext* ctx,
                                                      CPUExecutionContext* ectx) {
                        kernel(ctx->buffer_data[arg_fwd_buffer_index],
                               ctx->buffer_data[delta_buffer_index],
                               ctx->buffer_data[out_buffer_index],
                               count,
                               ectx->arena);
                    };
                    functors.emplace_back(functor);
                }
            }

            void register_builders_relu_cpp()
            {
                REGISTER_OP_BUILDER(ngraph::op::v0::Relu);
                REGISTER_OP_BUILDER(ngraph::op::v0::ReluBackprop);
            }
        }
    }
}
