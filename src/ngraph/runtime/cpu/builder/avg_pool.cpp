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

#include "ngraph/runtime/cpu/kernel/avg_pool.hpp"
#include "ngraph/op/avg_pool.hpp"
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
            void Builder::BUILDER_DECL(ngraph::op::v0::AvgPool)
            {
                auto avg_pool = static_cast<const ngraph::op::v0::AvgPool*>(node);

                auto& functors = external_function->get_functors();

                auto arg0_shape = args[0].get_shape();
                auto out_shape = out[0].get_shape();

                auto arg0_buffer_index = external_function->get_buffer_index(args[0].get_name());
                auto out_buffer_index = external_function->get_buffer_index(out[0].get_name());

                auto window_shape = avg_pool->get_window_shape();
                auto window_movement_strides = avg_pool->get_window_movement_strides();
                auto padding_below = avg_pool->get_padding_below();
                auto padding_above = avg_pool->get_padding_above();
                auto include_padding_in_avg_computation =
                    avg_pool->get_include_padding_in_avg_computation();

                if (runtime::cpu::dnnl_utils::use_dnnl_kernel(node))
                {
                    auto& dnnl_emitter = external_function->get_dnnl_emitter();
                    auto avg_pool_desc =
                        dnnl_emitter->get_avg_pooling_forward_desc<ngraph::op::v0::AvgPool>(node,
                                                                                            false);
                    size_t scratchpad_size = QUERY_SCRATCHPAD(pooling_forward, avg_pool_desc);

                    // AvgPool needs 3 primitives: input, result, and pooling_forward.
                    size_t avg_pool_index = dnnl_emitter->reserve_primitive_space(3);
                    auto& deps = dnnl_emitter->get_primitive_deps(avg_pool_index);

                    auto functor = [&,
                                    avg_pool_desc,
                                    avg_pool_index,
                                    scratchpad_size,
                                    arg0_buffer_index,
                                    out_buffer_index](CPURuntimeContext* ctx,
                                                      CPUExecutionContext* /* ectx */) {
                        if (ctx->first_iteration)
                        {
                            dnnl_emitter->build_pooling_forward(ctx->dnnl_memories,
                                                                ctx->dnnl_primitives,
                                                                ctx->dnnl_scratchpad_mds,
                                                                avg_pool_desc,
                                                                deps,
                                                                avg_pool_index);
                        }
                        cpu::dnnl_utils::set_memory_ptr(
                            ctx, deps[0], ctx->buffer_data[arg0_buffer_index]);
                        cpu::dnnl_utils::set_memory_ptr(
                            ctx, deps[1], ctx->buffer_data[out_buffer_index]);

                        cpu::dnnl_utils::dnnl_invoke_primitive(ctx,
                                                               avg_pool_index,
                                                               deps,
                                                               cpu::dnnl_utils::OpType::AVGPOOL,
                                                               scratchpad_size);
                    };
                    functors.emplace_back(functor);
                }
                else
                {
                    std::function<decltype(runtime::cpu::kernel::avg_pool<float>)> kernel;

                    SELECT_KERNEL(kernel, out[0].get_element_type(), runtime::cpu::kernel::avg_pool)

                    auto functor = [&,
                                    kernel,
                                    arg0_shape,
                                    out_shape,
                                    window_shape,
                                    window_movement_strides,
                                    padding_below,
                                    padding_above,
                                    include_padding_in_avg_computation,
                                    arg0_buffer_index,
                                    out_buffer_index](CPURuntimeContext* ctx,
                                                      CPUExecutionContext* /* ectx */) {
                        kernel(ctx->buffer_data[arg0_buffer_index],
                               ctx->buffer_data[out_buffer_index],
                               arg0_shape,
                               out_shape,
                               window_shape,
                               window_movement_strides,
                               padding_below,
                               padding_above,
                               include_padding_in_avg_computation);
                    };
                    functors.emplace_back(functor);
                }
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::v0::AvgPoolBackprop)
            {
                auto apb = static_cast<const ngraph::op::v0::AvgPoolBackprop*>(node);

                auto& functors = external_function->get_functors();

                auto delta_shape = args[0].get_shape();
                auto out_shape = out[0].get_shape();

                auto delta_buffer_index = external_function->get_buffer_index(args[0].get_name());
                auto out_buffer_index = external_function->get_buffer_index(out[0].get_name());

                auto window_shape = apb->get_window_shape();
                auto window_movement_strides = apb->get_window_movement_strides();
                auto padding_below = apb->get_padding_below();
                auto padding_above = apb->get_padding_above();
                auto include_padding_in_avg_computation =
                    apb->get_include_padding_in_avg_computation();

                if (runtime::cpu::dnnl_utils::use_dnnl_kernel(node))
                {
                    auto& dnnl_emitter = external_function->get_dnnl_emitter();
                    auto avg_pool_fwd_desc =
                        dnnl_emitter->get_avg_pooling_forward_desc<ngraph::op::v0::AvgPoolBackprop>(
                            node, true);
                    auto avg_pool_desc =
                        dnnl_emitter
                            ->get_avg_pooling_backward_desc<ngraph::op::v0::AvgPoolBackprop>(node);
                    size_t scratchpad_size = QUERY_SCRATCHPAD_2ARGS(
                        avg_pooling_backward, avg_pool_fwd_desc, avg_pool_desc);

                    // AvgPoolBackprop needs 3 primitives: input, result, and pooling_backward.
                    size_t avg_pool_index = dnnl_emitter->reserve_primitive_space(3);
                    auto& deps = dnnl_emitter->get_primitive_deps(avg_pool_index);

                    auto functor = [&,
                                    avg_pool_desc,
                                    avg_pool_fwd_desc,
                                    avg_pool_index,
                                    scratchpad_size,
                                    delta_buffer_index,
                                    out_buffer_index](CPURuntimeContext* ctx,
                                                      CPUExecutionContext* /* ectx */) {
                        if (ctx->first_iteration)
                        {
                            dnnl_emitter->build_pooling_backward(ctx->dnnl_memories,
                                                                 ctx->dnnl_primitives,
                                                                 ctx->dnnl_scratchpad_mds,
                                                                 avg_pool_desc,
                                                                 avg_pool_fwd_desc,
                                                                 deps,
                                                                 avg_pool_index);
                        }
                        cpu::dnnl_utils::set_memory_ptr(
                            ctx, deps[0], ctx->buffer_data[delta_buffer_index]);
                        cpu::dnnl_utils::set_memory_ptr(
                            ctx, deps[1], ctx->buffer_data[out_buffer_index]);

                        cpu::dnnl_utils::dnnl_invoke_primitive(
                            ctx,
                            avg_pool_index,
                            deps,
                            cpu::dnnl_utils::OpType::AVGPOOLBACKPROP,
                            scratchpad_size);
                    };
                    functors.emplace_back(functor);
                }
                else
                {
                    std::function<decltype(runtime::cpu::kernel::avg_pool_backprop<float>)> kernel;
                    SELECT_KERNEL(
                        kernel, out[0].get_element_type(), runtime::cpu::kernel::avg_pool_backprop)

                    auto functor = [&,
                                    kernel,
                                    delta_shape,
                                    out_shape,
                                    window_shape,
                                    window_movement_strides,
                                    padding_below,
                                    padding_above,
                                    include_padding_in_avg_computation,
                                    delta_buffer_index,
                                    out_buffer_index](CPURuntimeContext* ctx,
                                                      CPUExecutionContext* /* ectx */) {
                        kernel(ctx->buffer_data[delta_buffer_index],
                               ctx->buffer_data[out_buffer_index],
                               delta_shape,
                               out_shape,
                               window_shape,
                               window_movement_strides,
                               padding_below,
                               padding_above,
                               include_padding_in_avg_computation);
                    };
                    functors.emplace_back(functor);
                }
            }

            void register_builders_avg_pool_cpp()
            {
                REGISTER_OP_BUILDER(ngraph::op::v0::AvgPool);
                REGISTER_OP_BUILDER(ngraph::op::v0::AvgPoolBackprop);
            }
        }
    }
}
