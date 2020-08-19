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

#include "ngraph/runtime/cpu/kernel/max_pool.hpp"
#include "ngraph/op/max_pool.hpp"
#include "ngraph/runtime/cpu/cpu_builder.hpp"
#include "ngraph/runtime/cpu/dnnl_invoke.hpp"
#include "ngraph/runtime/cpu/dnnl_utils.hpp"
#include "ngraph/runtime/cpu/op/max_pool_with_indices.hpp"

using namespace std;
using namespace ngraph;

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            template <>
            void Builder::BUILDER_DECL(ngraph::op::v0::MaxPool)
            {
                auto max_pool = static_cast<const ngraph::op::v0::MaxPool*>(node);

                auto& functors = external_function->get_functors();

                auto arg0_shape = args[0].get_shape();
                auto out_shape = out[0].get_shape();

                auto arg0_buffer_index = external_function->get_buffer_index(args[0].get_name());
                auto out_buffer_index = external_function->get_buffer_index(out[0].get_name());

                auto window_shape = max_pool->get_window_shape();
                auto window_movement_strides = max_pool->get_window_movement_strides();
                auto padding_below = max_pool->get_padding_below();
                auto padding_above = max_pool->get_padding_above();

                if (runtime::cpu::dnnl_utils::use_dnnl_kernel(node))
                {
                    auto& dnnl_emitter = external_function->get_dnnl_emitter();
                    auto max_pool_desc =
                        dnnl_emitter->get_max_pooling_forward_desc<ngraph::op::v0::MaxPool>(node,
                                                                                            false);
                    size_t scratchpad_size = QUERY_SCRATCHPAD(pooling_forward, max_pool_desc);

                    // MaxPool needs 3 primitives: input, result, and pooling_forward.
                    size_t max_pool_index = dnnl_emitter->reserve_primitive_space(3);
                    auto& deps = dnnl_emitter->get_primitive_deps(max_pool_index);

                    auto functor = [&,
                                    max_pool_desc,
                                    max_pool_index,
                                    scratchpad_size,
                                    arg0_buffer_index,
                                    out_buffer_index](CPURuntimeContext* ctx,
                                                      CPUExecutionContext* /* ectx */) {
                        if (ctx->first_iteration)
                        {
                            dnnl_emitter->build_pooling_forward(ctx->dnnl_memories,
                                                                ctx->dnnl_primitives,
                                                                ctx->dnnl_scratchpad_mds,
                                                                max_pool_desc,
                                                                deps,
                                                                max_pool_index);
                        }
                        cpu::dnnl_utils::set_memory_ptr(
                            ctx, deps[0], ctx->buffer_data[arg0_buffer_index]);
                        cpu::dnnl_utils::set_memory_ptr(
                            ctx, deps[1], ctx->buffer_data[out_buffer_index]);

                        cpu::dnnl_utils::dnnl_invoke_primitive(ctx,
                                                               max_pool_index,
                                                               deps,
                                                               cpu::dnnl_utils::OpType::MAXPOOL,
                                                               scratchpad_size);
                    };
                    functors.emplace_back(functor);
                }
                else
                {
                    std::function<decltype(runtime::cpu::kernel::max_pool<float>)> kernel;

                    SELECT_KERNEL(kernel, out[0].get_element_type(), runtime::cpu::kernel::max_pool)

                    auto functor = [&,
                                    kernel,
                                    arg0_shape,
                                    out_shape,
                                    window_shape,
                                    window_movement_strides,
                                    padding_below,
                                    padding_above,
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
                               padding_above);
                    };
                    functors.emplace_back(functor);
                }
            }
            template <>
            void Builder::BUILDER_DECL(ngraph::op::v0::MaxPoolBackprop)
            {
                auto mpb = static_cast<const ngraph::op::v0::MaxPoolBackprop*>(node);

                auto& functors = external_function->get_functors();

                auto arg_fwd_shape = args[0].get_shape();
                auto delta_shape = args[1].get_shape();
                auto out_shape = out[0].get_shape();

                auto arg_fwd_buffer_index = external_function->get_buffer_index(args[0].get_name());
                auto delta_buffer_index = external_function->get_buffer_index(args[1].get_name());
                auto out_buffer_index = external_function->get_buffer_index(out[0].get_name());

                auto window_shape = mpb->get_window_shape();
                auto window_movement_strides = mpb->get_window_movement_strides();
                auto padding_below = mpb->get_padding_below();
                auto padding_above = mpb->get_padding_above();

                if (runtime::cpu::dnnl_utils::use_dnnl_kernel(node))
                {
                    auto& dnnl_emitter = external_function->get_dnnl_emitter();
                    auto fwd_pool_desc =
                        dnnl_emitter->get_max_pooling_forward_desc<ngraph::op::v0::MaxPoolBackprop>(
                            node, true);
                    auto bwd_pool_desc =
                        dnnl_emitter
                            ->get_max_pooling_backward_desc<ngraph::op::v0::MaxPoolBackprop>(node);
                    auto fprop_src_desc = dnnl_utils::get_input_dnnl_md(node, 0);
                    size_t scratchpad_size =
                        QUERY_SCRATCHPAD_2ARGS(max_pooling_backward, fwd_pool_desc, bwd_pool_desc);

                    // MaxPoolBackprop forward needs 4 primitives: fprop_src, diff_src, workspace,
                    // and pooling_forward.
                    // It needs a new workspace.
                    size_t fwd_pool_index = dnnl_emitter->reserve_primitive_space(
                        4, false /* fwd and bwd */, true /* new workspace */);
                    auto& fdeps = dnnl_emitter->get_primitive_deps(fwd_pool_index);

                    auto functor_fprop = [&,
                                          fwd_pool_index,
                                          arg_fwd_buffer_index,
                                          scratchpad_size,
                                          out_buffer_index](CPURuntimeContext* ctx,
                                                            CPUExecutionContext* /* ectx */) {
                        cpu::dnnl_utils::set_memory_ptr(
                            ctx, fdeps[0], ctx->buffer_data[arg_fwd_buffer_index]);
                        cpu::dnnl_utils::set_memory_ptr(
                            ctx, fdeps[1], ctx->buffer_data[out_buffer_index]);
                        cpu::dnnl_utils::set_memory_ptr(
                            ctx, fdeps[2], ctx->dnnl_workspaces[fdeps[3]]);
                        cpu::dnnl_utils::dnnl_invoke_primitive(
                            ctx,
                            fwd_pool_index,
                            fdeps,
                            cpu::dnnl_utils::OpType::MAXPOOLBACKPROPFORWARD,
                            scratchpad_size);
                    };

                    // MaxPoolBackprop backward needs 4 primitives: diff_dst, workspace, diff_src,
                    // and pooling_backward.
                    // It needs a new workspace.
                    size_t bwd_pool_index = dnnl_emitter->reserve_primitive_space(
                        4, false /* fwd and bwd */, true /* new workspace */);
                    auto& bdeps = dnnl_emitter->get_primitive_deps(bwd_pool_index);
                    auto functor_bprop = [&, bwd_pool_index, delta_buffer_index, out_buffer_index](
                                             CPURuntimeContext* ctx,
                                             CPUExecutionContext* /* ectx */) {
                        cpu::dnnl_utils::set_memory_ptr(
                            ctx, bdeps[0], ctx->buffer_data[delta_buffer_index]);
                        cpu::dnnl_utils::set_memory_ptr(
                            ctx, bdeps[1], ctx->dnnl_workspaces[bdeps[3]]);
                        cpu::dnnl_utils::set_memory_ptr(
                            ctx, bdeps[2], ctx->buffer_data[out_buffer_index]);
                        cpu::dnnl_utils::dnnl_invoke_primitive(
                            ctx,
                            bwd_pool_index,
                            bdeps,
                            cpu::dnnl_utils::OpType::MAXPOOLBACKPROPBACKWARD);
                    };
                    auto functor = [&,
                                    bwd_pool_desc,
                                    fwd_pool_desc,
                                    fprop_src_desc,
                                    fwd_pool_index,
                                    bwd_pool_index,
                                    functor_fprop,
                                    functor_bprop](CPURuntimeContext* ctx,
                                                   CPUExecutionContext* ectx) {
                        if (ctx->first_iteration)
                        {
                            dnnl_emitter->build_max_pooling_backward(ctx->dnnl_memories,
                                                                     ctx->dnnl_primitives,
                                                                     ctx->dnnl_scratchpad_mds,
                                                                     ctx->dnnl_workspaces,
                                                                     bwd_pool_desc,
                                                                     fwd_pool_desc,
                                                                     fprop_src_desc,
                                                                     fdeps,
                                                                     bdeps,
                                                                     fwd_pool_index,
                                                                     bwd_pool_index);
                        }
                        functor_fprop(ctx, ectx);
                        functor_bprop(ctx, ectx);
                    };
                    functors.emplace_back(functor);
                }
                else
                {
                    std::function<decltype(runtime::cpu::kernel::max_pool_backprop<float>)> kernel;

                    SELECT_KERNEL(
                        kernel, out[0].get_element_type(), runtime::cpu::kernel::max_pool_backprop)

                    auto functor = [&,
                                    kernel,
                                    arg_fwd_shape,
                                    delta_shape,
                                    out_shape,
                                    window_shape,
                                    window_movement_strides,
                                    padding_below,
                                    padding_above,
                                    arg_fwd_buffer_index,
                                    delta_buffer_index,
                                    out_buffer_index](CPURuntimeContext* ctx,
                                                      CPUExecutionContext* /* ectx */) {
                        kernel(ctx->buffer_data[arg_fwd_buffer_index],
                               ctx->buffer_data[delta_buffer_index],
                               ctx->buffer_data[out_buffer_index],
                               delta_shape,
                               arg_fwd_shape,
                               window_shape,
                               window_movement_strides,
                               padding_below,
                               padding_above);
                    };
                    functors.emplace_back(functor);
                }
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::MaxPoolWithIndices)
            {
                if (!runtime::cpu::dnnl_utils::use_dnnl_kernel(node))
                {
                    throw ngraph_error("MaxPoolWithIndices isn't supported");
                }

                auto& functors = external_function->get_functors();

                auto arg0_buffer_index = external_function->get_buffer_index(args[0].get_name());
                auto out0_buffer_index = external_function->get_buffer_index(out[0].get_name());
                auto out1_buffer_index = external_function->get_buffer_index(out[1].get_name());

                auto& dnnl_emitter = external_function->get_dnnl_emitter();
                auto max_pool_desc =
                    dnnl_emitter
                        ->get_max_pooling_with_indices_forward_desc<ngraph::op::MaxPoolWithIndices>(
                            node);
                size_t scratchpad_size = QUERY_SCRATCHPAD(pooling_forward, max_pool_desc);

                // MaxPoolWithIndices needs 4 primitives: src, dst, workspace, and pooling_forward.
                size_t max_pool_index = dnnl_emitter->reserve_primitive_space(4);
                auto& deps = dnnl_emitter->get_primitive_deps(max_pool_index);

                auto functor = [&,
                                max_pool_desc,
                                max_pool_index,
                                scratchpad_size,
                                arg0_buffer_index,
                                out0_buffer_index,
                                out1_buffer_index](CPURuntimeContext* ctx,
                                                   CPUExecutionContext* /* ectx */) {
                    if (ctx->first_iteration)
                    {
                        dnnl_emitter->build_max_pooling_with_indices_forward(
                            ctx->dnnl_memories,
                            ctx->dnnl_primitives,
                            ctx->dnnl_scratchpad_mds,
                            max_pool_desc,
                            deps,
                            max_pool_index);
                    }
                    cpu::dnnl_utils::set_memory_ptr(
                        ctx, deps[0], ctx->buffer_data[arg0_buffer_index]);
                    cpu::dnnl_utils::set_memory_ptr(
                        ctx, deps[1], ctx->buffer_data[out0_buffer_index]);
                    cpu::dnnl_utils::set_memory_ptr(
                        ctx, deps[2], ctx->buffer_data[out1_buffer_index]);

                    cpu::dnnl_utils::dnnl_invoke_primitive(
                        ctx,
                        max_pool_index,
                        deps,
                        cpu::dnnl_utils::OpType::MAXPOOLWITHINDICES,
                        scratchpad_size);
                };
                functors.emplace_back(functor);
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::MaxPoolWithIndicesBackprop)
            {
                if (!runtime::cpu::dnnl_utils::use_dnnl_kernel(node))
                {
                    throw ngraph_error("MaxPoolWithIndicesBackprop isn't supported");
                }

                auto& functors = external_function->get_functors();

                auto arg1_buffer_index = external_function->get_buffer_index(args[1].get_name());
                auto arg2_buffer_index = external_function->get_buffer_index(args[2].get_name());
                auto out_buffer_index = external_function->get_buffer_index(out[0].get_name());

                auto& dnnl_emitter = external_function->get_dnnl_emitter();
                auto fwd_pool_desc =
                    dnnl_emitter
                        ->get_max_pooling_forward_desc<ngraph::op::MaxPoolWithIndicesBackprop>(
                            node, true);
                auto bwd_pool_desc =
                    dnnl_emitter
                        ->get_max_pooling_backward_desc<ngraph::op::MaxPoolWithIndicesBackprop>(
                            node);
                size_t scratchpad_size = QUERY_SCRATCHPAD_2ARGS(
                    max_pooling_with_indices_backward, fwd_pool_desc, bwd_pool_desc);

                // MaxPoolWithIndicesBackprop needs 4 primitives: diff_dst, fprop_workspace,
                // diff_src, and pooling_backward.
                size_t max_pool_index = dnnl_emitter->reserve_primitive_space(4);
                auto& deps = dnnl_emitter->get_primitive_deps(max_pool_index);

                auto functor = [&,
                                bwd_pool_desc,
                                fwd_pool_desc,
                                max_pool_index,
                                scratchpad_size,
                                arg1_buffer_index,
                                arg2_buffer_index,
                                out_buffer_index](CPURuntimeContext* ctx,
                                                  CPUExecutionContext* /* ectx */) {
                    if (ctx->first_iteration)
                    {
                        dnnl_emitter->build_max_pooling_with_indices_backward(
                            ctx->dnnl_memories,
                            ctx->dnnl_primitives,
                            ctx->dnnl_scratchpad_mds,
                            bwd_pool_desc,
                            fwd_pool_desc,
                            deps,
                            max_pool_index);
                    }
                    cpu::dnnl_utils::set_memory_ptr(
                        ctx, deps[0], ctx->buffer_data[arg1_buffer_index]);
                    cpu::dnnl_utils::set_memory_ptr(
                        ctx, deps[1], ctx->buffer_data[arg2_buffer_index]);
                    cpu::dnnl_utils::set_memory_ptr(
                        ctx, deps[2], ctx->buffer_data[out_buffer_index]);

                    cpu::dnnl_utils::dnnl_invoke_primitive(
                        ctx,
                        max_pool_index,
                        deps,
                        cpu::dnnl_utils::OpType::MAXPOOLWITHINDICESBACKPROP,
                        scratchpad_size);
                };
                functors.emplace_back(functor);
            }

            void register_builders_max_pool_cpp()
            {
                REGISTER_OP_BUILDER(ngraph::op::v0::MaxPool);
                REGISTER_OP_BUILDER(ngraph::op::v0::MaxPoolBackprop);
                REGISTER_OP_BUILDER(ngraph::op::MaxPoolWithIndices);
                REGISTER_OP_BUILDER(ngraph::op::MaxPoolWithIndicesBackprop);
            }
        }
    }
}
