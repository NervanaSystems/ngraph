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

#include "ngraph/runtime/cpu/kernel/max_pool.hpp"
#include "ngraph/op/max_pool.hpp"
#include "ngraph/runtime/cpu/cpu_builder.hpp"
#include "ngraph/runtime/cpu/mkldnn_invoke.hpp"
#include "ngraph/runtime/cpu/mkldnn_utils.hpp"
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
            void Builder::BUILDER_DECL(ngraph::op::MaxPool)
            {
                auto max_pool = static_cast<const ngraph::op::MaxPool*>(node);

                auto& functors = external_function->get_functors();

                auto arg0_shape = args[0].get_shape();
                auto out_shape = out[0].get_shape();

                auto arg0_buffer_index = external_function->get_buffer_index(args[0].get_name());
                auto out_buffer_index = external_function->get_buffer_index(out[0].get_name());

                auto window_shape = max_pool->get_window_shape();
                auto window_movement_strides = max_pool->get_window_movement_strides();
                auto padding_below = max_pool->get_padding_below();
                auto padding_above = max_pool->get_padding_above();

                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto max_pool_desc =
                        mkldnn_emitter->get_max_pooling_forward_desc<ngraph::op::MaxPool>(node,
                                                                                          false);
                    // MaxPool needs 3 primitives: input, result, and pooling_forward.
                    size_t max_pool_index = mkldnn_emitter->reserve_primitive_space(3);
                    auto& deps = mkldnn_emitter->get_primitive_deps(max_pool_index);

                    auto functor =
                        [&, max_pool_desc, max_pool_index, arg0_buffer_index, out_buffer_index](
                            CPURuntimeContext* ctx, CPUExecutionContext* ectx) {
                            if (ctx->first_iteration)
                            {
                                mkldnn_emitter->build_pooling_forward(
                                    ctx->mkldnn_primitives, max_pool_desc, deps, max_pool_index);
                            }
                            cpu::mkldnn_utils::set_memory_ptr(
                                ctx, deps[0], ctx->buffer_data[arg0_buffer_index]);
                            cpu::mkldnn_utils::set_memory_ptr(
                                ctx, deps[1], ctx->buffer_data[out_buffer_index]);
                            cpu::mkldnn_utils::mkldnn_invoke_primitive(ctx, max_pool_index);
                        };
                    functors.emplace_back(functor);
                }
                else
                {
                    std::function<decltype(runtime::cpu::kernel::max_pool<float>)> kernel;

                    SELECT_KERNEL(
                        kernel, out[0].get_element_type(), runtime::cpu::kernel::max_pool);

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
                                                      CPUExecutionContext* ectx) {
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
            void Builder::BUILDER_DECL(ngraph::op::MaxPoolBackprop)
            {
                auto mpb = static_cast<const ngraph::op::MaxPoolBackprop*>(node);

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

                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto fwd_pool_desc =
                        mkldnn_emitter->get_max_pooling_forward_desc<ngraph::op::MaxPoolBackprop>(
                            node, true);
                    auto bwd_pool_desc =
                        mkldnn_emitter->get_max_pooling_backward_desc<ngraph::op::MaxPoolBackprop>(
                            node);
                    auto fprop_src_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);

                    // MaxPoolBackprop forward needs 4 primitives: fprop_src, diff_src, workspace,
                    // and pooling_forward.
                    // It needs a new workspace.
                    size_t fwd_pool_index =
                        mkldnn_emitter->reserve_primitive_space(4, true /* new workspace */);
                    auto& fdeps = mkldnn_emitter->get_primitive_deps(fwd_pool_index);

                    auto functor_fprop =
                        [&, fwd_pool_index, arg_fwd_buffer_index, out_buffer_index](
                            CPURuntimeContext* ctx, CPUExecutionContext* ectx) {
                            cpu::mkldnn_utils::set_memory_ptr(
                                ctx, fdeps[0], ctx->buffer_data[arg_fwd_buffer_index]);
                            cpu::mkldnn_utils::set_memory_ptr(
                                ctx, fdeps[1], ctx->buffer_data[out_buffer_index]);
                            cpu::mkldnn_utils::set_memory_ptr(
                                ctx, fdeps[2], ctx->mkldnn_workspaces[fdeps[3]]);
                            cpu::mkldnn_utils::mkldnn_invoke_primitive(ctx, fwd_pool_index);
                        };

                    // MaxPoolBackprop backward needs 4 primitives: diff_dst, workspace, diff_src,
                    // and pooling_backward.
                    // It needs a new workspace.
                    size_t bwd_pool_index =
                        mkldnn_emitter->reserve_primitive_space(4, true /* new workspace */);
                    auto& bdeps = mkldnn_emitter->get_primitive_deps(bwd_pool_index);
                    auto functor_bprop = [&, bwd_pool_index, delta_buffer_index, out_buffer_index](
                        CPURuntimeContext* ctx, CPUExecutionContext* ectx) {
                        cpu::mkldnn_utils::set_memory_ptr(
                            ctx, bdeps[0], ctx->buffer_data[delta_buffer_index]);
                        cpu::mkldnn_utils::set_memory_ptr(
                            ctx, bdeps[1], ctx->mkldnn_workspaces[bdeps[3]]);
                        cpu::mkldnn_utils::set_memory_ptr(
                            ctx, bdeps[2], ctx->buffer_data[out_buffer_index]);
                        cpu::mkldnn_utils::mkldnn_invoke_primitive(ctx, bwd_pool_index);
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
                            mkldnn_emitter->build_max_pooling_backward(ctx->mkldnn_primitives,
                                                                       ctx->mkldnn_workspaces,
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
                        kernel, out[0].get_element_type(), runtime::cpu::kernel::max_pool_backprop);

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
                                                      CPUExecutionContext* ectx) {
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
                if (!runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    throw ngraph_error("MaxPoolWithIndices isn't supported");
                }

                auto& functors = external_function->get_functors();

                auto arg0_buffer_index = external_function->get_buffer_index(args[0].get_name());
                auto out0_buffer_index = external_function->get_buffer_index(out[0].get_name());
                auto out1_buffer_index = external_function->get_buffer_index(out[1].get_name());

                auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                auto max_pool_desc =
                    mkldnn_emitter
                        ->get_max_pooling_with_indices_forward_desc<ngraph::op::MaxPoolWithIndices>(
                            node);

                // MaxPoolWithIndices needs 4 primitives: src, dst, workspace, and pooling_forward.
                size_t max_pool_index = mkldnn_emitter->reserve_primitive_space(4);
                auto& deps = mkldnn_emitter->get_primitive_deps(max_pool_index);

                auto functor = [&,
                                max_pool_desc,
                                max_pool_index,
                                arg0_buffer_index,
                                out0_buffer_index,
                                out1_buffer_index](CPURuntimeContext* ctx,
                                                   CPUExecutionContext* ectx) {
                    if (ctx->first_iteration)
                    {
                        mkldnn_emitter->build_max_pooling_with_indices_forward(
                            ctx->mkldnn_primitives, max_pool_desc, deps, max_pool_index);
                    }
                    cpu::mkldnn_utils::set_memory_ptr(
                        ctx, deps[0], ctx->buffer_data[arg0_buffer_index]);
                    cpu::mkldnn_utils::set_memory_ptr(
                        ctx, deps[1], ctx->buffer_data[out0_buffer_index]);
                    cpu::mkldnn_utils::set_memory_ptr(
                        ctx, deps[2], ctx->buffer_data[out1_buffer_index]);
                    cpu::mkldnn_utils::mkldnn_invoke_primitive(ctx, max_pool_index);
                };
                functors.emplace_back(functor);
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::MaxPoolWithIndicesBackprop)
            {
                if (!runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    throw ngraph_error("MaxPoolWithIndicesBackprop isn't supported");
                }

                auto& functors = external_function->get_functors();

                auto arg1_buffer_index = external_function->get_buffer_index(args[1].get_name());
                auto arg2_buffer_index = external_function->get_buffer_index(args[2].get_name());
                auto out_buffer_index = external_function->get_buffer_index(out[0].get_name());

                auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                auto fwd_pool_desc =
                    mkldnn_emitter
                        ->get_max_pooling_forward_desc<ngraph::op::MaxPoolWithIndicesBackprop>(
                            node, true);
                auto bwd_pool_desc =
                    mkldnn_emitter
                        ->get_max_pooling_backward_desc<ngraph::op::MaxPoolWithIndicesBackprop>(
                            node);
                // MaxPoolWithIndicesBackprop needs 4 primitives: diff_dst, fprop_workspace,
                // diff_src, and pooling_backward.
                size_t max_pool_index = mkldnn_emitter->reserve_primitive_space(4);
                auto& deps = mkldnn_emitter->get_primitive_deps(max_pool_index);

                auto functor = [&,
                                bwd_pool_desc,
                                fwd_pool_desc,
                                max_pool_index,
                                arg1_buffer_index,
                                arg2_buffer_index,
                                out_buffer_index](CPURuntimeContext* ctx,
                                                  CPUExecutionContext* ectx) {
                    if (ctx->first_iteration)
                    {
                        mkldnn_emitter->build_max_pooling_with_indices_backward(
                            ctx->mkldnn_primitives,
                            bwd_pool_desc,
                            fwd_pool_desc,
                            deps,
                            max_pool_index);
                    }
                    cpu::mkldnn_utils::set_memory_ptr(
                        ctx, deps[0], ctx->buffer_data[arg1_buffer_index]);
                    cpu::mkldnn_utils::set_memory_ptr(
                        ctx, deps[1], ctx->buffer_data[arg2_buffer_index]);
                    cpu::mkldnn_utils::set_memory_ptr(
                        ctx, deps[2], ctx->buffer_data[out_buffer_index]);
                    cpu::mkldnn_utils::mkldnn_invoke_primitive(ctx, max_pool_index);
                };
                functors.emplace_back(functor);
            }

            void register_builders_max_pool_cpp()
            {
                REGISTER_OP_BUILDER(MaxPool);
                REGISTER_OP_BUILDER(MaxPoolBackprop);
                REGISTER_OP_BUILDER(MaxPoolWithIndices);
                REGISTER_OP_BUILDER(MaxPoolWithIndicesBackprop);
            }
        }
    }
}
