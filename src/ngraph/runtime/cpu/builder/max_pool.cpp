//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
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

                auto& arg0_tensor = external_function->get_tensor_data(args[0].get_name());
                auto& out_tensor = external_function->get_tensor_data(out[0].get_name());

                auto window_shape = max_pool->get_window_shape();
                auto window_movement_strides = max_pool->get_window_movement_strides();
                auto padding_below = max_pool->get_padding_below();
                auto padding_above = max_pool->get_padding_above();

                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);

                    size_t max_pool_index =
                        mkldnn_emitter->build_pooling_forward(mkldnn::algorithm::pooling_max,
                                                              input_desc,
                                                              result_desc,
                                                              window_movement_strides,
                                                              window_shape,
                                                              padding_below,
                                                              padding_above);

                    auto& deps = mkldnn_emitter->get_primitive_deps(max_pool_index);

                    auto functor = [&, max_pool_index](CPURuntimeContext* ctx,
                                                       CPUExecutionContext* ectx) {
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[0], arg0_tensor);
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[1], out_tensor);
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
                                    padding_above](CPURuntimeContext* ctx,
                                                   CPUExecutionContext* ectx) {
                        kernel(arg0_tensor,
                               out_tensor,
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

                auto& arg_fwd_tensor = external_function->get_tensor_data(args[0].get_name());
                auto& delta_tensor = external_function->get_tensor_data(args[1].get_name());
                auto& out_tensor = external_function->get_tensor_data(out[0].get_name());

                auto window_shape = mpb->get_window_shape();
                auto window_movement_strides = mpb->get_window_movement_strides();
                auto padding_below = mpb->get_padding_below();
                auto padding_above = mpb->get_padding_above();

                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto fprop_src_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto diff_dst_desc = mkldnn_utils::get_input_mkldnn_md(node, 1);
                    auto diff_src_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);

                    size_t max_pool_index = mkldnn_emitter->build_max_pooling_backward(
                        mkldnn::algorithm::pooling_max,
                        fprop_src_desc,
                        diff_dst_desc,
                        diff_src_desc,
                        mpb->get_window_movement_strides(),
                        mpb->get_window_shape(),
                        mpb->get_padding_below(),
                        mpb->get_padding_above());

                    auto& fdeps = mkldnn_emitter->get_primitive_deps(max_pool_index - 1);
                    auto functor_fprop = [&, max_pool_index](CPURuntimeContext* ctx,
                                                             CPUExecutionContext* ectx) {
                        cpu::mkldnn_utils::set_memory_ptr(ctx, fdeps[0], arg_fwd_tensor);
                        cpu::mkldnn_utils::set_memory_ptr(ctx, fdeps[1], out_tensor);
                        cpu::mkldnn_utils::set_memory_ptr(
                            ctx, fdeps[2], ctx->mkldnn_workspaces[fdeps[3]]);
                        cpu::mkldnn_utils::mkldnn_invoke_primitive(ctx, max_pool_index - 1);
                    };
                    auto& bdeps = mkldnn_emitter->get_primitive_deps(max_pool_index);
                    auto functor_bprop = [&, max_pool_index](CPURuntimeContext* ctx,
                                                             CPUExecutionContext* ectx) {
                        cpu::mkldnn_utils::set_memory_ptr(ctx, bdeps[0], delta_tensor);
                        cpu::mkldnn_utils::set_memory_ptr(
                            ctx, bdeps[1], ctx->mkldnn_workspaces[bdeps[3]]);
                        cpu::mkldnn_utils::set_memory_ptr(ctx, bdeps[2], out_tensor);
                        cpu::mkldnn_utils::mkldnn_invoke_primitive(ctx, max_pool_index);
                    };
                    auto functor = [&, functor_fprop, functor_bprop](CPURuntimeContext* ctx,
                                                                     CPUExecutionContext* ectx) {
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
                                    padding_above](CPURuntimeContext* ctx,
                                                   CPUExecutionContext* ectx) {
                        kernel(arg_fwd_tensor,
                               delta_tensor,
                               out_tensor,
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

                auto max_pool = static_cast<const ngraph::op::MaxPoolWithIndices*>(node);

                auto& functors = external_function->get_functors();

                auto& arg0_tensor = external_function->get_tensor_data(args[0].get_name());
                auto& out0_tensor = external_function->get_tensor_data(out[0].get_name());
                auto& out1_tensor = external_function->get_tensor_data(out[1].get_name());

                auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                auto input_desc = runtime::cpu::mkldnn_utils::get_input_mkldnn_md(node, 0);
                auto result_desc = runtime::cpu::mkldnn_utils::get_output_mkldnn_md(node, 0);

                size_t max_pool_index = mkldnn_emitter->build_max_pooling_with_indices_forward(
                    mkldnn::algorithm::pooling_max,
                    input_desc,
                    result_desc,
                    max_pool->get_window_movement_strides(),
                    max_pool->get_window_shape(),
                    max_pool->get_padding_below(),
                    max_pool->get_padding_above());

                auto& deps = mkldnn_emitter->get_primitive_deps(max_pool_index);

                auto functor = [&, max_pool_index](CPURuntimeContext* ctx,
                                                   CPUExecutionContext* ectx) {
                    cpu::mkldnn_utils::set_memory_ptr(ctx, deps[0], arg0_tensor);
                    cpu::mkldnn_utils::set_memory_ptr(ctx, deps[1], out0_tensor);
                    cpu::mkldnn_utils::set_memory_ptr(ctx, deps[2], out1_tensor);
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

                auto& arg1_tensor = external_function->get_tensor_data(args[1].get_name());
                auto& arg2_tensor = external_function->get_tensor_data(args[2].get_name());
                auto& out_tensor = external_function->get_tensor_data(out[0].get_name());

                auto mpb = static_cast<const ngraph::op::MaxPoolWithIndicesBackprop*>(node);

                auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                auto diff_dst_desc = runtime::cpu::mkldnn_utils::get_input_mkldnn_md(node, 1);
                auto diff_src_desc = runtime::cpu::mkldnn_utils::get_output_mkldnn_md(node, 0);

                size_t max_pool_index = mkldnn_emitter->build_max_pooling_with_indices_backward(
                    mkldnn::algorithm::pooling_max,
                    diff_dst_desc,
                    diff_src_desc,
                    mpb->get_window_movement_strides(),
                    mpb->get_window_shape(),
                    mpb->get_padding_below(),
                    mpb->get_padding_above());

                auto& deps = mkldnn_emitter->get_primitive_deps(max_pool_index);

                auto functor = [&, max_pool_index](CPURuntimeContext* ctx,
                                                   CPUExecutionContext* ectx) {
                    cpu::mkldnn_utils::set_memory_ptr(ctx, deps[0], arg1_tensor);
                    cpu::mkldnn_utils::set_memory_ptr(ctx, deps[1], arg2_tensor);
                    cpu::mkldnn_utils::set_memory_ptr(ctx, deps[2], out_tensor);
                    cpu::mkldnn_utils::mkldnn_invoke_primitive(ctx, max_pool_index);
                };
                functors.emplace_back(functor);
            }

            REGISTER_OP_BUILDER(MaxPool);
            REGISTER_OP_BUILDER(MaxPoolBackprop);
            REGISTER_OP_BUILDER(MaxPoolWithIndices);
            REGISTER_OP_BUILDER(MaxPoolWithIndicesBackprop);
        }
    }
}
