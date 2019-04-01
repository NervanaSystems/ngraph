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

#include "ngraph/runtime/cpu/kernel/avg_pool.hpp"
#include "ngraph/op/avg_pool.hpp"
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
            void Builder::BUILDER_DECL(ngraph::op::AvgPool)
            {
                auto avg_pool = static_cast<const ngraph::op::AvgPool*>(node);

                auto& functors = external_function->get_functors();

                auto arg0_shape = args[0].get_shape();
                auto out_shape = out[0].get_shape();

                auto& arg0_tensor = external_function->get_tensor_data(args[0].get_name());
                auto& out_tensor = external_function->get_tensor_data(out[0].get_name());

                auto window_shape = avg_pool->get_window_shape();
                auto window_movement_strides = avg_pool->get_window_movement_strides();
                auto padding_below = avg_pool->get_padding_below();
                auto padding_above = avg_pool->get_padding_above();
                auto include_padding_in_avg_computation =
                    avg_pool->get_include_padding_in_avg_computation();

                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto avg_pool_desc =
                        mkldnn_emitter->get_avg_pooling_forward_desc<ngraph::op::AvgPool>(node,
                                                                                          false);
                    // AvgPool needs 3 primitives: input, result, and pooling_forward.
                    size_t avg_pool_index = mkldnn_emitter->reserve_primitive_space(3);
                    auto& deps = mkldnn_emitter->get_primitive_deps(avg_pool_index);

                    auto functor = [&, avg_pool_desc, avg_pool_index](CPURuntimeContext* ctx,
                                                                      CPUExecutionContext* ectx) {
                        if (ctx->first_iteration)
                        {
                            mkldnn_emitter->build_pooling_forward(
                                ctx->mkldnn_primitives, avg_pool_desc, deps, avg_pool_index);
                        }
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[0], arg0_tensor);
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[1], out_tensor);
                        cpu::mkldnn_utils::mkldnn_invoke_primitive(ctx, avg_pool_index);
                    };
                    functors.emplace_back(functor);
                }
                else
                {
                    std::function<decltype(runtime::cpu::kernel::avg_pool<float>)> kernel;

                    SELECT_KERNEL(
                        kernel, out[0].get_element_type(), runtime::cpu::kernel::avg_pool);

                    auto functor = [&,
                                    kernel,
                                    arg0_shape,
                                    out_shape,
                                    window_shape,
                                    window_movement_strides,
                                    padding_below,
                                    padding_above,
                                    include_padding_in_avg_computation](CPURuntimeContext* ctx,
                                                                        CPUExecutionContext* ectx) {
                        kernel(arg0_tensor,
                               out_tensor,
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
            void Builder::BUILDER_DECL(ngraph::op::AvgPoolBackprop)
            {
                auto apb = static_cast<const ngraph::op::AvgPoolBackprop*>(node);

                auto& functors = external_function->get_functors();

                auto delta_shape = args[0].get_shape();
                auto out_shape = out[0].get_shape();

                auto& delta_tensor = external_function->get_tensor_data(args[0].get_name());
                auto& out_tensor = external_function->get_tensor_data(out[0].get_name());

                auto window_shape = apb->get_window_shape();
                auto window_movement_strides = apb->get_window_movement_strides();
                auto padding_below = apb->get_padding_below();
                auto padding_above = apb->get_padding_above();
                auto include_padding_in_avg_computation =
                    apb->get_include_padding_in_avg_computation();

                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto avg_pool_fwd_desc =
                        mkldnn_emitter->get_avg_pooling_forward_desc<ngraph::op::AvgPoolBackprop>(
                            node, true);
                    auto avg_pool_desc =
                        mkldnn_emitter->get_avg_pooling_backward_desc<ngraph::op::AvgPoolBackprop>(
                            node);
                    // AvgPoolBackprop needs 3 primitives: input, result, and pooling_backward.
                    size_t avg_pool_index = mkldnn_emitter->reserve_primitive_space(3);
                    auto& deps = mkldnn_emitter->get_primitive_deps(avg_pool_index);

                    auto functor = [&, avg_pool_desc, avg_pool_fwd_desc, avg_pool_index](
                        CPURuntimeContext* ctx, CPUExecutionContext* ectx) {
                        if (ctx->first_iteration)
                        {
                            mkldnn_emitter->build_pooling_backward(ctx->mkldnn_primitives,
                                                                   avg_pool_desc,
                                                                   avg_pool_fwd_desc,
                                                                   deps,
                                                                   avg_pool_index);
                        }
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[0], delta_tensor);
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[1], out_tensor);
                        cpu::mkldnn_utils::mkldnn_invoke_primitive(ctx, avg_pool_index);
                    };
                    functors.emplace_back(functor);
                }
                else
                {
                    std::function<decltype(runtime::cpu::kernel::avg_pool_backprop<float>)> kernel;
                    SELECT_KERNEL(
                        kernel, out[0].get_element_type(), runtime::cpu::kernel::avg_pool_backprop);

                    auto functor = [&,
                                    kernel,
                                    delta_shape,
                                    out_shape,
                                    window_shape,
                                    window_movement_strides,
                                    padding_below,
                                    padding_above,
                                    include_padding_in_avg_computation](CPURuntimeContext* ctx,
                                                                        CPUExecutionContext* ectx) {
                        kernel(delta_tensor,
                               out_tensor,
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
            REGISTER_OP_BUILDER(AvgPool);
            REGISTER_OP_BUILDER(AvgPoolBackprop);
        }
    }
}
