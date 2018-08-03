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
                auto& tensor_data = external_function->get_tensor_data();

                auto arg0_shape = args[0].get_shape();
                auto out_shape = out[0].get_shape();

                auto& arg0_tensor = tensor_data[args[0].get_name()];
                auto& out_tensor = tensor_data[out[0].get_name()];

                auto window_shape = avg_pool->get_window_shape();
                auto window_movement_strides = avg_pool->get_window_movement_strides();
                auto padding_below = avg_pool->get_padding_below();
                auto padding_above = avg_pool->get_padding_above();
                auto include_padding_in_avg_computation =
                    avg_pool->get_include_padding_in_avg_computation();

                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto input_desc = mkldnn_emitter->build_memory_descriptor(
                        args[0], runtime::cpu::mkldnn_utils::get_input_mkldnn_format(node, 0));
                    auto result_desc = mkldnn_emitter->build_memory_descriptor(
                        out[0], runtime::cpu::mkldnn_utils::get_output_mkldnn_format(node, 0));

                    size_t avg_pool_index = mkldnn_emitter->build_pooling_forward(
                        (include_padding_in_avg_computation
                             ? mkldnn::algorithm::pooling_avg_include_padding
                             : mkldnn::algorithm::pooling_avg_exclude_padding),
                        input_desc,
                        result_desc,
                        window_movement_strides,
                        window_shape,
                        padding_below,
                        padding_above);

                    auto& deps = mkldnn_emitter->get_primitive_deps(avg_pool_index);

                    auto functor = [&, avg_pool_index](CPURuntimeContext* ctx) {
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
                                    include_padding_in_avg_computation](CPURuntimeContext* ctx) {
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
                auto& tensor_data = external_function->get_tensor_data();

                auto delta_shape = args[0].get_shape();
                auto out_shape = out[0].get_shape();

                auto& delta_tensor = tensor_data[args[0].get_name()];
                auto& out_tensor = tensor_data[out[0].get_name()];

                auto window_shape = apb->get_window_shape();
                auto window_movement_strides = apb->get_window_movement_strides();
                auto padding_below = apb->get_padding_below();
                auto padding_above = apb->get_padding_above();
                auto include_padding_in_avg_computation =
                    apb->get_include_padding_in_avg_computation();

                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto diff_dst_desc = mkldnn_emitter->build_memory_descriptor(
                        args[0], runtime::cpu::mkldnn_utils::get_input_mkldnn_format(node, 0));
                    auto diff_src_desc = mkldnn_emitter->build_memory_descriptor(
                        out[0], runtime::cpu::mkldnn_utils::get_output_mkldnn_format(node, 0));

                    size_t avg_pool_index = mkldnn_emitter->build_pooling_backward(
                        (apb->get_include_padding_in_avg_computation()
                             ? mkldnn::algorithm::pooling_avg_include_padding
                             : mkldnn::algorithm::pooling_avg_exclude_padding),
                        diff_dst_desc,
                        diff_src_desc,
                        apb->get_window_movement_strides(),
                        apb->get_window_shape(),
                        apb->get_padding_below(),
                        apb->get_padding_above());

                    auto& deps = mkldnn_emitter->get_primitive_deps(avg_pool_index);
                    auto functor = [&, avg_pool_index](CPURuntimeContext* ctx) {
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
                                    include_padding_in_avg_computation](CPURuntimeContext* ctx) {
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
