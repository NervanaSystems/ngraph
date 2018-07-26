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

#include "ngraph/runtime/cpu/kernel/max_pool.hpp"
#include "ngraph/op/max_pool.hpp"
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
            void Builder::BUILDER_DECL(ngraph::op::MaxPool)
            {
                auto max_pool = static_cast<const ngraph::op::MaxPool*>(node);

                auto& functors = external_function->get_functors();
                auto& tensor_data = external_function->get_tensor_data();

                auto arg0_shape = args[0].get_shape();
                auto out_shape = out[0].get_shape();

                auto& arg0_tensor = tensor_data[args[0].get_name()];
                auto& out_tensor = tensor_data[out[0].get_name()];

                auto window_shape = max_pool->get_window_shape();
                auto window_movement_strides = max_pool->get_window_movement_strides();
                auto padding_below = max_pool->get_padding_below();
                auto padding_above = max_pool->get_padding_above();

                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto input_desc = mkldnn_emitter->build_memory_descriptor(
                        args[0], runtime::cpu::mkldnn_utils::get_input_mkldnn_format(node, 0));
                    auto result_desc = mkldnn_emitter->build_memory_descriptor(
                        out[0], runtime::cpu::mkldnn_utils::get_output_mkldnn_format(node, 0));

                    size_t max_pool_index =
                        mkldnn_emitter->build_pooling_forward(mkldnn::algorithm::pooling_max,
                                                              input_desc,
                                                              result_desc,
                                                              window_movement_strides,
                                                              window_shape,
                                                              padding_below,
                                                              padding_above);

                    auto& deps = mkldnn_emitter->get_primitive_deps(max_pool_index);

                    auto functor = [&, max_pool_index](CPURuntimeContext* ctx) {
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
                                    padding_above](CPURuntimeContext* ctx) {
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

            REGISTER_OP_BUILDER(MaxPool);
        }
    }
}
