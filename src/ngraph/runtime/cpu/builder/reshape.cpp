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

#include <cstring>

#include "ngraph/op/reshape.hpp"
#include "ngraph/runtime/cpu/cpu_builder.hpp"
#include "ngraph/runtime/cpu/kernel/reshape.hpp"
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
            void Builder::BUILDER_DECL(ngraph::op::Reshape)
            {
                auto& functors = external_function->get_functors();
                auto& tensor_data = external_function->get_tensor_data();

                auto& arg_tensor = tensor_data[args[0].get_name()];
                auto& out_tensor = tensor_data[out[0].get_name()];

                auto reshape = static_cast<const ngraph::op::Reshape*>(node);

                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    auto input_tvl = node->get_inputs()[0]
                                         .get_output()
                                         .get_tensor_view()
                                         ->get_tensor_view_layout();
                    auto input_cpu_tvl =
                        dynamic_pointer_cast<runtime::cpu::LayoutDescriptor>(input_tvl);

                    // Reorder input shape if needed
                    auto input_axis_order = input_cpu_tvl->get_axis_order();
                    Shape input_shape(input_axis_order.size());
                    for (size_t idx = 0; idx < input_axis_order.size(); idx++)
                    {
                        input_shape[idx] = args[0].get_shape()[input_axis_order[idx]];
                    }

                    auto output_tvl = node->get_output_tensor_view(0)->get_tensor_view_layout();
                    auto input_strides = input_tvl->get_strides();
                    auto output_strides = output_tvl->get_strides();
                    auto axis_order = reshape->get_input_order();

                    Strides new_output_strides(output_strides.size());
                    for (int i = 0; i < output_strides.size(); i++)
                        new_output_strides[axis_order[i]] = output_strides[i];

                    mkldnn::memory::data_type et = runtime::cpu::mkldnn_utils::get_mkldnn_data_type(
                        node->get_input_element_type(0));

                    mkldnn::memory::dims mkldnn_input_shape(input_shape.begin(), input_shape.end());
                    mkldnn::memory::dims mkldnn_input_strides(input_strides.begin(),
                                                              input_strides.end());
                    mkldnn::memory::dims mkldnn_output_strides(new_output_strides.begin(),
                                                               new_output_strides.end());

                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();

                    auto input_desc = mkldnn_emitter->build_blocked_memory_descriptor(
                        mkldnn_input_shape, mkldnn_input_strides, et);
                    auto result_desc = mkldnn_emitter->build_blocked_memory_descriptor(
                        mkldnn_input_shape, mkldnn_output_strides, et);

                    size_t reorder_index = mkldnn_emitter->build_reorder(input_desc, result_desc);

                    auto& deps = mkldnn_emitter->get_primitive_deps(reorder_index);
                    auto functor = [&, reorder_index](CPURuntimeContext* ctx) {
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[0], arg_tensor);
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[1], out_tensor);
                        cpu::mkldnn_utils::mkldnn_invoke_primitive(ctx, reorder_index);
                    };
                    functors.emplace_back(functor);
                }
                else
                {
                    auto arg_shape = args[0].get_shape();
                    auto arg_rank = arg_shape.size();

                    auto result_shape = out[0].get_shape();
                    auto result_rank = result_shape.size();
                    auto& result_element_type = out[0].get_element_type();

                    auto input_order = reshape->get_input_order();

                    bool same_layout = is_sorted(input_order.begin(), input_order.end());

                    auto result_size = shape_size(result_shape);

                    if (same_layout || result_size < 2)
                    {
                        size_t size = out[0].get_size() * out[0].get_element_type().size();
                        auto functor = [&, size](CPURuntimeContext* ctx) {
                            memcpy(out_tensor, arg_tensor, size);
                        };
                        functors.emplace_back(functor);
                        return;
                    }

                    std::function<decltype(runtime::cpu::kernel::reshape_1d<float, 2>)> kernel;
                    if (arg_rank == 1)
                    {
                        SELECT_KERNEL_BY_RANK(kernel,
                                              result_element_type,
                                              result_rank,
                                              runtime::cpu::kernel::reshape_1d);
                    }
                    else if (arg_rank == 2)
                    {
                        SELECT_KERNEL_BY_RANK(kernel,
                                              result_element_type,
                                              result_rank,
                                              runtime::cpu::kernel::reshape_2d);
                    }
                    else if (arg_rank == 3)
                    {
                        SELECT_KERNEL_BY_RANK(kernel,
                                              result_element_type,
                                              result_rank,
                                              runtime::cpu::kernel::reshape_3d);
                    }
                    else if (arg_rank == 4)
                    {
                        SELECT_KERNEL_BY_RANK(kernel,
                                              result_element_type,
                                              result_rank,
                                              runtime::cpu::kernel::reshape_4d);
                    }
                    else
                    {
                        std::function<decltype(runtime::cpu::kernel::reshape<float>)> ref_kernel;

                        SELECT_KERNEL(
                            ref_kernel, result_element_type, runtime::cpu::kernel::reshape);

                        auto functor = [&, ref_kernel, arg_shape, input_order, result_shape](
                            CPURuntimeContext* ctx) {
                            ref_kernel(
                                arg_tensor, out_tensor, arg_shape, input_order, result_shape);
                        };
                        functors.emplace_back(functor);
                        return;
                    }

                    auto functor =
                        [&, kernel, arg_shape, input_order, result_shape](CPURuntimeContext* ctx) {
                            kernel(arg_tensor, out_tensor, arg_shape, input_order, result_shape);
                        };
                    functors.emplace_back(functor);
                }
            }

            REGISTER_OP_BUILDER(Reshape);
        }
    }
}
