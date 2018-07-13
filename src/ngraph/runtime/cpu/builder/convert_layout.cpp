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

#include "ngraph/runtime/cpu/op/convert_layout.hpp"
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
            void Builder::BUILDER_DECL(ngraph::runtime::cpu::op::ConvertLayout)
            {
                auto& functors = external_function->get_functors();
                auto& tensor_data = external_function->get_tensor_data();

                auto& arg_tensor = tensor_data[args[0].get_name()];
                auto& out_tensor = tensor_data[out[0].get_name()];

                auto input_tvl =
                    node->get_inputs()[0].get_output().get_tensor_view()->get_tensor_view_layout();
                auto input_cpu_tvl =
                    dynamic_pointer_cast<runtime::cpu::LayoutDescriptor>(input_tvl);
                auto input_format = input_cpu_tvl->get_mkldnn_format();

                // Reorder input shape if needed
                auto input_axis_order = input_cpu_tvl->get_axis_order();
                Shape input_shape(input_axis_order.size());
                for (size_t idx = 0; idx < input_axis_order.size(); idx++)
                {
                    input_shape[idx] = args[0].get_shape()[input_axis_order[idx]];
                }

                auto output_tvl = node->get_output_tensor_view(0)->get_tensor_view_layout();
                auto output_format =
                    dynamic_cast<runtime::cpu::LayoutDescriptor&>(*output_tvl).get_mkldnn_format();

                // MKLDNN relies on format names for selecting optimized kernel implementations
                // Hacky way to deal with this until they move to using canonicalized layouts
                if (input_format == mkldnn::memory::format::nchw &&
                    runtime::cpu::mkldnn_utils::is_mkldnn_filter_format(output_format))
                {
                    input_format = mkldnn::memory::format::oihw;
                }
                if (output_format == mkldnn::memory::format::nchw &&
                    runtime::cpu::mkldnn_utils::is_mkldnn_filter_format(input_format))
                {
                    output_format = mkldnn::memory::format::oihw;
                }

                auto& mkldnn_emitter = external_function->get_mkldnn_emitter();

                auto input_desc = mkldnn_emitter->build_memory_descriptor(
                    input_shape, args[0].get_element_type(), input_format);
                auto result_desc = mkldnn_emitter->build_memory_descriptor(out[0], output_format);

                size_t reorder_index = mkldnn_emitter->build_reorder(input_desc, result_desc);

                auto& deps = mkldnn_emitter->get_primitive_deps(reorder_index);
                auto functor = [&, reorder_index](CPURuntimeContext* ctx) {
                    cpu::mkldnn_utils::set_memory_ptr(ctx, deps[0], arg_tensor);
                    cpu::mkldnn_utils::set_memory_ptr(ctx, deps[1], out_tensor);
                    cpu::mkldnn_utils::mkldnn_invoke_primitive(ctx, reorder_index);
                };
                functors.emplace_back(functor);
            }
        }
    }
}
