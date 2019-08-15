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

#include "ngraph/op/experimental/quantized_dot.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/experimental/quantized_dot_bias.hpp"
#include "ngraph/runtime/cpu/cpu_builder.hpp"
#include "ngraph/runtime/cpu/cpu_executor.hpp"
#include "ngraph/runtime/cpu/kernel/dot.hpp"
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
            void Builder::BUILDER_DECL(ngraph::op::QuantizedDotBias)
            {
                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    if (node->get_input_element_type(0) == element::u8 &&
                        node->get_input_element_type(1) == element::u8)
                    {
                        throw ngraph_error(
                            "Unsupported data types for QuantizedDot MKLDNN kernel.");
                    }
                    auto& functors = external_function->get_functors();
                    auto arg0_buffer_index =
                        external_function->get_buffer_index(args[0].get_name());
                    auto arg1_buffer_index =
                        external_function->get_buffer_index(args[1].get_name());
                    auto arg2_buffer_index =
                        external_function->get_buffer_index(args[2].get_name());
                    auto arg3_buffer_index =
                        external_function->get_buffer_index(args[3].get_name());
                    auto out0_buffer_index = external_function->get_buffer_index(out[0].get_name());

                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto scales_size = shape_size(args[3].get_shape());

                    auto ip_desc =
                        mkldnn_emitter
                            ->get_inner_product_forward_desc<ngraph::op::QuantizedDotBias>(node);
                    auto ip_attr =
                        mkldnn_emitter
                            ->get_inner_product_forward_attr<ngraph::op::QuantizedDotBias>(node);
                    size_t ip_index = mkldnn_emitter->inner_product_forward_init(true);
                    auto& deps = mkldnn_emitter->get_primitive_deps(ip_index);

                    auto functor = [&,
                                    scales_size,
                                    ip_desc,
                                    ip_attr,
                                    deps,
                                    ip_index,
                                    arg0_buffer_index,
                                    arg1_buffer_index,
                                    arg2_buffer_index,
                                    arg3_buffer_index,
                                    out0_buffer_index](CPURuntimeContext* ctx,
                                                       CPUExecutionContext* ectx) mutable {
                        if (ctx->first_iteration)
                        {
                            vector<float> dyn_scales;
                            dyn_scales.assign(
                                static_cast<float*>(ctx->buffer_data[arg3_buffer_index]),
                                static_cast<float*>(ctx->buffer_data[arg3_buffer_index]) +
                                    scales_size);
                            ip_attr.set_output_scales(0, dyn_scales);
                            mkldnn_emitter->build_inner_product_forward<true>(
                                ctx->mkldnn_primitives,
                                ip_desc,
                                ip_attr,
                                executor::global_cpu_engine,
                                deps,
                                ip_index);
                        }
                        cpu::mkldnn_utils::set_memory_ptr(
                            ctx, deps[0], ctx->buffer_data[arg0_buffer_index]);
                        cpu::mkldnn_utils::set_memory_ptr(
                            ctx, deps[1], ctx->buffer_data[arg1_buffer_index]);
                        cpu::mkldnn_utils::set_memory_ptr(
                            ctx, deps[2], ctx->buffer_data[arg2_buffer_index]);
                        cpu::mkldnn_utils::set_memory_ptr(
                            ctx, deps[3], ctx->buffer_data[out0_buffer_index]);
                        cpu::mkldnn_utils::mkldnn_invoke_primitive(ctx, ip_index);
                    };
                    functors.emplace_back(functor);
                }
                else
                {
                    throw ngraph_error("unsupported parameters for QuantizedDotBias via DEX");
                }
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::QuantizedDot)
            {
                auto& functors = external_function->get_functors();

                auto arg0_shape = args[0].get_shape();
                auto arg1_shape = args[1].get_shape();
                auto result_shape = out[0].get_shape();

                auto arg0_buffer_index = external_function->get_buffer_index(args[0].get_name());
                auto arg1_buffer_index = external_function->get_buffer_index(args[1].get_name());
                auto arg2_buffer_index = external_function->get_buffer_index(args[2].get_name());
                auto out0_buffer_index = external_function->get_buffer_index(out[0].get_name());

                if (shape_size(args[2].get_shape()) != 1)
                {
                    throw ngraph_error("Scale size should be 1 for QuantizedDot");
                }

                std::function<decltype(
                    runtime::cpu::kernel::dot_ref<uint8_t, uint8_t, uint8_t, int32_t>)>
                    kernel;

                kernel = runtime::cpu::kernel::dot_ref<uint8_t, uint8_t, uint8_t, int32_t>;

                auto functor = [&,
                                kernel,
                                arg0_shape,
                                arg1_shape,
                                result_shape,
                                arg0_buffer_index,
                                arg1_buffer_index,
                                arg2_buffer_index,
                                out0_buffer_index](CPURuntimeContext* ctx,
                                                   CPUExecutionContext* ectx) {

                    float dyn_scale = *(static_cast<float*>(ctx->buffer_data[arg2_buffer_index]));

                    kernel(ctx->buffer_data[arg0_buffer_index],
                           ctx->buffer_data[arg1_buffer_index],
                           ctx->buffer_data[out0_buffer_index],
                           arg0_shape,
                           arg1_shape,
                           result_shape,
                           1,
                           dyn_scale);
                };
                functors.emplace_back(functor);
            }
            REGISTER_OP_BUILDER(QuantizedDotBias);
            REGISTER_OP_BUILDER(QuantizedDot);
#ifdef NGRAPH_CPU_STATIC_LIB_ENABLE
            void register_builders_quantized_dot_cpp() {}
#endif
        }
    }
}
