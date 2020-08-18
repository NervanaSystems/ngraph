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

#include <cstring>

#include "ngraph/op/constant.hpp"
#include "ngraph/op/dequantize.hpp"
#include "ngraph/op/quantize.hpp"
#include "ngraph/runtime/cpu/cpu_builder.hpp"
#include "ngraph/runtime/cpu/cpu_executor.hpp"
#include "ngraph/runtime/cpu/dnnl_invoke.hpp"
#include "ngraph/runtime/cpu/dnnl_utils.hpp"
#include "ngraph/runtime/reference/dequantize.hpp"
#include "ngraph/runtime/reference/quantize.hpp"

using namespace std;
using namespace ngraph;

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            template <>
            void Builder::BUILDER_DECL(ngraph::op::v0::Dequantize)
            {
                auto& functors = external_function->get_functors();

                const ngraph::op::v0::Dequantize* dequantize =
                    static_cast<const ngraph::op::v0::Dequantize*>(node);
                CPUKernelFunctor functor;

                if (runtime::cpu::dnnl_utils::use_dnnl_kernel(node))
                {
                    auto arg0_buffer_index =
                        external_function->get_buffer_index(args[0].get_name());
                    auto out_buffer_index = external_function->get_buffer_index(out[0].get_name());

                    auto& dnnl_emitter = external_function->get_dnnl_emitter();
                    auto input_desc = dnnl_utils::get_input_dnnl_md(node, 0);
                    auto result_desc = dnnl_utils::get_output_dnnl_md(node, 0);
                    size_t scratchpad_size =
                        QUERY_SCRATCHPAD_2ARGS(reorder, input_desc, result_desc);

                    auto scale_const_op =
                        as_type_ptr<ngraph::op::v0::Constant>(dequantize->get_argument(1));

                    if (scale_const_op == nullptr)
                    {
                        auto arg1_buffer_index =
                            external_function->get_buffer_index(args[1].get_name());
                        auto scales_size = shape_size(args[1].get_shape());

                        // Dequantize needs 3 primitives: input, result, and reorder.
                        size_t dequantize_index = dnnl_emitter->reserve_primitive_space(3);
                        auto& deps = dnnl_emitter->get_primitive_deps(dequantize_index);

                        functor = [&,
                                   input_desc,
                                   result_desc,
                                   scales_size,
                                   dequantize_index,
                                   scratchpad_size,
                                   arg0_buffer_index,
                                   arg1_buffer_index,
                                   out_buffer_index](CPURuntimeContext* ctx,
                                                     CPUExecutionContext* /* ectx */) {
                            // Create DNNL reorder primitive during the first iteration.
                            // Assumes the scales dont change for the duration of the graph
                            if (ctx->first_iteration)
                            {
                                vector<float> dyn_scales;
                                dyn_scales.assign(
                                    static_cast<float*>(ctx->buffer_data[arg1_buffer_index]),
                                    static_cast<float*>(ctx->buffer_data[arg1_buffer_index]) +
                                        scales_size);
                                dnnl_emitter->build_quantize_reorder(ctx->dnnl_memories,
                                                                     ctx->dnnl_primitives,
                                                                     ctx->dnnl_scratchpad_mds,
                                                                     input_desc,
                                                                     result_desc,
                                                                     dyn_scales,
                                                                     deps,
                                                                     dequantize_index);
                            }
                            cpu::dnnl_utils::set_memory_ptr(
                                ctx, deps[0], ctx->buffer_data[arg0_buffer_index]);
                            cpu::dnnl_utils::set_memory_ptr(
                                ctx, deps[1], ctx->buffer_data[out_buffer_index]);

                            cpu::dnnl_utils::dnnl_invoke_primitive(
                                ctx,
                                dequantize_index,
                                deps,
                                cpu::dnnl_utils::OpType::DEQUANTIZE,
                                scratchpad_size);
                        };
                        functors.emplace_back(functor);
                    }
                    else
                    {
                        std::vector<float> scale = scale_const_op->get_vector<float>();
                        std::vector<float> scales;
                        scales.push_back(scale[0]);
                        size_t dequantize_index = dnnl_emitter->reserve_primitive_space(3);
                        auto& deps = dnnl_emitter->get_primitive_deps(dequantize_index);

                        functor = [&,
                                   input_desc,
                                   result_desc,
                                   scales,
                                   dequantize_index,
                                   scratchpad_size,
                                   arg0_buffer_index,
                                   out_buffer_index](CPURuntimeContext* ctx,
                                                     CPUExecutionContext* /* ectx */) {
                            if (ctx->first_iteration)
                            {
                                dnnl_emitter->build_quantize_reorder(ctx->dnnl_memories,
                                                                     ctx->dnnl_primitives,
                                                                     ctx->dnnl_scratchpad_mds,
                                                                     input_desc,
                                                                     result_desc,
                                                                     scales,
                                                                     deps,
                                                                     dequantize_index);
                            }
                            cpu::dnnl_utils::set_memory_ptr(
                                ctx, deps[0], ctx->buffer_data[arg0_buffer_index]);
                            cpu::dnnl_utils::set_memory_ptr(
                                ctx, deps[1], ctx->buffer_data[out_buffer_index]);

                            cpu::dnnl_utils::dnnl_invoke_primitive(
                                ctx,
                                dequantize_index,
                                deps,
                                cpu::dnnl_utils::OpType::DEQUANTIZE,
                                scratchpad_size);
                        };
                        functors.emplace_back(functor);
                    }
                }
                else
                {
                    auto arg0_buffer_index =
                        external_function->get_buffer_index(args[0].get_name());
                    auto arg1_buffer_index =
                        external_function->get_buffer_index(args[1].get_name());
                    auto arg2_buffer_index =
                        external_function->get_buffer_index(args[2].get_name());
                    auto out_buffer_index = external_function->get_buffer_index(out[0].get_name());

                    auto arg0_shape = args[0].get_shape();
                    auto arg1_shape = args[1].get_shape();
                    auto daxes = dequantize->get_axes();

                    if (args[0].get_element_type() == element::i8)
                    {
                        if (out[0].get_element_type() == element::f32)
                        {
                            functor = [&,
                                       arg0_shape,
                                       arg1_shape,
                                       daxes,
                                       arg0_buffer_index,
                                       arg1_buffer_index,
                                       arg2_buffer_index,
                                       out_buffer_index](CPURuntimeContext* ctx,
                                                         CPUExecutionContext* /* ectx */) {
                                ngraph::runtime::reference::dequantize<int8_t>(
                                    static_cast<int8_t*>(ctx->buffer_data[arg0_buffer_index]),
                                    static_cast<float*>(ctx->buffer_data[arg1_buffer_index]),
                                    static_cast<int8_t*>(ctx->buffer_data[arg2_buffer_index]),
                                    static_cast<float*>(ctx->buffer_data[out_buffer_index]),
                                    arg0_shape,
                                    arg1_shape,
                                    daxes);
                            };
                        }
                        else if (out[0].get_element_type() == element::f64)
                        {
                            functor = [&,
                                       arg0_shape,
                                       arg1_shape,
                                       daxes,
                                       arg0_buffer_index,
                                       arg1_buffer_index,
                                       arg2_buffer_index,
                                       out_buffer_index](CPURuntimeContext* ctx,
                                                         CPUExecutionContext* /* ectx */) {
                                ngraph::runtime::reference::dequantize<int8_t>(
                                    static_cast<int8_t*>(ctx->buffer_data[arg0_buffer_index]),
                                    static_cast<double*>(ctx->buffer_data[arg1_buffer_index]),
                                    static_cast<int8_t*>(ctx->buffer_data[arg2_buffer_index]),
                                    static_cast<double*>(ctx->buffer_data[out_buffer_index]),
                                    arg0_shape,
                                    arg1_shape,
                                    daxes);
                            };
                        }
                        else
                        {
                            throw ngraph_error("Unsupported dequantization element type");
                        }
                    }
                    else if (args[0].get_element_type() == element::u8)
                    {
                        if (out[0].get_element_type() == element::f32)
                        {
                            functor = [&,
                                       arg0_shape,
                                       arg1_shape,
                                       daxes,
                                       arg0_buffer_index,
                                       arg1_buffer_index,
                                       arg2_buffer_index,
                                       out_buffer_index](CPURuntimeContext* ctx,
                                                         CPUExecutionContext* /* ectx */) {
                                ngraph::runtime::reference::dequantize<uint8_t>(
                                    static_cast<uint8_t*>(ctx->buffer_data[arg0_buffer_index]),
                                    static_cast<float*>(ctx->buffer_data[arg1_buffer_index]),
                                    static_cast<uint8_t*>(ctx->buffer_data[arg2_buffer_index]),
                                    static_cast<float*>(ctx->buffer_data[out_buffer_index]),
                                    arg0_shape,
                                    arg1_shape,
                                    daxes);
                            };
                        }
                        else if (out[0].get_element_type() == element::f64)
                        {
                            functor = [&,
                                       arg0_shape,
                                       arg1_shape,
                                       daxes,
                                       arg0_buffer_index,
                                       arg1_buffer_index,
                                       arg2_buffer_index,
                                       out_buffer_index](CPURuntimeContext* ctx,
                                                         CPUExecutionContext* /* ectx */) {
                                ngraph::runtime::reference::dequantize<uint8_t>(
                                    static_cast<uint8_t*>(ctx->buffer_data[arg0_buffer_index]),
                                    static_cast<double*>(ctx->buffer_data[arg1_buffer_index]),
                                    static_cast<uint8_t*>(ctx->buffer_data[arg2_buffer_index]),
                                    static_cast<double*>(ctx->buffer_data[out_buffer_index]),
                                    arg0_shape,
                                    arg1_shape,
                                    daxes);
                            };
                        }
                        else
                        {
                            throw ngraph_error("Unsupported dequantization element type");
                        }
                    }
                    else if (args[0].get_element_type() == element::i32)
                    {
                        if (out[0].get_element_type() == element::f32)
                        {
                            functor = [&,
                                       arg0_shape,
                                       arg1_shape,
                                       daxes,
                                       arg0_buffer_index,
                                       arg1_buffer_index,
                                       arg2_buffer_index,
                                       out_buffer_index](CPURuntimeContext* ctx,
                                                         CPUExecutionContext* /* ectx */) {
                                ngraph::runtime::reference::dequantize<int32_t>(
                                    static_cast<int32_t*>(ctx->buffer_data[arg0_buffer_index]),
                                    static_cast<float*>(ctx->buffer_data[arg1_buffer_index]),
                                    static_cast<int32_t*>(ctx->buffer_data[arg2_buffer_index]),
                                    static_cast<float*>(ctx->buffer_data[out_buffer_index]),
                                    arg0_shape,
                                    arg1_shape,
                                    daxes);
                            };
                        }
                        else if (out[0].get_element_type() == element::f64)
                        {
                            functor = [&,
                                       arg0_shape,
                                       arg1_shape,
                                       daxes,
                                       arg0_buffer_index,
                                       arg1_buffer_index,
                                       arg2_buffer_index,
                                       out_buffer_index](CPURuntimeContext* ctx,
                                                         CPUExecutionContext* /* ectx */) {
                                ngraph::runtime::reference::dequantize<int32_t>(
                                    static_cast<int32_t*>(ctx->buffer_data[arg0_buffer_index]),
                                    static_cast<double*>(ctx->buffer_data[arg1_buffer_index]),
                                    static_cast<int32_t*>(ctx->buffer_data[arg2_buffer_index]),
                                    static_cast<double*>(ctx->buffer_data[out_buffer_index]),
                                    arg0_shape,
                                    arg1_shape,
                                    daxes);
                            };
                        }
                        else
                        {
                            throw ngraph_error("Unsupported dequantization element type");
                        }
                    }
                    else
                    {
                        throw ngraph_error("Unsupported input element type");
                    }
                    functors.emplace_back(functor);
                }
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::v0::Quantize)
            {
                if (runtime::cpu::dnnl_utils::use_dnnl_kernel(node))
                {
                    auto quantize = static_cast<const ngraph::op::v0::Quantize*>(node);
                    auto& functors = external_function->get_functors();
                    auto arg0_buffer_index =
                        external_function->get_buffer_index(args[0].get_name());
                    auto out_buffer_index = external_function->get_buffer_index(out[0].get_name());
                    auto& dnnl_emitter = external_function->get_dnnl_emitter();
                    auto input_desc = dnnl_utils::get_input_dnnl_md(node, 0);
                    auto result_desc = dnnl_utils::get_output_dnnl_md(node, 0);
                    size_t scratchpad_size =
                        QUERY_SCRATCHPAD_2ARGS(reorder, input_desc, result_desc);

                    auto scale_const_op = as_type_ptr<ngraph::op::v0::Constant>(
                        quantize->get_input_node_shared_ptr(1));
                    if (scale_const_op == nullptr)
                    {
                        auto arg1_buffer_index =
                            external_function->get_buffer_index(args[1].get_name());
                        auto scales_size = shape_size(args[1].get_shape());

                        // Quantize needs 3 primitives: input, result, and reorder.
                        size_t quantize_index = dnnl_emitter->reserve_primitive_space(3);
                        auto& deps = dnnl_emitter->get_primitive_deps(quantize_index);

                        auto functor = [&,
                                        input_desc,
                                        result_desc,
                                        scales_size,
                                        quantize_index,
                                        scratchpad_size,
                                        arg0_buffer_index,
                                        arg1_buffer_index,
                                        out_buffer_index](CPURuntimeContext* ctx,
                                                          CPUExecutionContext* /* ectx */) {
                            // Create DNNL reorder primitive during the first iteration.
                            // Assumes the scales dont change for the duration of the graph
                            if (ctx->first_iteration)
                            {
                                vector<float> dyn_scales;
                                dyn_scales.assign(
                                    static_cast<float*>(ctx->buffer_data[arg1_buffer_index]),
                                    static_cast<float*>(ctx->buffer_data[arg1_buffer_index]) +
                                        scales_size);
                                for (size_t i = 0; i < scales_size; i++)
                                {
                                    dyn_scales[i] = 1.0f / dyn_scales[i];
                                }
                                // quantize across first dim (mask=2^0) if dyn_scales is a vector
                                const int mask = scales_size == 1 ? 0 : 1;
                                dnnl_emitter->build_quantize_reorder(ctx->dnnl_memories,
                                                                     ctx->dnnl_primitives,
                                                                     ctx->dnnl_scratchpad_mds,
                                                                     input_desc,
                                                                     result_desc,
                                                                     dyn_scales,
                                                                     deps,
                                                                     quantize_index,
                                                                     mask);
                            }
                            cpu::dnnl_utils::set_memory_ptr(
                                ctx, deps[0], ctx->buffer_data[arg0_buffer_index]);
                            cpu::dnnl_utils::set_memory_ptr(
                                ctx, deps[1], ctx->buffer_data[out_buffer_index]);

                            cpu::dnnl_utils::dnnl_invoke_primitive(
                                ctx,
                                quantize_index,
                                deps,
                                cpu::dnnl_utils::OpType::QUANTIZE,
                                scratchpad_size);
                        };
                        functors.emplace_back(functor);
                    }
                    else
                    {
                        auto scale = scale_const_op->get_vector<float>();
                        std::vector<float> scales;
                        scales.push_back(1.0f / scale[0]);
                        size_t quantize_index = dnnl_emitter->reserve_primitive_space(3);
                        auto& deps = dnnl_emitter->get_primitive_deps(quantize_index);

                        auto functor = [&,
                                        input_desc,
                                        result_desc,
                                        scales,
                                        quantize_index,
                                        scratchpad_size,
                                        arg0_buffer_index,
                                        out_buffer_index](CPURuntimeContext* ctx,
                                                          CPUExecutionContext* /* ectx */) {
                            if (ctx->first_iteration)
                            {
                                dnnl_emitter->build_quantize_reorder(ctx->dnnl_memories,
                                                                     ctx->dnnl_primitives,
                                                                     ctx->dnnl_scratchpad_mds,
                                                                     input_desc,
                                                                     result_desc,
                                                                     scales,
                                                                     deps,
                                                                     quantize_index);
                            }
                            cpu::dnnl_utils::set_memory_ptr(
                                ctx, deps[0], ctx->buffer_data[arg0_buffer_index]);
                            cpu::dnnl_utils::set_memory_ptr(
                                ctx, deps[1], ctx->buffer_data[out_buffer_index]);

                            cpu::dnnl_utils::dnnl_invoke_primitive(
                                ctx,
                                quantize_index,
                                deps,
                                cpu::dnnl_utils::OpType::QUANTIZE,
                                scratchpad_size);
                        };
                        functors.emplace_back(functor);
                    }
                }
                else
                {
                    auto& functors = external_function->get_functors();

                    const ngraph::op::v0::Quantize* quantize =
                        static_cast<const ngraph::op::v0::Quantize*>(node);
                    CPUKernelFunctor functor;

                    auto arg0_buffer_index =
                        external_function->get_buffer_index(args[0].get_name());
                    auto arg1_buffer_index =
                        external_function->get_buffer_index(args[1].get_name());
                    auto arg2_buffer_index =
                        external_function->get_buffer_index(args[2].get_name());
                    auto out_buffer_index = external_function->get_buffer_index(out[0].get_name());

                    auto arg0_shape = args[0].get_shape();
                    auto arg1_shape = args[1].get_shape();
                    auto daxes = quantize->get_axes();
                    ngraph::op::v0::Quantize::RoundMode round_mode = quantize->get_round_mode();

                    if (args[0].get_element_type() == element::f32)
                    {
                        if (out[0].get_element_type() == element::i8)
                        {
                            functor = [&,
                                       arg0_shape,
                                       arg1_shape,
                                       daxes,
                                       round_mode,
                                       arg0_buffer_index,
                                       arg1_buffer_index,
                                       arg2_buffer_index,
                                       out_buffer_index](CPURuntimeContext* ctx,
                                                         CPUExecutionContext* /* ectx */) {
                                ngraph::runtime::reference::quantize<float>(
                                    static_cast<float*>(ctx->buffer_data[arg0_buffer_index]),
                                    static_cast<float*>(ctx->buffer_data[arg1_buffer_index]),
                                    static_cast<int8_t*>(ctx->buffer_data[arg2_buffer_index]),
                                    static_cast<int8_t*>(ctx->buffer_data[out_buffer_index]),
                                    arg0_shape,
                                    arg1_shape,
                                    daxes,
                                    round_mode);
                            };
                        }
                        else if (out[0].get_element_type() == element::u8)
                        {
                            functor = [&,
                                       arg0_shape,
                                       arg1_shape,
                                       daxes,
                                       round_mode,
                                       arg0_buffer_index,
                                       arg1_buffer_index,
                                       arg2_buffer_index,
                                       out_buffer_index](CPURuntimeContext* ctx,
                                                         CPUExecutionContext* /* ectx */) {
                                ngraph::runtime::reference::quantize<float>(
                                    static_cast<float*>(ctx->buffer_data[arg0_buffer_index]),
                                    static_cast<float*>(ctx->buffer_data[arg1_buffer_index]),
                                    static_cast<uint8_t*>(ctx->buffer_data[arg2_buffer_index]),
                                    static_cast<uint8_t*>(ctx->buffer_data[out_buffer_index]),
                                    arg0_shape,
                                    arg1_shape,
                                    daxes,
                                    round_mode);
                            };
                        }
                        else if (out[0].get_element_type() == element::i32)
                        {
                            functor = [&,
                                       arg0_shape,
                                       arg1_shape,
                                       daxes,
                                       round_mode,
                                       arg0_buffer_index,
                                       arg1_buffer_index,
                                       arg2_buffer_index,
                                       out_buffer_index](CPURuntimeContext* ctx,
                                                         CPUExecutionContext* /* ectx */) {
                                ngraph::runtime::reference::quantize<float>(
                                    static_cast<float*>(ctx->buffer_data[arg0_buffer_index]),
                                    static_cast<float*>(ctx->buffer_data[arg1_buffer_index]),
                                    static_cast<int32_t*>(ctx->buffer_data[arg2_buffer_index]),
                                    static_cast<int32_t*>(ctx->buffer_data[out_buffer_index]),
                                    arg0_shape,
                                    arg1_shape,
                                    daxes,
                                    round_mode);
                            };
                        }
                        else
                        {
                            throw ngraph_error("Unsupported quantization element type");
                        }
                    }
                    else if (args[0].get_element_type() == element::f64)
                    {
                        if (out[0].get_element_type() == element::i8)
                        {
                            functor = [&,
                                       arg0_shape,
                                       arg1_shape,
                                       daxes,
                                       round_mode,
                                       arg0_buffer_index,
                                       arg1_buffer_index,
                                       arg2_buffer_index,
                                       out_buffer_index](CPURuntimeContext* ctx,
                                                         CPUExecutionContext* /* ectx */) {
                                ngraph::runtime::reference::quantize<double>(
                                    static_cast<double*>(ctx->buffer_data[arg0_buffer_index]),
                                    static_cast<double*>(ctx->buffer_data[arg1_buffer_index]),
                                    static_cast<int8_t*>(ctx->buffer_data[arg2_buffer_index]),
                                    static_cast<int8_t*>(ctx->buffer_data[out_buffer_index]),
                                    arg0_shape,
                                    arg1_shape,
                                    daxes,
                                    round_mode);
                            };
                        }
                        else if (out[0].get_element_type() == element::u8)
                        {
                            functor = [&,
                                       arg0_shape,
                                       arg1_shape,
                                       daxes,
                                       round_mode,
                                       arg0_buffer_index,
                                       arg1_buffer_index,
                                       arg2_buffer_index,
                                       out_buffer_index](CPURuntimeContext* ctx,
                                                         CPUExecutionContext* /* ectx */) {
                                ngraph::runtime::reference::quantize<double>(
                                    static_cast<double*>(ctx->buffer_data[arg0_buffer_index]),
                                    static_cast<double*>(ctx->buffer_data[arg1_buffer_index]),
                                    static_cast<uint8_t*>(ctx->buffer_data[arg2_buffer_index]),
                                    static_cast<uint8_t*>(ctx->buffer_data[out_buffer_index]),
                                    arg0_shape,
                                    arg1_shape,
                                    daxes,
                                    round_mode);
                            };
                        }
                        else if (out[0].get_element_type() == element::i32)
                        {
                            functor = [&,
                                       arg0_shape,
                                       arg1_shape,
                                       daxes,
                                       round_mode,
                                       arg0_buffer_index,
                                       arg1_buffer_index,
                                       arg2_buffer_index,
                                       out_buffer_index](CPURuntimeContext* ctx,
                                                         CPUExecutionContext* /* ectx */) {
                                ngraph::runtime::reference::quantize<double>(
                                    static_cast<double*>(ctx->buffer_data[arg0_buffer_index]),
                                    static_cast<double*>(ctx->buffer_data[arg1_buffer_index]),
                                    static_cast<int32_t*>(ctx->buffer_data[arg2_buffer_index]),
                                    static_cast<int32_t*>(ctx->buffer_data[out_buffer_index]),
                                    arg0_shape,
                                    arg1_shape,
                                    daxes,
                                    round_mode);
                            };
                        }
                        else
                        {
                            throw ngraph_error("Unsupported quantization element type");
                        }
                    }
                    else
                    {
                        throw ngraph_error("Unsupported input element type");
                    }

                    functors.emplace_back(functor);
                }
            }

            void register_builders_quantization_cpp()
            {
                REGISTER_OP_BUILDER(ngraph::op::v0::Dequantize);
                REGISTER_OP_BUILDER(ngraph::op::v0::Quantize);
            }
        }
    }
}
