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

#include "ngraph/op/constant.hpp"
#include "ngraph/op/experimental/quantized_conv_bias.hpp"
#include "ngraph/op/experimental/quantized_conv_relu.hpp"
#include "ngraph/op/quantized_convolution.hpp"
#include "ngraph/runtime/cpu/cpu_builder.hpp"
#include "ngraph/runtime/cpu/cpu_executor.hpp"
#include "ngraph/runtime/cpu/kernel/convolution.hpp"
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
            void Builder::BUILDER_DECL(ngraph::op::QuantizedConvolution)
            {
                auto qconvolution = static_cast<const ngraph::op::QuantizedConvolution*>(node);

                auto& functors = external_function->get_functors();

                auto arg0_shape = args[0].get_shape();
                auto arg1_shape = args[1].get_shape();
                auto result_shape = out[0].get_shape();

                auto arg0_buffer_index = external_function->get_buffer_index(args[0].get_name());
                auto arg1_buffer_index = external_function->get_buffer_index(args[1].get_name());
                auto arg2_buffer_index =
                    external_function->get_buffer_index(args[2].get_name()); // input scale
                auto arg4_buffer_index =
                    external_function->get_buffer_index(args[4].get_name()); // filter scale
                auto arg6_buffer_index =
                    external_function->get_buffer_index(args[6].get_name()); // output scale
                auto out0_buffer_index = external_function->get_buffer_index(out[0].get_name());

                auto scales_size = shape_size(args[2].get_shape());

                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();

                    auto conv_desc =
                        mkldnn_emitter
                            ->get_convolution_forward_desc<ngraph::op::QuantizedConvolution>(node);
                    auto conv_attr =
                        mkldnn_emitter
                            ->get_convolution_forward_attr<ngraph::op::QuantizedConvolution>(node);
                    size_t scratchpad_size =
                        QUERY_SCRATCHPAD_2ARGS(convolution_forward, conv_desc, conv_attr);

                    size_t conv_index = mkldnn_emitter->convolution_forward_init();
                    auto& deps = mkldnn_emitter->get_primitive_deps(conv_index);

                    auto functor = [&,
                                    conv_desc,
                                    conv_attr,
                                    deps,
                                    conv_index,
                                    scratchpad_size,
                                    arg0_buffer_index,
                                    arg1_buffer_index,
                                    arg2_buffer_index,
                                    arg4_buffer_index,
                                    arg6_buffer_index,
                                    out0_buffer_index](CPURuntimeContext* ctx,
                                                       CPUExecutionContext* /* ectx */) mutable {
                        // Create MKLDNN convolution primitive during the first iteration.
                        // Assumes the scales dont change for the duration of the graph
                        if (ctx->first_iteration)
                        {
                            vector<float> dyn_scales;
                            // Calculate the requantization scale
                            dyn_scales.push_back(
                                *(static_cast<float*>(ctx->buffer_data[arg2_buffer_index])) *
                                *(static_cast<float*>(ctx->buffer_data[arg4_buffer_index])) /
                                *(static_cast<float*>(ctx->buffer_data[arg6_buffer_index])));
                            // use conv channelwise (dim 1, mask=2^1) if dyn_scales is a vector
                            conv_attr.set_output_scales(0, dyn_scales);
                            mkldnn_emitter->build_convolution_forward<false>(
                                ctx->mkldnn_memories,
                                ctx->mkldnn_primitives,
                                ctx->mkldnn_scratchpad_mds,
                                conv_desc,
                                conv_attr,
                                executor::global_cpu_engine,
                                deps,
                                conv_index);
                        }
                        cpu::mkldnn_utils::set_memory_ptr(
                            ctx, deps[0], ctx->buffer_data[arg0_buffer_index]);
                        cpu::mkldnn_utils::set_memory_ptr(
                            ctx, deps[1], ctx->buffer_data[arg1_buffer_index]);
                        cpu::mkldnn_utils::set_memory_ptr(
                            ctx, deps[2], ctx->buffer_data[out0_buffer_index]);

                        cpu::mkldnn_utils::mkldnn_invoke_primitive(
                            ctx,
                            conv_index,
                            deps,
                            cpu::mkldnn_utils::OpType::QUANTIZEDCONVOLUTION,
                            scratchpad_size);
                    };
                    functors.emplace_back(functor);
                }
                else if (args[0].get_element_type() == element::u8 &&
                         args[1].get_element_type() == element::u8 &&
                         out[0].get_element_type() == element::u8)
                {
                    std::function<decltype(
                        runtime::cpu::kernel::convolution<uint8_t, uint8_t, uint8_t, int32_t>)>
                        kernel;
                    kernel = runtime::cpu::kernel::convolution<uint8_t, uint8_t, uint8_t, int32_t>;

                    auto arg3_buffer_index =
                        external_function->get_buffer_index(args[3].get_name()); // input scale
                    auto arg5_buffer_index =
                        external_function->get_buffer_index(args[5].get_name()); // filter scale
                    auto arg7_buffer_index =
                        external_function->get_buffer_index(args[7].get_name()); // output scale

                    auto window_movement_strides = qconvolution->get_window_movement_strides();
                    auto window_dilation_strides = qconvolution->get_window_dilation_strides();
                    auto padding_below = qconvolution->get_padding_below();
                    auto padding_above = qconvolution->get_padding_above();
                    auto data_dilation_strides = qconvolution->get_data_dilation_strides();

                    auto functor = [&,
                                    kernel,
                                    arg0_shape,
                                    arg1_shape,
                                    arg0_buffer_index,
                                    arg1_buffer_index,
                                    arg2_buffer_index,
                                    arg3_buffer_index,
                                    arg4_buffer_index,
                                    arg5_buffer_index,
                                    arg6_buffer_index,
                                    arg7_buffer_index,
                                    out0_buffer_index,
                                    result_shape,
                                    window_movement_strides,
                                    window_dilation_strides,
                                    padding_below,
                                    padding_above,
                                    data_dilation_strides,
                                    scales_size](CPURuntimeContext* ctx,
                                                 CPUExecutionContext* /* ectx */) {
                        vector<float> dyn_scales;
                        dyn_scales.assign(static_cast<float*>(ctx->buffer_data[arg2_buffer_index]),
                                          static_cast<float*>(ctx->buffer_data[arg2_buffer_index]) +
                                              scales_size);
                        kernel(ctx->buffer_data[arg0_buffer_index],
                               ctx->buffer_data[arg1_buffer_index],
                               ctx->buffer_data[out0_buffer_index],
                               arg0_shape,
                               arg1_shape,
                               result_shape,
                               window_movement_strides,
                               window_dilation_strides,
                               padding_below,
                               padding_above,
                               data_dilation_strides,
                               ctx->buffer_data[arg2_buffer_index],
                               ctx->buffer_data[arg3_buffer_index],
                               ctx->buffer_data[arg4_buffer_index],
                               ctx->buffer_data[arg5_buffer_index],
                               ctx->buffer_data[arg6_buffer_index],
                               ctx->buffer_data[arg7_buffer_index]);
                    };
                    functors.emplace_back(functor);
                }
                else if (args[0].get_element_type() == element::u8 &&
                         args[1].get_element_type() == element::u8 &&
                         out[0].get_element_type() == element::i32)
                {
                    std::function<decltype(
                        runtime::cpu::kernel::convolution<uint8_t, uint8_t, int32_t, int32_t>)>
                        kernel;
                    kernel = runtime::cpu::kernel::convolution<uint8_t, uint8_t, int32_t, int32_t>;

                    auto arg3_buffer_index =
                        external_function->get_buffer_index(args[3].get_name()); // input scale
                    auto arg5_buffer_index =
                        external_function->get_buffer_index(args[5].get_name()); // filter scale
                    auto arg7_buffer_index =
                        external_function->get_buffer_index(args[7].get_name()); // output scale

                    auto window_movement_strides = qconvolution->get_window_movement_strides();
                    auto window_dilation_strides = qconvolution->get_window_dilation_strides();
                    auto padding_below = qconvolution->get_padding_below();
                    auto padding_above = qconvolution->get_padding_above();
                    auto data_dilation_strides = qconvolution->get_data_dilation_strides();

                    auto functor = [&,
                                    kernel,
                                    arg0_shape,
                                    arg1_shape,
                                    arg0_buffer_index,
                                    arg1_buffer_index,
                                    arg2_buffer_index,
                                    arg3_buffer_index,
                                    arg4_buffer_index,
                                    arg5_buffer_index,
                                    arg6_buffer_index,
                                    arg7_buffer_index,
                                    out0_buffer_index,
                                    result_shape,
                                    window_movement_strides,
                                    window_dilation_strides,
                                    padding_below,
                                    padding_above,
                                    data_dilation_strides,
                                    scales_size](CPURuntimeContext* ctx,
                                                 CPUExecutionContext* /* ectx */) {
                        vector<float> dyn_scales;
                        dyn_scales.assign(static_cast<float*>(ctx->buffer_data[arg2_buffer_index]),
                                          static_cast<float*>(ctx->buffer_data[arg2_buffer_index]) +
                                              scales_size);
                        kernel(ctx->buffer_data[arg0_buffer_index],
                               ctx->buffer_data[arg1_buffer_index],
                               ctx->buffer_data[out0_buffer_index],
                               arg0_shape,
                               arg1_shape,
                               result_shape,
                               window_movement_strides,
                               window_dilation_strides,
                               padding_below,
                               padding_above,
                               data_dilation_strides,
                               ctx->buffer_data[arg2_buffer_index],
                               ctx->buffer_data[arg3_buffer_index],
                               ctx->buffer_data[arg4_buffer_index],
                               ctx->buffer_data[arg5_buffer_index],
                               ctx->buffer_data[arg6_buffer_index],
                               ctx->buffer_data[arg7_buffer_index]);
                    };
                    functors.emplace_back(functor);
                }
                else if (args[0].get_element_type() == element::u8 &&
                         args[1].get_element_type() == element::i8 &&
                         out[0].get_element_type() == element::i32)
                {
                    std::function<decltype(
                        runtime::cpu::kernel::convolution<uint8_t, int8_t, int32_t, int32_t>)>
                        kernel;
                    kernel = runtime::cpu::kernel::convolution<uint8_t, int8_t, int32_t, int32_t>;

                    auto arg3_buffer_index =
                        external_function->get_buffer_index(args[3].get_name()); // input scale
                    auto arg5_buffer_index =
                        external_function->get_buffer_index(args[5].get_name()); // filter scale
                    auto arg7_buffer_index =
                        external_function->get_buffer_index(args[7].get_name()); // output scale

                    auto window_movement_strides = qconvolution->get_window_movement_strides();
                    auto window_dilation_strides = qconvolution->get_window_dilation_strides();
                    auto padding_below = qconvolution->get_padding_below();
                    auto padding_above = qconvolution->get_padding_above();
                    auto data_dilation_strides = qconvolution->get_data_dilation_strides();

                    auto functor = [&,
                                    kernel,
                                    arg0_shape,
                                    arg1_shape,
                                    arg0_buffer_index,
                                    arg1_buffer_index,
                                    arg2_buffer_index,
                                    arg3_buffer_index,
                                    arg4_buffer_index,
                                    arg5_buffer_index,
                                    arg6_buffer_index,
                                    arg7_buffer_index,
                                    out0_buffer_index,
                                    result_shape,
                                    window_movement_strides,
                                    window_dilation_strides,
                                    padding_below,
                                    padding_above,
                                    data_dilation_strides,
                                    scales_size](CPURuntimeContext* ctx,
                                                 CPUExecutionContext* /* ectx */) {
                        vector<float> dyn_scales;
                        dyn_scales.assign(static_cast<float*>(ctx->buffer_data[arg2_buffer_index]),
                                          static_cast<float*>(ctx->buffer_data[arg2_buffer_index]) +
                                              scales_size);
                        kernel(ctx->buffer_data[arg0_buffer_index],
                               ctx->buffer_data[arg1_buffer_index],
                               ctx->buffer_data[out0_buffer_index],
                               arg0_shape,
                               arg1_shape,
                               result_shape,
                               window_movement_strides,
                               window_dilation_strides,
                               padding_below,
                               padding_above,
                               data_dilation_strides,
                               ctx->buffer_data[arg2_buffer_index],
                               ctx->buffer_data[arg3_buffer_index],
                               ctx->buffer_data[arg4_buffer_index],
                               ctx->buffer_data[arg5_buffer_index],
                               ctx->buffer_data[arg6_buffer_index],
                               ctx->buffer_data[arg7_buffer_index]);
                    };
                    functors.emplace_back(functor);
                }
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::QuantizedConvolutionRelu)
            {
                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    auto& functors = external_function->get_functors();
                    auto arg0_buffer_index =
                        external_function->get_buffer_index(args[0].get_name());
                    auto arg1_buffer_index =
                        external_function->get_buffer_index(args[1].get_name());
                    auto arg2_buffer_index =
                        external_function->get_buffer_index(args[2].get_name());
                    auto out0_buffer_index = external_function->get_buffer_index(out[0].get_name());

                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto scales_size = shape_size(args[2].get_shape());

                    auto conv_desc =
                        mkldnn_emitter
                            ->get_convolution_forward_desc<ngraph::op::QuantizedConvolutionRelu>(
                                node);
                    auto conv_attr =
                        mkldnn_emitter
                            ->get_convolution_forward_attr<ngraph::op::QuantizedConvolutionRelu>(
                                node);
                    size_t scratchpad_size =
                        QUERY_SCRATCHPAD_2ARGS(convolution_forward, conv_desc, conv_attr);

                    size_t conv_index = mkldnn_emitter->convolution_forward_init();
                    auto& deps = mkldnn_emitter->get_primitive_deps(conv_index);

                    auto functor = [&,
                                    scales_size,
                                    conv_desc,
                                    conv_attr,
                                    deps,
                                    conv_index,
                                    scratchpad_size,
                                    arg0_buffer_index,
                                    arg1_buffer_index,
                                    arg2_buffer_index,
                                    out0_buffer_index](CPURuntimeContext* ctx,
                                                       CPUExecutionContext* /* ectx */) mutable {
                        if (ctx->first_iteration)
                        {
                            vector<float> dyn_scales;
                            dyn_scales.assign(
                                static_cast<float*>(ctx->buffer_data[arg2_buffer_index]),
                                static_cast<float*>(ctx->buffer_data[arg2_buffer_index]) +
                                    scales_size);
                            // use conv channelwise (dim 1, mask=2^1) if dyn_scales is a vector
                            const int mask = scales_size == 1 ? 0 : 2;
                            conv_attr.set_output_scales(mask, dyn_scales);
                            mkldnn_emitter->build_convolution_forward<false>(
                                ctx->mkldnn_memories,
                                ctx->mkldnn_primitives,
                                ctx->mkldnn_scratchpad_mds,
                                conv_desc,
                                conv_attr,
                                executor::global_cpu_engine,
                                deps,
                                conv_index);
                        }
                        cpu::mkldnn_utils::set_memory_ptr(
                            ctx, deps[0], ctx->buffer_data[arg0_buffer_index]);
                        cpu::mkldnn_utils::set_memory_ptr(
                            ctx, deps[1], ctx->buffer_data[arg1_buffer_index]);
                        cpu::mkldnn_utils::set_memory_ptr(
                            ctx, deps[2], ctx->buffer_data[out0_buffer_index]);

                        cpu::mkldnn_utils::mkldnn_invoke_primitive(
                            ctx,
                            conv_index,
                            deps,
                            cpu::mkldnn_utils::OpType::QUANTIZEDCONVOLUTIONRELU,
                            scratchpad_size);
                    };
                    functors.emplace_back(functor);
                }
                else
                {
                    throw ngraph_error(
                        "unsupported parameters for QuantizedConvolutionRelu via DEX");
                }
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::QuantizedConvolutionBias)
            {
                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
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

                    auto conv_desc =
                        mkldnn_emitter
                            ->get_convolution_forward_desc<ngraph::op::QuantizedConvolutionBias>(
                                node);
                    auto conv_attr =
                        mkldnn_emitter
                            ->get_convolution_forward_attr<ngraph::op::QuantizedConvolutionBias>(
                                node);
                    size_t scratchpad_size =
                        QUERY_SCRATCHPAD_2ARGS(convolution_forward, conv_desc, conv_attr);

                    size_t conv_index = mkldnn_emitter->convolution_forward_init(true);
                    auto& deps = mkldnn_emitter->get_primitive_deps(conv_index);

                    auto functor = [&,
                                    scales_size,
                                    conv_desc,
                                    conv_attr,
                                    deps,
                                    conv_index,
                                    scratchpad_size,
                                    arg0_buffer_index,
                                    arg1_buffer_index,
                                    arg2_buffer_index,
                                    arg3_buffer_index,
                                    out0_buffer_index](CPURuntimeContext* ctx,
                                                       CPUExecutionContext* /* ectx */) mutable {
                        if (ctx->first_iteration)
                        {
                            vector<float> dyn_scales;
                            dyn_scales.assign(
                                static_cast<float*>(ctx->buffer_data[arg3_buffer_index]),
                                static_cast<float*>(ctx->buffer_data[arg3_buffer_index]) +
                                    scales_size);
                            // use conv channelwise (dim 1, mask=2^1) if dyn_scales is a vector
                            const int mask = scales_size == 1 ? 0 : 2;
                            conv_attr.set_output_scales(mask, dyn_scales);
                            mkldnn_emitter->build_convolution_forward<true>(
                                ctx->mkldnn_memories,
                                ctx->mkldnn_primitives,
                                ctx->mkldnn_scratchpad_mds,
                                conv_desc,
                                conv_attr,
                                executor::global_cpu_engine,
                                deps,
                                conv_index);
                        }
                        cpu::mkldnn_utils::set_memory_ptr(
                            ctx, deps[0], ctx->buffer_data[arg0_buffer_index]);
                        cpu::mkldnn_utils::set_memory_ptr(
                            ctx, deps[1], ctx->buffer_data[arg1_buffer_index]);
                        cpu::mkldnn_utils::set_memory_ptr(
                            ctx, deps[2], ctx->buffer_data[arg2_buffer_index]);
                        cpu::mkldnn_utils::set_memory_ptr(
                            ctx, deps[3], ctx->buffer_data[out0_buffer_index]);

                        cpu::mkldnn_utils::mkldnn_invoke_primitive(
                            ctx,
                            conv_index,
                            deps,
                            cpu::mkldnn_utils::OpType::QUANTIZEDCONVOLUTIONBIAS,
                            scratchpad_size);
                    };
                    functors.emplace_back(functor);
                }
                else
                {
                    throw ngraph_error(
                        "unsupported parameters for QuantizedConvolutionBias via DEX");
                }
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::QuantizedConvolutionBiasAdd)
            {
                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    auto& functors = external_function->get_functors();
                    auto arg0_buffer_index =
                        external_function->get_buffer_index(args[0].get_name());
                    auto arg1_buffer_index =
                        external_function->get_buffer_index(args[1].get_name());
                    auto arg2_buffer_index =
                        external_function->get_buffer_index(args[2].get_name());
                    auto arg3_buffer_index =
                        external_function->get_buffer_index(args[3].get_name());
                    auto arg4_buffer_index =
                        external_function->get_buffer_index(args[4].get_name());
                    auto arg5_buffer_index =
                        external_function->get_buffer_index(args[5].get_name());
                    auto out0_buffer_index = external_function->get_buffer_index(out[0].get_name());
                    size_t arg3_size = node->input(3).get_tensor().size();

                    auto scales_size = shape_size(args[4].get_shape());
                    auto sum_scales_size = shape_size(args[5].get_shape());

                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();

                    auto conv_desc =
                        mkldnn_emitter
                            ->get_convolution_forward_desc<ngraph::op::QuantizedConvolutionBiasAdd>(
                                node);
                    auto conv_attr =
                        mkldnn_emitter
                            ->get_convolution_forward_attr<ngraph::op::QuantizedConvolutionBiasAdd>(
                                node);
                    size_t scratchpad_size =
                        QUERY_SCRATCHPAD_2ARGS(convolution_forward, conv_desc, conv_attr);

                    size_t conv_index = mkldnn_emitter->convolution_forward_init(true);
                    auto& deps = mkldnn_emitter->get_primitive_deps(conv_index);

                    auto functor = [&,
                                    scales_size,
                                    sum_scales_size,
                                    conv_desc,
                                    conv_attr,
                                    deps,
                                    conv_index,
                                    scratchpad_size,
                                    arg3_size,
                                    arg0_buffer_index,
                                    arg1_buffer_index,
                                    arg2_buffer_index,
                                    arg3_buffer_index,
                                    arg4_buffer_index,
                                    arg5_buffer_index,
                                    out0_buffer_index](CPURuntimeContext* ctx,
                                                       CPUExecutionContext* /* ectx */) mutable {
                        if (ctx->first_iteration)
                        {
                            vector<float> dyn_scales;
                            vector<float> dyn_post_op_scales;
                            dyn_scales.assign(
                                static_cast<float*>(ctx->buffer_data[arg4_buffer_index]),
                                static_cast<float*>(ctx->buffer_data[arg4_buffer_index]) +
                                    scales_size);
                            dyn_post_op_scales.assign(
                                static_cast<float*>(ctx->buffer_data[arg5_buffer_index]),
                                static_cast<float*>(ctx->buffer_data[arg5_buffer_index]) +
                                    sum_scales_size);
                            auto old_pops = conv_attr.get_post_ops();
                            mkldnn::post_ops new_pops;
                            for (int i = 0; i < old_pops.len(); i++)
                            {
                                if (old_pops.kind(i) == mkldnn::primitive::kind::eltwise)
                                {
                                    mkldnn::algorithm alg;
                                    float scale, alpha, beta;
                                    old_pops.get_params_eltwise(i, scale, alg, alpha, beta);
                                    new_pops.append_eltwise(scale, alg, alpha, beta);
                                }
                                if (old_pops.kind(i) == mkldnn::primitive::kind::sum)
                                {
                                    new_pops.append_sum(dyn_post_op_scales[0]);
                                }
                            }
                            // use conv channelwise (dim 1, mask=2^1) if dyn_scales is a vector
                            const int mask = scales_size == 1 ? 0 : 2;
                            conv_attr.set_output_scales(mask, dyn_scales);
                            conv_attr.set_post_ops(new_pops);
                            mkldnn_emitter->build_convolution_forward<true>(
                                ctx->mkldnn_memories,
                                ctx->mkldnn_primitives,
                                ctx->mkldnn_scratchpad_mds,
                                conv_desc,
                                conv_attr,
                                executor::global_cpu_engine,
                                deps,
                                conv_index);
                        }

                        if (ctx->buffer_data[out0_buffer_index] !=
                            ctx->buffer_data[arg3_buffer_index])
                        {
                            memcpy(static_cast<char*>(ctx->buffer_data[out0_buffer_index]),
                                   static_cast<char*>(ctx->buffer_data[arg3_buffer_index]),
                                   arg3_size);
                        }
                        cpu::mkldnn_utils::set_memory_ptr(
                            ctx, deps[0], ctx->buffer_data[arg0_buffer_index]);
                        cpu::mkldnn_utils::set_memory_ptr(
                            ctx, deps[1], ctx->buffer_data[arg1_buffer_index]);
                        cpu::mkldnn_utils::set_memory_ptr(
                            ctx, deps[2], ctx->buffer_data[arg2_buffer_index]);
                        cpu::mkldnn_utils::set_memory_ptr(
                            ctx, deps[3], ctx->buffer_data[out0_buffer_index]);

                        cpu::mkldnn_utils::mkldnn_invoke_primitive(
                            ctx,
                            conv_index,
                            deps,
                            cpu::mkldnn_utils::OpType::QUANTIZEDCONVOLUTIONBIASADD,
                            scratchpad_size);
                    };
                    functors.emplace_back(functor);
                }
                else
                {
                    throw ngraph_error(
                        "unsupported parameters for QuantizedConvolutionBiasAdd via DEX");
                }
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::QuantizedConvolutionBiasSignedAdd)
            {
                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    auto& functors = external_function->get_functors();
                    auto arg0_buffer_index =
                        external_function->get_buffer_index(args[0].get_name());
                    auto arg1_buffer_index =
                        external_function->get_buffer_index(args[1].get_name());
                    auto arg2_buffer_index =
                        external_function->get_buffer_index(args[2].get_name());
                    auto arg3_buffer_index =
                        external_function->get_buffer_index(args[3].get_name());
                    auto arg4_buffer_index =
                        external_function->get_buffer_index(args[4].get_name());
                    auto arg5_buffer_index =
                        external_function->get_buffer_index(args[5].get_name());
                    auto out0_buffer_index = external_function->get_buffer_index(out[0].get_name());
                    size_t arg3_size = node->input(3).get_tensor().size();

                    auto scales_size = shape_size(args[4].get_shape());
                    auto sum_scales_size = shape_size(args[5].get_shape());

                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();

                    auto conv_desc = mkldnn_emitter->get_convolution_forward_desc<
                        ngraph::op::QuantizedConvolutionBiasSignedAdd>(node);
                    auto conv_attr = mkldnn_emitter->get_convolution_forward_attr<
                        ngraph::op::QuantizedConvolutionBiasSignedAdd>(node);
                    size_t scratchpad_size =
                        QUERY_SCRATCHPAD_2ARGS(convolution_forward, conv_desc, conv_attr);

                    size_t conv_index = mkldnn_emitter->convolution_forward_init(true);
                    auto& deps = mkldnn_emitter->get_primitive_deps(conv_index);

                    auto functor = [&,
                                    scales_size,
                                    sum_scales_size,
                                    conv_desc,
                                    conv_attr,
                                    deps,
                                    conv_index,
                                    scratchpad_size,
                                    arg3_size,
                                    arg0_buffer_index,
                                    arg1_buffer_index,
                                    arg2_buffer_index,
                                    arg3_buffer_index,
                                    arg4_buffer_index,
                                    arg5_buffer_index,
                                    out0_buffer_index](CPURuntimeContext* ctx,
                                                       CPUExecutionContext* /* ectx */) mutable {
                        if (ctx->first_iteration)
                        {
                            vector<float> dyn_scales;
                            vector<float> dyn_post_op_scales;
                            dyn_scales.assign(
                                static_cast<float*>(ctx->buffer_data[arg4_buffer_index]),
                                static_cast<float*>(ctx->buffer_data[arg4_buffer_index]) +
                                    scales_size);
                            dyn_post_op_scales.assign(
                                static_cast<float*>(ctx->buffer_data[arg5_buffer_index]),
                                static_cast<float*>(ctx->buffer_data[arg5_buffer_index]) +
                                    sum_scales_size);
                            auto old_pops = conv_attr.get_post_ops();
                            mkldnn::post_ops new_pops;
                            for (int i = 0; i < old_pops.len(); i++)
                            {
                                if (old_pops.kind(i) == mkldnn::primitive::kind::eltwise)
                                {
                                    mkldnn::algorithm alg;
                                    float scale, alpha, beta;
                                    old_pops.get_params_eltwise(i, scale, alg, alpha, beta);
                                    new_pops.append_eltwise(scale, alg, alpha, beta);
                                }
                                if (old_pops.kind(i) == mkldnn::primitive::kind::sum)
                                {
                                    new_pops.append_sum(dyn_post_op_scales[0]);
                                }
                            }
                            conv_attr.set_post_ops(new_pops);
                            // use conv channelwise (dim 1, mask=2^1) if dyn_scales is a vector
                            const int mask = scales_size == 1 ? 0 : 2;
                            conv_attr.set_output_scales(mask, dyn_scales);
                            mkldnn_emitter->build_convolution_forward<true>(
                                ctx->mkldnn_memories,
                                ctx->mkldnn_primitives,
                                ctx->mkldnn_scratchpad_mds,
                                conv_desc,
                                conv_attr,
                                executor::global_cpu_engine,
                                deps,
                                conv_index);
                        }

                        if (ctx->buffer_data[out0_buffer_index] !=
                            ctx->buffer_data[arg3_buffer_index])
                        {
                            memcpy(static_cast<char*>(ctx->buffer_data[out0_buffer_index]),
                                   static_cast<char*>(ctx->buffer_data[arg3_buffer_index]),
                                   arg3_size);
                        }
                        cpu::mkldnn_utils::set_memory_ptr(
                            ctx, deps[0], ctx->buffer_data[arg0_buffer_index]);
                        cpu::mkldnn_utils::set_memory_ptr(
                            ctx, deps[1], ctx->buffer_data[arg1_buffer_index]);
                        cpu::mkldnn_utils::set_memory_ptr(
                            ctx, deps[2], ctx->buffer_data[arg2_buffer_index]);
                        cpu::mkldnn_utils::set_memory_ptr(
                            ctx, deps[3], ctx->buffer_data[out0_buffer_index]);

                        cpu::mkldnn_utils::mkldnn_invoke_primitive(
                            ctx,
                            conv_index,
                            deps,
                            cpu::mkldnn_utils::OpType::QUANTIZEDCONVOLUTIONBIASSIGNEDADD,
                            scratchpad_size);
                    };
                    functors.emplace_back(functor);
                }
                else
                {
                    throw ngraph_error(
                        "unsupported parameters for QuantizedConvolutionBiasSignedAdd via DEX");
                }
            }

            void register_builders_quantized_conv_cpp()
            {
                REGISTER_OP_BUILDER(QuantizedConvolution);
                REGISTER_OP_BUILDER(QuantizedConvolutionRelu);
                REGISTER_OP_BUILDER(QuantizedConvolutionBias);
                REGISTER_OP_BUILDER(QuantizedConvolutionBiasAdd);
                REGISTER_OP_BUILDER(QuantizedConvolutionBiasSignedAdd);
            }
        }
    }
}
