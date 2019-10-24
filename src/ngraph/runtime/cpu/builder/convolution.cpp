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

#include "ngraph/runtime/cpu/kernel/convolution.hpp"
#include "ngraph/op/fused/conv_fused.hpp"
#include "ngraph/op/fused/group_conv.hpp"
#include "ngraph/runtime/cpu/cpu_builder.hpp"
#include "ngraph/runtime/cpu/mkldnn_invoke.hpp"
#include "ngraph/runtime/cpu/mkldnn_utils.hpp"
#include "ngraph/runtime/cpu/op/conv_add.hpp"
#include "ngraph/runtime/cpu/op/conv_relu.hpp"
#include "ngraph/runtime/cpu/op/group_conv_bias.hpp"

using namespace std;
using namespace ngraph;

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            template <>
            void Builder::BUILDER_DECL(ngraph::op::Convolution)
            {
                auto convolution = static_cast<const ngraph::op::Convolution*>(node);

                auto& functors = external_function->get_functors();

                auto arg0_shape = args[0].get_shape();
                auto arg1_shape = args[1].get_shape();
                auto result_shape = out[0].get_shape();

                auto arg0_buffer_index = external_function->get_buffer_index(args[0].get_name());
                auto arg1_buffer_index = external_function->get_buffer_index(args[1].get_name());
                auto out_buffer_index = external_function->get_buffer_index(out[0].get_name());

                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto conv_desc =
                        mkldnn_emitter->get_convolution_forward_desc<ngraph::op::Convolution>(node);
                    auto conv_attr =
                        mkldnn_emitter->get_convolution_forward_attr<ngraph::op::Convolution>(node);
                    size_t scratchpad_size =
                        QUERY_SCRATCHPAD_2ARGS(convolution_forward, conv_desc, conv_attr);

                    size_t conv_index = mkldnn_emitter->convolution_forward_init();
                    auto& deps = mkldnn_emitter->get_primitive_deps(conv_index);

                    auto functor = [&,
                                    conv_desc,
                                    conv_attr,
                                    conv_index,
                                    scratchpad_size,
                                    arg0_buffer_index,
                                    arg1_buffer_index,
                                    out_buffer_index](CPURuntimeContext* ctx,
                                                      CPUExecutionContext* /* ectx */) {
                        if (ctx->first_iteration)
                        {
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
                            ctx, deps[2], ctx->buffer_data[out_buffer_index]);

                        cpu::mkldnn_utils::mkldnn_invoke_primitive(
                            ctx,
                            conv_index,
                            deps,
                            cpu::mkldnn_utils::OpType::CONVOLUTION,
                            scratchpad_size);
                    };
                    functors.emplace_back(functor);
                }
                else
                {
                    std::function<decltype(runtime::cpu::kernel::convolution<float, float, float>)>
                        kernel;

                    SELECT_KERNEL_3ARGS(
                        kernel, out[0].get_element_type(), runtime::cpu::kernel::convolution)

                    auto window_movement_strides = convolution->get_window_movement_strides();
                    auto window_dilation_strides = convolution->get_window_dilation_strides();
                    auto padding_below = convolution->get_padding_below();
                    auto padding_above = convolution->get_padding_above();
                    auto data_dilation_strides = convolution->get_data_dilation_strides();

                    auto functor = [&,
                                    kernel,
                                    arg0_shape,
                                    arg1_shape,
                                    result_shape,
                                    window_movement_strides,
                                    window_dilation_strides,
                                    padding_below,
                                    padding_above,
                                    data_dilation_strides,
                                    arg0_buffer_index,
                                    arg1_buffer_index,
                                    out_buffer_index](CPURuntimeContext* ctx,
                                                      CPUExecutionContext* /* ectx */) {
                        kernel(ctx->buffer_data[arg0_buffer_index],
                               ctx->buffer_data[arg1_buffer_index],
                               ctx->buffer_data[out_buffer_index],
                               arg0_shape,
                               arg1_shape,
                               result_shape,
                               window_movement_strides,
                               window_dilation_strides,
                               padding_below,
                               padding_above,
                               data_dilation_strides,
                               nullptr,
                               nullptr,
                               nullptr,
                               nullptr,
                               nullptr,
                               nullptr);
                    };
                    functors.emplace_back(functor);
                }
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::ConvolutionRelu)
            {
                auto& functors = external_function->get_functors();

                auto arg0_buffer_index = external_function->get_buffer_index(args[0].get_name());
                auto arg1_buffer_index = external_function->get_buffer_index(args[1].get_name());
                auto out_buffer_index = external_function->get_buffer_index(out[0].get_name());

                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto conv_desc =
                        mkldnn_emitter->get_convolution_forward_desc<ngraph::op::ConvolutionRelu>(
                            node);
                    auto conv_attr =
                        mkldnn_emitter->get_convolution_forward_attr<ngraph::op::ConvolutionRelu>(
                            node);
                    size_t scratchpad_size =
                        QUERY_SCRATCHPAD_2ARGS(convolution_forward, conv_desc, conv_attr);

                    size_t conv_index = mkldnn_emitter->convolution_forward_init();
                    auto& deps = mkldnn_emitter->get_primitive_deps(conv_index);

                    auto functor = [&,
                                    conv_desc,
                                    conv_attr,
                                    conv_index,
                                    scratchpad_size,
                                    arg0_buffer_index,
                                    arg1_buffer_index,
                                    out_buffer_index](CPURuntimeContext* ctx,
                                                      CPUExecutionContext* /* ectx */) {
                        if (ctx->first_iteration)
                        {
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
                            ctx, deps[2], ctx->buffer_data[out_buffer_index]);

                        cpu::mkldnn_utils::mkldnn_invoke_primitive(
                            ctx,
                            conv_index,
                            deps,
                            cpu::mkldnn_utils::OpType::CONVOLUTIONRELU,
                            scratchpad_size);
                    };
                    functors.emplace_back(functor);
                }
                else
                {
                    throw ngraph_error("ConvolutionRelu is only supported with MKLDNN kernel.");
                }
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::ConvolutionBias)
            {
                auto& functors = external_function->get_functors();

                auto arg0_buffer_index = external_function->get_buffer_index(args[0].get_name());
                auto arg1_buffer_index = external_function->get_buffer_index(args[1].get_name());
                auto arg2_buffer_index = external_function->get_buffer_index(args[2].get_name());
                auto out_buffer_index = external_function->get_buffer_index(out[0].get_name());

                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto conv_desc =
                        mkldnn_emitter->get_convolution_forward_desc<ngraph::op::ConvolutionBias>(
                            node);
                    auto conv_attr =
                        mkldnn_emitter->get_convolution_forward_attr<ngraph::op::ConvolutionBias>(
                            node);
                    size_t scratchpad_size =
                        QUERY_SCRATCHPAD_2ARGS(convolution_forward, conv_desc, conv_attr);

                    size_t conv_index = mkldnn_emitter->convolution_forward_init(true);
                    auto& deps = mkldnn_emitter->get_primitive_deps(conv_index);

                    auto functor = [&,
                                    conv_desc,
                                    conv_attr,
                                    conv_index,
                                    scratchpad_size,
                                    arg0_buffer_index,
                                    arg1_buffer_index,
                                    arg2_buffer_index,
                                    out_buffer_index](CPURuntimeContext* ctx,
                                                      CPUExecutionContext* /* ectx */) {
                        if (ctx->first_iteration)
                        {
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
                            ctx, deps[3], ctx->buffer_data[out_buffer_index]);

                        cpu::mkldnn_utils::mkldnn_invoke_primitive(
                            ctx,
                            conv_index,
                            deps,
                            cpu::mkldnn_utils::OpType::CONVOLUTIONBIAS,
                            scratchpad_size);
                    };
                    functors.emplace_back(functor);
                }
                else
                {
                    throw ngraph_error("ConvolutionBias is only supported with MKLDNN kernel.");
                }
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::ConvolutionBiasAdd)
            {
                auto& functors = external_function->get_functors();

                auto arg0_buffer_index = external_function->get_buffer_index(args[0].get_name());
                auto arg1_buffer_index = external_function->get_buffer_index(args[1].get_name());
                auto arg2_buffer_index = external_function->get_buffer_index(args[2].get_name());
                auto arg3_buffer_index = external_function->get_buffer_index(args[3].get_name());
                auto out_buffer_index = external_function->get_buffer_index(out[0].get_name());
                size_t arg3_size = node->input(3).get_tensor().size();

                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto conv_desc =
                        mkldnn_emitter
                            ->get_convolution_forward_desc<ngraph::op::ConvolutionBiasAdd>(node);
                    auto conv_attr =
                        mkldnn_emitter
                            ->get_convolution_forward_attr<ngraph::op::ConvolutionBiasAdd>(node);
                    size_t scratchpad_size =
                        QUERY_SCRATCHPAD_2ARGS(convolution_forward, conv_desc, conv_attr);

                    size_t conv_index = mkldnn_emitter->convolution_forward_init(true);
                    auto& deps = mkldnn_emitter->get_primitive_deps(conv_index);

                    auto functor = [&,
                                    conv_desc,
                                    conv_attr,
                                    conv_index,
                                    scratchpad_size,
                                    arg3_size,
                                    arg0_buffer_index,
                                    arg1_buffer_index,
                                    arg2_buffer_index,
                                    arg3_buffer_index,
                                    out_buffer_index](CPURuntimeContext* ctx,
                                                      CPUExecutionContext* /* ectx */) {
                        if (ctx->first_iteration)
                        {
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
                        if (ctx->buffer_data[out_buffer_index] !=
                            ctx->buffer_data[arg3_buffer_index])
                        {
                            memcpy(static_cast<char*>(ctx->buffer_data[out_buffer_index]),
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
                            ctx, deps[3], ctx->buffer_data[out_buffer_index]);

                        cpu::mkldnn_utils::mkldnn_invoke_primitive(
                            ctx,
                            conv_index,
                            deps,
                            cpu::mkldnn_utils::OpType::CONVOLUTIONBIASADD,
                            scratchpad_size);
                    };
                    functors.emplace_back(functor);
                }
                else
                {
                    throw ngraph_error("ConvolutionBiasAdd is only supported with MKLDNN kernel.");
                }
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::ConvolutionAdd)
            {
                auto& functors = external_function->get_functors();

                auto arg0_buffer_index = external_function->get_buffer_index(args[0].get_name());
                auto arg1_buffer_index = external_function->get_buffer_index(args[1].get_name());
                auto arg2_buffer_index = external_function->get_buffer_index(args[2].get_name());
                auto out_buffer_index = external_function->get_buffer_index(out[0].get_name());
                size_t arg2_size = node->input(2).get_tensor().size();

                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto conv_desc =
                        mkldnn_emitter->get_convolution_forward_desc<ngraph::op::ConvolutionAdd>(
                            node);
                    auto conv_attr =
                        mkldnn_emitter->get_convolution_forward_attr<ngraph::op::ConvolutionAdd>(
                            node);
                    size_t scratchpad_size =
                        QUERY_SCRATCHPAD_2ARGS(convolution_forward, conv_desc, conv_attr);

                    size_t conv_index = mkldnn_emitter->convolution_forward_init(false);
                    auto& deps = mkldnn_emitter->get_primitive_deps(conv_index);

                    auto functor = [&,
                                    conv_desc,
                                    conv_attr,
                                    conv_index,
                                    scratchpad_size,
                                    arg2_size,
                                    arg0_buffer_index,
                                    arg1_buffer_index,
                                    arg2_buffer_index,
                                    out_buffer_index](CPURuntimeContext* ctx,
                                                      CPUExecutionContext* /* ectx */) {
                        if (ctx->first_iteration)
                        {
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
                        if (ctx->buffer_data[out_buffer_index] !=
                            ctx->buffer_data[arg2_buffer_index])
                        {
                            memcpy(static_cast<char*>(ctx->buffer_data[out_buffer_index]),
                                   static_cast<char*>(ctx->buffer_data[arg2_buffer_index]),
                                   arg2_size);
                        }
                        cpu::mkldnn_utils::set_memory_ptr(
                            ctx, deps[0], ctx->buffer_data[arg0_buffer_index]);
                        cpu::mkldnn_utils::set_memory_ptr(
                            ctx, deps[1], ctx->buffer_data[arg1_buffer_index]);
                        cpu::mkldnn_utils::set_memory_ptr(
                            ctx, deps[2], ctx->buffer_data[out_buffer_index]);

                        cpu::mkldnn_utils::mkldnn_invoke_primitive(
                            ctx,
                            conv_index,
                            deps,
                            cpu::mkldnn_utils::OpType::CONVOLUTIONADD,
                            scratchpad_size);
                    };
                    functors.emplace_back(functor);
                }
                else
                {
                    throw ngraph_error("ConvolutionAdd is only supported with MKLDNN kernel.");
                }
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::ConvolutionBackpropData)
            {
                auto convolution = static_cast<const ngraph::op::ConvolutionBackpropData*>(node);

                auto& functors = external_function->get_functors();

                auto arg0_shape = args[0].get_shape();
                auto arg1_shape = args[1].get_shape();

                auto arg0_buffer_index = external_function->get_buffer_index(args[0].get_name());
                auto arg1_buffer_index = external_function->get_buffer_index(args[1].get_name());
                auto out_buffer_index = external_function->get_buffer_index(out[0].get_name());

                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto bwd_desc = mkldnn_emitter->get_convolution_backward_data_desc<
                        ngraph::op::ConvolutionBackpropData>(node);
                    auto fwd_desc = mkldnn_emitter->get_convolution_forward_desc_for_backward_op<
                        ngraph::op::ConvolutionBackpropData>(node);
                    size_t scratchpad_size =
                        QUERY_SCRATCHPAD_2ARGS(convolution_backward_data, fwd_desc, bwd_desc);

                    // ConvolutionBackpropData needs 4 primitives: weights, diff_dst, diff_src,
                    // and convolution_backward_data.
                    auto conv_index = mkldnn_emitter->reserve_primitive_space(4);
                    auto& deps = mkldnn_emitter->get_primitive_deps(conv_index);

                    auto functor = [&,
                                    bwd_desc,
                                    fwd_desc,
                                    conv_index,
                                    scratchpad_size,
                                    arg0_buffer_index,
                                    arg1_buffer_index,
                                    out_buffer_index](CPURuntimeContext* ctx,
                                                      CPUExecutionContext* /* ectx */) {
                        if (ctx->first_iteration)
                        {
                            mkldnn_emitter->build_convolution_backward_data(
                                ctx->mkldnn_memories,
                                ctx->mkldnn_primitives,
                                ctx->mkldnn_scratchpad_mds,
                                bwd_desc,
                                fwd_desc,
                                deps,
                                conv_index);
                        }
                        cpu::mkldnn_utils::set_memory_ptr(
                            ctx, deps[0], ctx->buffer_data[arg0_buffer_index]);
                        cpu::mkldnn_utils::set_memory_ptr(
                            ctx, deps[1], ctx->buffer_data[arg1_buffer_index]);
                        cpu::mkldnn_utils::set_memory_ptr(
                            ctx, deps[2], ctx->buffer_data[out_buffer_index]);

                        cpu::mkldnn_utils::mkldnn_invoke_primitive(
                            ctx,
                            conv_index,
                            deps,
                            cpu::mkldnn_utils::OpType::CONVOLUTIONBACKPROPDATA,
                            scratchpad_size);
                    };
                    functors.emplace_back(functor);
                }
                else
                {
                    std::function<decltype(runtime::cpu::kernel::convolution_backprop_in<float>)>
                        kernel;

                    SELECT_KERNEL(kernel,
                                  out[0].get_element_type(),
                                  runtime::cpu::kernel::convolution_backprop_in)
                    auto& in_shape = convolution->get_data_batch_shape();
                    auto data_dilation_strides = convolution->get_data_dilation_strides_forward();
                    auto window_dilation_strides =
                        convolution->get_window_dilation_strides_forward();
                    auto padding_below = convolution->get_padding_below_forward();
                    auto padding_above = convolution->get_padding_above_forward();
                    auto window_movement_strides =
                        convolution->get_window_movement_strides_forward();
                    auto backward_delta_out_pad_below =
                        convolution->compute_backward_delta_out_pad_below();
                    auto backward_delta_out_pad_above =
                        convolution->compute_backward_delta_out_pad_above();

                    auto functor = [&,
                                    kernel,
                                    arg0_shape,
                                    arg1_shape,
                                    in_shape,
                                    data_dilation_strides,
                                    window_dilation_strides,
                                    backward_delta_out_pad_below,
                                    backward_delta_out_pad_above,
                                    window_movement_strides,
                                    arg0_buffer_index,
                                    arg1_buffer_index,
                                    out_buffer_index](CPURuntimeContext* ctx,
                                                      CPUExecutionContext* /* ectx */) {
                        kernel(ctx->buffer_data[arg1_buffer_index],
                               ctx->buffer_data[arg0_buffer_index],
                               ctx->buffer_data[out_buffer_index],
                               arg1_shape,
                               arg0_shape,
                               in_shape,
                               data_dilation_strides,
                               window_dilation_strides,
                               backward_delta_out_pad_below,
                               backward_delta_out_pad_above,
                               window_movement_strides);
                    };
                    functors.emplace_back(functor);
                }
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::ConvolutionBackpropFilters)
            {
                auto convolution = static_cast<const ngraph::op::ConvolutionBackpropFilters*>(node);

                auto& functors = external_function->get_functors();

                auto arg0_shape = args[0].get_shape();
                auto arg1_shape = args[1].get_shape();

                auto arg0_buffer_index = external_function->get_buffer_index(args[0].get_name());
                auto arg1_buffer_index = external_function->get_buffer_index(args[1].get_name());
                auto out_buffer_index = external_function->get_buffer_index(out[0].get_name());

                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto bwd_desc = mkldnn_emitter->get_convolution_backward_weights_desc<
                        ngraph::op::ConvolutionBackpropFilters>(node);
                    auto fwd_desc = mkldnn_emitter->get_convolution_forward_desc_for_backward_op<
                        ngraph::op::ConvolutionBackpropFilters>(node);
                    size_t scratchpad_size =
                        QUERY_SCRATCHPAD_2ARGS(convolution_backward_weights, fwd_desc, bwd_desc);

                    // ConvolutionBackpropFilter needs 4 primitives: src, diff_dst, diff_weights,
                    // and convolution_backward_weights.
                    auto conv_index = mkldnn_emitter->reserve_primitive_space(4);
                    auto& deps = mkldnn_emitter->get_primitive_deps(conv_index);

                    auto functor = [&,
                                    bwd_desc,
                                    fwd_desc,
                                    conv_index,
                                    scratchpad_size,
                                    arg0_buffer_index,
                                    arg1_buffer_index,
                                    out_buffer_index](CPURuntimeContext* ctx,
                                                      CPUExecutionContext* /* ectx */) {
                        if (ctx->first_iteration)
                        {
                            mkldnn_emitter->build_convolution_backward_weights(
                                ctx->mkldnn_memories,
                                ctx->mkldnn_primitives,
                                ctx->mkldnn_scratchpad_mds,
                                bwd_desc,
                                fwd_desc,
                                deps,
                                conv_index);
                        }
                        cpu::mkldnn_utils::set_memory_ptr(
                            ctx, deps[0], ctx->buffer_data[arg0_buffer_index]);
                        cpu::mkldnn_utils::set_memory_ptr(
                            ctx, deps[1], ctx->buffer_data[arg1_buffer_index]);
                        cpu::mkldnn_utils::set_memory_ptr(
                            ctx, deps[2], ctx->buffer_data[out_buffer_index]);

                        cpu::mkldnn_utils::mkldnn_invoke_primitive(
                            ctx,
                            conv_index,
                            deps,
                            cpu::mkldnn_utils::OpType::CONVOLUTIONBACKPROPWEIGHTS,
                            scratchpad_size);
                    };
                    functors.emplace_back(functor);
                }
                else
                {
                    std::function<decltype(
                        runtime::cpu::kernel::convolution_backprop_filter<float>)>
                        kernel;

                    SELECT_KERNEL(kernel,
                                  out[0].get_element_type(),
                                  runtime::cpu::kernel::convolution_backprop_filter)

                    auto& filters_shape = convolution->get_filters_shape();
                    auto window_dilation_strides =
                        convolution->get_window_dilation_strides_forward();
                    auto window_movement_strides =
                        convolution->get_window_movement_strides_forward();
                    auto padding_below = convolution->get_padding_below_forward();
                    auto padding_above = convolution->get_padding_above_forward();
                    auto data_dilation_strides = convolution->get_data_dilation_strides_forward();
                    CoordinateDiff backward_in_pad_above =
                        convolution->compute_backward_in_pad_above();

                    auto functor = [&,
                                    kernel,
                                    arg0_shape,
                                    arg1_shape,
                                    filters_shape,
                                    window_dilation_strides,
                                    window_movement_strides,
                                    padding_below,
                                    backward_in_pad_above,
                                    data_dilation_strides,
                                    arg0_buffer_index,
                                    arg1_buffer_index,
                                    out_buffer_index](CPURuntimeContext* ctx,
                                                      CPUExecutionContext* /* ectx */) {
                        kernel(ctx->buffer_data[arg0_buffer_index],
                               ctx->buffer_data[arg1_buffer_index],
                               ctx->buffer_data[out_buffer_index],
                               arg0_shape,
                               arg1_shape,
                               filters_shape,
                               window_movement_strides,
                               window_dilation_strides,
                               padding_below,
                               backward_in_pad_above,
                               data_dilation_strides);
                    };
                    functors.emplace_back(functor);
                }
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::ConvolutionBiasBackpropFiltersBias)
            {
                auto& functors = external_function->get_functors();

                auto arg0_buffer_index = external_function->get_buffer_index(args[0].get_name());
                auto arg1_buffer_index = external_function->get_buffer_index(args[1].get_name());
                auto out0_buffer_index = external_function->get_buffer_index(out[0].get_name());
                auto out1_buffer_index = external_function->get_buffer_index(out[1].get_name());

                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto bwd_desc = mkldnn_emitter->get_convolution_backward_weights_desc<
                        ngraph::op::ConvolutionBiasBackpropFiltersBias>(node);
                    auto fwd_desc = mkldnn_emitter->get_convolution_forward_desc_for_backward_op<
                        ngraph::op::ConvolutionBiasBackpropFiltersBias>(node);
                    size_t scratchpad_size =
                        QUERY_SCRATCHPAD_2ARGS(convolution_backward_weights, fwd_desc, bwd_desc);

                    // ConvolutionBackpropFiltersBias needs 5 primitives: src, diff_dst,
                    // diff_weights, diff_bias, and convolution_backward_weights.
                    auto conv_index = mkldnn_emitter->reserve_primitive_space(5);
                    auto& deps = mkldnn_emitter->get_primitive_deps(conv_index);

                    auto functor = [&,
                                    bwd_desc,
                                    fwd_desc,
                                    conv_index,
                                    scratchpad_size,
                                    arg0_buffer_index,
                                    arg1_buffer_index,
                                    out0_buffer_index,
                                    out1_buffer_index](CPURuntimeContext* ctx,
                                                       CPUExecutionContext* /* ectx */) {
                        if (ctx->first_iteration)
                        {
                            mkldnn_emitter->build_convolution_backward_weights_bias(
                                ctx->mkldnn_memories,
                                ctx->mkldnn_primitives,
                                ctx->mkldnn_scratchpad_mds,
                                bwd_desc,
                                fwd_desc,
                                deps,
                                conv_index);
                        }
                        cpu::mkldnn_utils::set_memory_ptr(
                            ctx, deps[0], ctx->buffer_data[arg0_buffer_index]);
                        cpu::mkldnn_utils::set_memory_ptr(
                            ctx, deps[1], ctx->buffer_data[arg1_buffer_index]);
                        cpu::mkldnn_utils::set_memory_ptr(
                            ctx, deps[2], ctx->buffer_data[out0_buffer_index]);
                        cpu::mkldnn_utils::set_memory_ptr(
                            ctx, deps[3], ctx->buffer_data[out1_buffer_index]);

                        cpu::mkldnn_utils::mkldnn_invoke_primitive(
                            ctx,
                            conv_index,
                            deps,
                            cpu::mkldnn_utils::OpType::CONVOLUTIONBACKPROPWEIGHTSBIAS,
                            scratchpad_size);
                    };
                    functors.emplace_back(functor);
                }
                else
                {
                    throw ngraph_error(
                        "ConvolutionBiasBackpropFiltersBias is only supported with MKLDNN kernel.");
                }
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::GroupConvolution)
            {
                auto& functors = external_function->get_functors();

                auto arg0_buffer_index = external_function->get_buffer_index(args[0].get_name());
                auto arg1_buffer_index = external_function->get_buffer_index(args[1].get_name());
                auto out_buffer_index = external_function->get_buffer_index(out[0].get_name());

                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto conv_desc =
                        mkldnn_emitter->get_convolution_forward_desc<ngraph::op::GroupConvolution>(
                            node);
                    auto conv_attr =
                        mkldnn_emitter->get_convolution_forward_attr<ngraph::op::GroupConvolution>(
                            node);
                    size_t scratchpad_size =
                        QUERY_SCRATCHPAD_2ARGS(convolution_forward, conv_desc, conv_attr);

                    size_t conv_index = mkldnn_emitter->convolution_forward_init();
                    auto& deps = mkldnn_emitter->get_primitive_deps(conv_index);

                    auto functor = [&,
                                    conv_desc,
                                    conv_attr,
                                    conv_index,
                                    scratchpad_size,
                                    arg0_buffer_index,
                                    arg1_buffer_index,
                                    out_buffer_index](CPURuntimeContext* ctx,
                                                      CPUExecutionContext* /* ectx */) {
                        if (ctx->first_iteration)
                        {
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

                        // group convolution
                        cpu::mkldnn_utils::set_memory_ptr(
                            ctx, deps[0], ctx->buffer_data[arg0_buffer_index]);
                        cpu::mkldnn_utils::set_memory_ptr(
                            ctx, deps[1], ctx->buffer_data[arg1_buffer_index]);
                        cpu::mkldnn_utils::set_memory_ptr(
                            ctx, deps[2], ctx->buffer_data[out_buffer_index]);

                        cpu::mkldnn_utils::mkldnn_invoke_primitive(
                            ctx,
                            conv_index,
                            deps,
                            cpu::mkldnn_utils::OpType::GROUPCONVOLUTION,
                            scratchpad_size);
                    };
                    functors.emplace_back(functor);
                }
                else
                {
                    throw ngraph_error("unsupported parameters for GroupConvolution");
                }
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::GroupConvolutionBias)
            {
                auto& functors = external_function->get_functors();

                auto arg0_buffer_index = external_function->get_buffer_index(args[0].get_name());
                auto arg1_buffer_index = external_function->get_buffer_index(args[1].get_name());
                auto arg2_buffer_index = external_function->get_buffer_index(args[2].get_name());
                auto out_buffer_index = external_function->get_buffer_index(out[0].get_name());

                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto conv_desc =
                        mkldnn_emitter
                            ->get_convolution_forward_desc<ngraph::op::GroupConvolutionBias>(node);
                    auto conv_attr =
                        mkldnn_emitter
                            ->get_convolution_forward_attr<ngraph::op::GroupConvolutionBias>(node);
                    size_t scratchpad_size =
                        QUERY_SCRATCHPAD_2ARGS(convolution_forward, conv_desc, conv_attr);

                    size_t conv_index = mkldnn_emitter->convolution_forward_init(true);
                    auto& deps = mkldnn_emitter->get_primitive_deps(conv_index);

                    auto functor = [&,
                                    conv_desc,
                                    conv_attr,
                                    conv_index,
                                    scratchpad_size,
                                    arg0_buffer_index,
                                    arg1_buffer_index,
                                    arg2_buffer_index,
                                    out_buffer_index](CPURuntimeContext* ctx,
                                                      CPUExecutionContext* /* ectx */) {
                        if (ctx->first_iteration)
                        {
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
                            ctx, deps[3], ctx->buffer_data[out_buffer_index]);

                        cpu::mkldnn_utils::mkldnn_invoke_primitive(
                            ctx,
                            conv_index,
                            deps,
                            cpu::mkldnn_utils::OpType::GROUPCONVOLUTIONBIAS,
                            scratchpad_size);
                    };
                    functors.emplace_back(functor);
                }
                else
                {
                    throw ngraph_error("unsupported parameters for GroupConvolutionBias");
                }
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::DeconvolutionBias)
            {
                auto& functors = external_function->get_functors();

                auto arg0_shape = args[0].get_shape();
                auto arg1_shape = args[1].get_shape();
                auto arg2_shape = args[2].get_shape();
                auto result_shape = out[0].get_shape();

                auto arg0_buffer_index = external_function->get_buffer_index(args[0].get_name());
                auto arg1_buffer_index = external_function->get_buffer_index(args[1].get_name());
                auto arg2_buffer_index = external_function->get_buffer_index(args[2].get_name());
                auto out_buffer_index = external_function->get_buffer_index(out[0].get_name());

                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto deconvbias_desc =
                        mkldnn_emitter
                            ->get_deconvolutionbias_forward_data<ngraph::op::DeconvolutionBias>(
                                node);
                    auto weights_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    size_t scratchpad_size =
                        QUERY_SCRATCHPAD(deconvolution_forward, deconvbias_desc);

                    // DeconvolutionBias needs 5 primitives: weights, delta, bias, result,
                    // and deconvolutionbias.
                    auto conv_index = mkldnn_emitter->reserve_primitive_space(5);
                    auto& deps = mkldnn_emitter->get_primitive_deps(conv_index);

                    auto functor = [&,
                                    deconvbias_desc,
                                    conv_index,
                                    scratchpad_size,
                                    weights_desc,
                                    arg0_buffer_index,
                                    arg1_buffer_index,
                                    arg2_buffer_index,
                                    out_buffer_index](CPURuntimeContext* ctx,
                                                      CPUExecutionContext* /* ectx */) {
                        if (ctx->first_iteration)
                        {
                            mkldnn_emitter->build_deconvolutionbias_forward(
                                ctx->mkldnn_memories,
                                ctx->mkldnn_primitives,
                                ctx->mkldnn_scratchpad_mds,
                                deconvbias_desc,
                                deps,
                                conv_index,
                                weights_desc);
                        }
                        cpu::mkldnn_utils::set_memory_ptr(
                            ctx, deps[0], ctx->buffer_data[arg0_buffer_index]);
                        cpu::mkldnn_utils::set_memory_ptr(
                            ctx, deps[1], ctx->buffer_data[arg1_buffer_index]);
                        cpu::mkldnn_utils::set_memory_ptr(
                            ctx, deps[2], ctx->buffer_data[arg2_buffer_index]);
                        cpu::mkldnn_utils::set_memory_ptr(
                            ctx, deps[3], ctx->buffer_data[out_buffer_index]);

                        cpu::mkldnn_utils::mkldnn_invoke_primitive(
                            ctx,
                            conv_index,
                            deps,
                            cpu::mkldnn_utils::OpType::DECONVOLUTIONBIAS,
                            scratchpad_size);
                    };
                    functors.emplace_back(functor);
                }
                else
                {
                    throw ngraph_error("DeconvolutionBias is only supported with MKLDNN kernel");
                }
            }

            void register_builders_convolution_cpp()
            {
                REGISTER_OP_BUILDER(Convolution);
                REGISTER_OP_BUILDER(ConvolutionRelu);
                REGISTER_OP_BUILDER(ConvolutionBias);
                REGISTER_OP_BUILDER(ConvolutionBiasAdd);
                REGISTER_OP_BUILDER(ConvolutionBackpropData);
                REGISTER_OP_BUILDER(ConvolutionBackpropFilters);
                REGISTER_OP_BUILDER(ConvolutionBiasBackpropFiltersBias);
                REGISTER_OP_BUILDER(GroupConvolution);
                REGISTER_OP_BUILDER(ConvolutionAdd);
                REGISTER_OP_BUILDER(GroupConvolutionBias);
                REGISTER_OP_BUILDER(DeconvolutionBias);
            }
        } // namespace cpu
    }     // namespace runtime
} // namespace ngraph
