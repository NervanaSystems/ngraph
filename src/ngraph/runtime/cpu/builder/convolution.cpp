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

#include "ngraph/runtime/cpu/kernel/convolution.hpp"
#include "ngraph/op/conv_fused.hpp"
#include "ngraph/op/group_conv.hpp"
#include "ngraph/runtime/cpu/cpu_builder.hpp"
#include "ngraph/runtime/cpu/dnnl_invoke.hpp"
#include "ngraph/runtime/cpu/dnnl_utils.hpp"
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
            void Builder::BUILDER_DECL(ngraph::op::v0::Convolution)
            {
                auto convolution = static_cast<const ngraph::op::v0::Convolution*>(node);

                auto& functors = external_function->get_functors();

                auto arg0_shape = args[0].get_shape();
                auto arg1_shape = args[1].get_shape();
                auto result_shape = out[0].get_shape();

                auto arg0_buffer_index = external_function->get_buffer_index(args[0].get_name());
                auto arg1_buffer_index = external_function->get_buffer_index(args[1].get_name());
                auto out_buffer_index = external_function->get_buffer_index(out[0].get_name());

                if (runtime::cpu::dnnl_utils::use_dnnl_kernel(node))
                {
                    auto& dnnl_emitter = external_function->get_dnnl_emitter();
                    auto conv_desc =
                        dnnl_emitter->get_convolution_forward_desc<ngraph::op::v0::Convolution>(
                            node);
                    auto conv_attr =
                        dnnl_emitter->get_convolution_forward_attr<ngraph::op::v0::Convolution>(
                            node);
                    size_t scratchpad_size =
                        QUERY_SCRATCHPAD_2ARGS(convolution_forward, conv_desc, conv_attr);

                    size_t conv_index = dnnl_emitter->convolution_forward_init();
                    auto& deps = dnnl_emitter->get_primitive_deps(conv_index);

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
                            dnnl_emitter->build_convolution_forward<false>(
                                ctx->dnnl_memories,
                                ctx->dnnl_primitives,
                                ctx->dnnl_scratchpad_mds,
                                conv_desc,
                                conv_attr,
                                executor::global_cpu_engine,
                                deps,
                                conv_index);
                        }
                        cpu::dnnl_utils::set_memory_ptr(
                            ctx, deps[0], ctx->buffer_data[arg0_buffer_index]);
                        cpu::dnnl_utils::set_memory_ptr(
                            ctx, deps[1], ctx->buffer_data[arg1_buffer_index]);
                        cpu::dnnl_utils::set_memory_ptr(
                            ctx, deps[2], ctx->buffer_data[out_buffer_index]);

                        cpu::dnnl_utils::dnnl_invoke_primitive(ctx,
                                                               conv_index,
                                                               deps,
                                                               cpu::dnnl_utils::OpType::CONVOLUTION,
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

                if (runtime::cpu::dnnl_utils::use_dnnl_kernel(node))
                {
                    auto& dnnl_emitter = external_function->get_dnnl_emitter();
                    auto conv_desc =
                        dnnl_emitter->get_convolution_forward_desc<ngraph::op::ConvolutionRelu>(
                            node);
                    auto conv_attr =
                        dnnl_emitter->get_convolution_forward_attr<ngraph::op::ConvolutionRelu>(
                            node);
                    size_t scratchpad_size =
                        QUERY_SCRATCHPAD_2ARGS(convolution_forward, conv_desc, conv_attr);

                    size_t conv_index = dnnl_emitter->convolution_forward_init();
                    auto& deps = dnnl_emitter->get_primitive_deps(conv_index);

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
                            dnnl_emitter->build_convolution_forward<false>(
                                ctx->dnnl_memories,
                                ctx->dnnl_primitives,
                                ctx->dnnl_scratchpad_mds,
                                conv_desc,
                                conv_attr,
                                executor::global_cpu_engine,
                                deps,
                                conv_index);
                        }
                        cpu::dnnl_utils::set_memory_ptr(
                            ctx, deps[0], ctx->buffer_data[arg0_buffer_index]);
                        cpu::dnnl_utils::set_memory_ptr(
                            ctx, deps[1], ctx->buffer_data[arg1_buffer_index]);
                        cpu::dnnl_utils::set_memory_ptr(
                            ctx, deps[2], ctx->buffer_data[out_buffer_index]);

                        cpu::dnnl_utils::dnnl_invoke_primitive(
                            ctx,
                            conv_index,
                            deps,
                            cpu::dnnl_utils::OpType::CONVOLUTIONRELU,
                            scratchpad_size);
                    };
                    functors.emplace_back(functor);
                }
                else
                {
                    throw ngraph_error("ConvolutionRelu is only supported with DNNL kernel.");
                }
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::v0::ConvolutionBias)
            {
                auto& functors = external_function->get_functors();

                auto arg0_buffer_index = external_function->get_buffer_index(args[0].get_name());
                auto arg1_buffer_index = external_function->get_buffer_index(args[1].get_name());
                auto arg2_buffer_index = external_function->get_buffer_index(args[2].get_name());
                auto out_buffer_index = external_function->get_buffer_index(out[0].get_name());

                if (runtime::cpu::dnnl_utils::use_dnnl_kernel(node))
                {
                    auto& dnnl_emitter = external_function->get_dnnl_emitter();
                    auto conv_desc =
                        dnnl_emitter->get_convolution_forward_desc<ngraph::op::v0::ConvolutionBias>(
                            node);
                    auto conv_attr =
                        dnnl_emitter->get_convolution_forward_attr<ngraph::op::v0::ConvolutionBias>(
                            node);
                    size_t scratchpad_size =
                        QUERY_SCRATCHPAD_2ARGS(convolution_forward, conv_desc, conv_attr);

                    size_t conv_index = dnnl_emitter->convolution_forward_init(true);
                    auto& deps = dnnl_emitter->get_primitive_deps(conv_index);

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
                            dnnl_emitter->build_convolution_forward<true>(
                                ctx->dnnl_memories,
                                ctx->dnnl_primitives,
                                ctx->dnnl_scratchpad_mds,
                                conv_desc,
                                conv_attr,
                                executor::global_cpu_engine,
                                deps,
                                conv_index);
                        }
                        cpu::dnnl_utils::set_memory_ptr(
                            ctx, deps[0], ctx->buffer_data[arg0_buffer_index]);
                        cpu::dnnl_utils::set_memory_ptr(
                            ctx, deps[1], ctx->buffer_data[arg1_buffer_index]);
                        cpu::dnnl_utils::set_memory_ptr(
                            ctx, deps[2], ctx->buffer_data[arg2_buffer_index]);
                        cpu::dnnl_utils::set_memory_ptr(
                            ctx, deps[3], ctx->buffer_data[out_buffer_index]);

                        cpu::dnnl_utils::dnnl_invoke_primitive(
                            ctx,
                            conv_index,
                            deps,
                            cpu::dnnl_utils::OpType::CONVOLUTIONBIAS,
                            scratchpad_size);
                    };
                    functors.emplace_back(functor);
                }
                else
                {
                    throw ngraph_error("ConvolutionBias is only supported with DNNL kernel.");
                }
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::v0::ConvolutionBiasAdd)
            {
                auto& functors = external_function->get_functors();

                auto arg0_buffer_index = external_function->get_buffer_index(args[0].get_name());
                auto arg1_buffer_index = external_function->get_buffer_index(args[1].get_name());
                auto arg2_buffer_index = external_function->get_buffer_index(args[2].get_name());
                auto arg3_buffer_index = external_function->get_buffer_index(args[3].get_name());
                auto out_buffer_index = external_function->get_buffer_index(out[0].get_name());
                size_t arg3_size = node->get_input_tensor(3).size();

                if (runtime::cpu::dnnl_utils::use_dnnl_kernel(node))
                {
                    auto& dnnl_emitter = external_function->get_dnnl_emitter();
                    auto conv_desc =
                        dnnl_emitter
                            ->get_convolution_forward_desc<ngraph::op::v0::ConvolutionBiasAdd>(
                                node);
                    auto conv_attr =
                        dnnl_emitter
                            ->get_convolution_forward_attr<ngraph::op::v0::ConvolutionBiasAdd>(
                                node);
                    size_t scratchpad_size =
                        QUERY_SCRATCHPAD_2ARGS(convolution_forward, conv_desc, conv_attr);

                    size_t conv_index = dnnl_emitter->convolution_forward_init(true);
                    auto& deps = dnnl_emitter->get_primitive_deps(conv_index);

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
                            dnnl_emitter->build_convolution_forward<true>(
                                ctx->dnnl_memories,
                                ctx->dnnl_primitives,
                                ctx->dnnl_scratchpad_mds,
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
                        cpu::dnnl_utils::set_memory_ptr(
                            ctx, deps[0], ctx->buffer_data[arg0_buffer_index]);
                        cpu::dnnl_utils::set_memory_ptr(
                            ctx, deps[1], ctx->buffer_data[arg1_buffer_index]);
                        cpu::dnnl_utils::set_memory_ptr(
                            ctx, deps[2], ctx->buffer_data[arg2_buffer_index]);
                        cpu::dnnl_utils::set_memory_ptr(
                            ctx, deps[3], ctx->buffer_data[out_buffer_index]);

                        cpu::dnnl_utils::dnnl_invoke_primitive(
                            ctx,
                            conv_index,
                            deps,
                            cpu::dnnl_utils::OpType::CONVOLUTIONBIASADD,
                            scratchpad_size);
                    };
                    functors.emplace_back(functor);
                }
                else
                {
                    throw ngraph_error("ConvolutionBiasAdd is only supported with DNNL kernel.");
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
                size_t arg2_size = node->get_input_tensor(2).size();

                if (runtime::cpu::dnnl_utils::use_dnnl_kernel(node))
                {
                    auto& dnnl_emitter = external_function->get_dnnl_emitter();
                    auto conv_desc =
                        dnnl_emitter->get_convolution_forward_desc<ngraph::op::ConvolutionAdd>(
                            node);
                    auto conv_attr =
                        dnnl_emitter->get_convolution_forward_attr<ngraph::op::ConvolutionAdd>(
                            node);
                    size_t scratchpad_size =
                        QUERY_SCRATCHPAD_2ARGS(convolution_forward, conv_desc, conv_attr);

                    size_t conv_index = dnnl_emitter->convolution_forward_init(false);
                    auto& deps = dnnl_emitter->get_primitive_deps(conv_index);

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
                            dnnl_emitter->build_convolution_forward<false>(
                                ctx->dnnl_memories,
                                ctx->dnnl_primitives,
                                ctx->dnnl_scratchpad_mds,
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
                        cpu::dnnl_utils::set_memory_ptr(
                            ctx, deps[0], ctx->buffer_data[arg0_buffer_index]);
                        cpu::dnnl_utils::set_memory_ptr(
                            ctx, deps[1], ctx->buffer_data[arg1_buffer_index]);
                        cpu::dnnl_utils::set_memory_ptr(
                            ctx, deps[2], ctx->buffer_data[out_buffer_index]);

                        cpu::dnnl_utils::dnnl_invoke_primitive(
                            ctx,
                            conv_index,
                            deps,
                            cpu::dnnl_utils::OpType::CONVOLUTIONADD,
                            scratchpad_size);
                    };
                    functors.emplace_back(functor);
                }
                else
                {
                    throw ngraph_error("ConvolutionAdd is only supported with DNNL kernel.");
                }
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::v0::ConvolutionBackpropData)
            {
                auto convolution =
                    static_cast<const ngraph::op::v0::ConvolutionBackpropData*>(node);

                auto& functors = external_function->get_functors();

                auto arg0_shape = args[0].get_shape();
                auto arg1_shape = args[1].get_shape();

                auto arg0_buffer_index = external_function->get_buffer_index(args[0].get_name());
                auto arg1_buffer_index = external_function->get_buffer_index(args[1].get_name());
                auto out_buffer_index = external_function->get_buffer_index(out[0].get_name());

                if (runtime::cpu::dnnl_utils::use_dnnl_kernel(node))
                {
                    auto& dnnl_emitter = external_function->get_dnnl_emitter();
                    auto bwd_desc = dnnl_emitter->get_convolution_backward_data_desc<
                        ngraph::op::v0::ConvolutionBackpropData>(node);
                    auto fwd_desc = dnnl_emitter->get_convolution_forward_desc_for_backward_op<
                        ngraph::op::v0::ConvolutionBackpropData>(node);
                    size_t scratchpad_size =
                        QUERY_SCRATCHPAD_2ARGS(convolution_backward_data, fwd_desc, bwd_desc);

                    // ConvolutionBackpropData needs 4 primitives: weights, diff_dst, diff_src,
                    // and convolution_backward_data.
                    auto conv_index = dnnl_emitter->reserve_primitive_space(4);
                    auto& deps = dnnl_emitter->get_primitive_deps(conv_index);

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
                            dnnl_emitter->build_convolution_backward_data(ctx->dnnl_memories,
                                                                          ctx->dnnl_primitives,
                                                                          ctx->dnnl_scratchpad_mds,
                                                                          bwd_desc,
                                                                          fwd_desc,
                                                                          deps,
                                                                          conv_index);
                        }
                        cpu::dnnl_utils::set_memory_ptr(
                            ctx, deps[0], ctx->buffer_data[arg0_buffer_index]);
                        cpu::dnnl_utils::set_memory_ptr(
                            ctx, deps[1], ctx->buffer_data[arg1_buffer_index]);
                        cpu::dnnl_utils::set_memory_ptr(
                            ctx, deps[2], ctx->buffer_data[out_buffer_index]);

                        cpu::dnnl_utils::dnnl_invoke_primitive(
                            ctx,
                            conv_index,
                            deps,
                            cpu::dnnl_utils::OpType::CONVOLUTIONBACKPROPDATA,
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
            void Builder::BUILDER_DECL(ngraph::op::v0::ConvolutionBackpropFilters)
            {
                auto convolution =
                    static_cast<const ngraph::op::v0::ConvolutionBackpropFilters*>(node);

                auto& functors = external_function->get_functors();

                auto arg0_shape = args[0].get_shape();
                auto arg1_shape = args[1].get_shape();

                auto arg0_buffer_index = external_function->get_buffer_index(args[0].get_name());
                auto arg1_buffer_index = external_function->get_buffer_index(args[1].get_name());
                auto out_buffer_index = external_function->get_buffer_index(out[0].get_name());

                if (runtime::cpu::dnnl_utils::use_dnnl_kernel(node))
                {
                    auto& dnnl_emitter = external_function->get_dnnl_emitter();
                    auto bwd_desc = dnnl_emitter->get_convolution_backward_weights_desc<
                        ngraph::op::v0::ConvolutionBackpropFilters>(node);
                    auto fwd_desc = dnnl_emitter->get_convolution_forward_desc_for_backward_op<
                        ngraph::op::v0::ConvolutionBackpropFilters>(node);
                    size_t scratchpad_size =
                        QUERY_SCRATCHPAD_2ARGS(convolution_backward_weights, fwd_desc, bwd_desc);

                    // ConvolutionBackpropFilter needs 4 primitives: src, diff_dst, diff_weights,
                    // and convolution_backward_weights.
                    auto conv_index = dnnl_emitter->reserve_primitive_space(4);
                    auto& deps = dnnl_emitter->get_primitive_deps(conv_index);

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
                            dnnl_emitter->build_convolution_backward_weights(
                                ctx->dnnl_memories,
                                ctx->dnnl_primitives,
                                ctx->dnnl_scratchpad_mds,
                                bwd_desc,
                                fwd_desc,
                                deps,
                                conv_index);
                        }
                        cpu::dnnl_utils::set_memory_ptr(
                            ctx, deps[0], ctx->buffer_data[arg0_buffer_index]);
                        cpu::dnnl_utils::set_memory_ptr(
                            ctx, deps[1], ctx->buffer_data[arg1_buffer_index]);
                        cpu::dnnl_utils::set_memory_ptr(
                            ctx, deps[2], ctx->buffer_data[out_buffer_index]);

                        cpu::dnnl_utils::dnnl_invoke_primitive(
                            ctx,
                            conv_index,
                            deps,
                            cpu::dnnl_utils::OpType::CONVOLUTIONBACKPROPWEIGHTS,
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
            void Builder::BUILDER_DECL(ngraph::op::v0::ConvolutionBiasBackpropFiltersBias)
            {
                auto& functors = external_function->get_functors();

                auto arg0_buffer_index = external_function->get_buffer_index(args[0].get_name());
                auto arg1_buffer_index = external_function->get_buffer_index(args[1].get_name());
                auto out0_buffer_index = external_function->get_buffer_index(out[0].get_name());
                auto out1_buffer_index = external_function->get_buffer_index(out[1].get_name());

                if (runtime::cpu::dnnl_utils::use_dnnl_kernel(node))
                {
                    auto& dnnl_emitter = external_function->get_dnnl_emitter();
                    auto bwd_desc = dnnl_emitter->get_convolution_backward_weights_desc<
                        ngraph::op::v0::ConvolutionBiasBackpropFiltersBias>(node);
                    auto fwd_desc = dnnl_emitter->get_convolution_forward_desc_for_backward_op<
                        ngraph::op::v0::ConvolutionBiasBackpropFiltersBias>(node);
                    size_t scratchpad_size =
                        QUERY_SCRATCHPAD_2ARGS(convolution_backward_weights, fwd_desc, bwd_desc);

                    // ConvolutionBackpropFiltersBias needs 5 primitives: src, diff_dst,
                    // diff_weights, diff_bias, and convolution_backward_weights.
                    auto conv_index = dnnl_emitter->reserve_primitive_space(5);
                    auto& deps = dnnl_emitter->get_primitive_deps(conv_index);

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
                            dnnl_emitter->build_convolution_backward_weights_bias(
                                ctx->dnnl_memories,
                                ctx->dnnl_primitives,
                                ctx->dnnl_scratchpad_mds,
                                bwd_desc,
                                fwd_desc,
                                deps,
                                conv_index);
                        }
                        cpu::dnnl_utils::set_memory_ptr(
                            ctx, deps[0], ctx->buffer_data[arg0_buffer_index]);
                        cpu::dnnl_utils::set_memory_ptr(
                            ctx, deps[1], ctx->buffer_data[arg1_buffer_index]);
                        cpu::dnnl_utils::set_memory_ptr(
                            ctx, deps[2], ctx->buffer_data[out0_buffer_index]);
                        cpu::dnnl_utils::set_memory_ptr(
                            ctx, deps[3], ctx->buffer_data[out1_buffer_index]);

                        cpu::dnnl_utils::dnnl_invoke_primitive(
                            ctx,
                            conv_index,
                            deps,
                            cpu::dnnl_utils::OpType::CONVOLUTIONBACKPROPWEIGHTSBIAS,
                            scratchpad_size);
                    };
                    functors.emplace_back(functor);
                }
                else
                {
                    throw ngraph_error(
                        "ConvolutionBiasBackpropFiltersBias is only supported with DNNL kernel.");
                }
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::v0::GroupConvolution)
            {
                auto& functors = external_function->get_functors();

                auto arg0_buffer_index = external_function->get_buffer_index(args[0].get_name());
                auto arg1_buffer_index = external_function->get_buffer_index(args[1].get_name());
                auto out_buffer_index = external_function->get_buffer_index(out[0].get_name());

                if (runtime::cpu::dnnl_utils::use_dnnl_kernel(node))
                {
                    auto& dnnl_emitter = external_function->get_dnnl_emitter();
                    auto conv_desc =
                        dnnl_emitter
                            ->get_convolution_forward_desc<ngraph::op::v0::GroupConvolution>(node);
                    auto conv_attr =
                        dnnl_emitter
                            ->get_convolution_forward_attr<ngraph::op::v0::GroupConvolution>(node);
                    size_t scratchpad_size =
                        QUERY_SCRATCHPAD_2ARGS(convolution_forward, conv_desc, conv_attr);

                    size_t conv_index = dnnl_emitter->convolution_forward_init();
                    auto& deps = dnnl_emitter->get_primitive_deps(conv_index);

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
                            dnnl_emitter->build_convolution_forward<false>(
                                ctx->dnnl_memories,
                                ctx->dnnl_primitives,
                                ctx->dnnl_scratchpad_mds,
                                conv_desc,
                                conv_attr,
                                executor::global_cpu_engine,
                                deps,
                                conv_index);
                        }

                        // group convolution
                        cpu::dnnl_utils::set_memory_ptr(
                            ctx, deps[0], ctx->buffer_data[arg0_buffer_index]);
                        cpu::dnnl_utils::set_memory_ptr(
                            ctx, deps[1], ctx->buffer_data[arg1_buffer_index]);
                        cpu::dnnl_utils::set_memory_ptr(
                            ctx, deps[2], ctx->buffer_data[out_buffer_index]);

                        cpu::dnnl_utils::dnnl_invoke_primitive(
                            ctx,
                            conv_index,
                            deps,
                            cpu::dnnl_utils::OpType::GROUPCONVOLUTION,
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

                if (runtime::cpu::dnnl_utils::use_dnnl_kernel(node))
                {
                    auto& dnnl_emitter = external_function->get_dnnl_emitter();
                    auto conv_desc =
                        dnnl_emitter
                            ->get_convolution_forward_desc<ngraph::op::GroupConvolutionBias>(node);
                    auto conv_attr =
                        dnnl_emitter
                            ->get_convolution_forward_attr<ngraph::op::GroupConvolutionBias>(node);
                    size_t scratchpad_size =
                        QUERY_SCRATCHPAD_2ARGS(convolution_forward, conv_desc, conv_attr);

                    size_t conv_index = dnnl_emitter->convolution_forward_init(true);
                    auto& deps = dnnl_emitter->get_primitive_deps(conv_index);

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
                            dnnl_emitter->build_convolution_forward<true>(
                                ctx->dnnl_memories,
                                ctx->dnnl_primitives,
                                ctx->dnnl_scratchpad_mds,
                                conv_desc,
                                conv_attr,
                                executor::global_cpu_engine,
                                deps,
                                conv_index);
                        }
                        cpu::dnnl_utils::set_memory_ptr(
                            ctx, deps[0], ctx->buffer_data[arg0_buffer_index]);
                        cpu::dnnl_utils::set_memory_ptr(
                            ctx, deps[1], ctx->buffer_data[arg1_buffer_index]);
                        cpu::dnnl_utils::set_memory_ptr(
                            ctx, deps[2], ctx->buffer_data[arg2_buffer_index]);
                        cpu::dnnl_utils::set_memory_ptr(
                            ctx, deps[3], ctx->buffer_data[out_buffer_index]);

                        cpu::dnnl_utils::dnnl_invoke_primitive(
                            ctx,
                            conv_index,
                            deps,
                            cpu::dnnl_utils::OpType::GROUPCONVOLUTIONBIAS,
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

                if (runtime::cpu::dnnl_utils::use_dnnl_kernel(node))
                {
                    auto& dnnl_emitter = external_function->get_dnnl_emitter();
                    auto deconvbias_desc =
                        dnnl_emitter
                            ->get_deconvolutionbias_forward_data<ngraph::op::DeconvolutionBias>(
                                node);
                    auto weights_desc = dnnl_utils::get_input_dnnl_md(node, 0);
                    size_t scratchpad_size =
                        QUERY_SCRATCHPAD(deconvolution_forward, deconvbias_desc);

                    // DeconvolutionBias needs 5 primitives: weights, delta, bias, result,
                    // and deconvolutionbias.
                    auto conv_index = dnnl_emitter->reserve_primitive_space(5);
                    auto& deps = dnnl_emitter->get_primitive_deps(conv_index);

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
                            dnnl_emitter->build_deconvolutionbias_forward(ctx->dnnl_memories,
                                                                          ctx->dnnl_primitives,
                                                                          ctx->dnnl_scratchpad_mds,
                                                                          deconvbias_desc,
                                                                          deps,
                                                                          conv_index,
                                                                          weights_desc);
                        }
                        cpu::dnnl_utils::set_memory_ptr(
                            ctx, deps[0], ctx->buffer_data[arg0_buffer_index]);
                        cpu::dnnl_utils::set_memory_ptr(
                            ctx, deps[1], ctx->buffer_data[arg1_buffer_index]);
                        cpu::dnnl_utils::set_memory_ptr(
                            ctx, deps[2], ctx->buffer_data[arg2_buffer_index]);
                        cpu::dnnl_utils::set_memory_ptr(
                            ctx, deps[3], ctx->buffer_data[out_buffer_index]);

                        cpu::dnnl_utils::dnnl_invoke_primitive(
                            ctx,
                            conv_index,
                            deps,
                            cpu::dnnl_utils::OpType::DECONVOLUTIONBIAS,
                            scratchpad_size);
                    };
                    functors.emplace_back(functor);
                }
                else
                {
                    throw ngraph_error("DeconvolutionBias is only supported with DNNL kernel");
                }
            }

            void register_builders_convolution_cpp()
            {
                REGISTER_OP_BUILDER(ngraph::op::v0::Convolution);
                REGISTER_OP_BUILDER(ngraph::op::ConvolutionRelu);
                REGISTER_OP_BUILDER(ngraph::op::v0::ConvolutionBias);
                REGISTER_OP_BUILDER(ngraph::op::v0::ConvolutionBiasAdd);
                REGISTER_OP_BUILDER(ngraph::op::v0::ConvolutionBackpropData);
                REGISTER_OP_BUILDER(ngraph::op::v0::ConvolutionBackpropFilters);
                REGISTER_OP_BUILDER(ngraph::op::v0::ConvolutionBiasBackpropFiltersBias);
                REGISTER_OP_BUILDER(ngraph::op::v0::GroupConvolution);
                REGISTER_OP_BUILDER(ngraph::op::ConvolutionAdd);
                REGISTER_OP_BUILDER(ngraph::op::GroupConvolutionBias);
                REGISTER_OP_BUILDER(ngraph::op::DeconvolutionBias);
            }
        }
    }
}
