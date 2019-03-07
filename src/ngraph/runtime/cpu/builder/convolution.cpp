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

#include "ngraph/op/convolution.hpp"
#include "ngraph/runtime/cpu/cpu_builder.hpp"
#include "ngraph/runtime/cpu/kernel/convolution.hpp"
#include "ngraph/runtime/cpu/mkldnn_invoke.hpp"
#include "ngraph/runtime/cpu/mkldnn_utils.hpp"
#include "ngraph/runtime/cpu/op/conv_add.hpp"
#include "ngraph/runtime/cpu/op/conv_bias.hpp"
#include "ngraph/runtime/cpu/op/conv_relu.hpp"
#include "ngraph/runtime/cpu/op/group_conv.hpp"
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

                auto& arg0_tensor = external_function->get_tensor_data(args[0].get_name());
                auto& arg1_tensor = external_function->get_tensor_data(args[1].get_name());
                auto& out_tensor = external_function->get_tensor_data(out[0].get_name());

                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto conv_desc =
                        mkldnn_emitter->get_convolution_forward_desc<ngraph::op::Convolution>(node);
                    auto conv_attr =
                        mkldnn_emitter->get_convolution_forward_attr<ngraph::op::Convolution>(node);
                    size_t conv_index = mkldnn_emitter->convolution_forward_init();
                    auto& deps = mkldnn_emitter->get_primitive_deps(conv_index);

                    auto functor = [&, conv_desc, conv_attr, conv_index](
                        CPURuntimeContext* ctx, CPUExecutionContext* ectx) {
                        if (ctx->first_iteration)
                        {
                            mkldnn_emitter->build_convolution_forward<false>(
                                conv_desc, conv_attr, executor::global_cpu_engine, conv_index);
                        }
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[0], arg0_tensor);
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[1], arg1_tensor);
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[2], out_tensor);
                        cpu::mkldnn_utils::mkldnn_invoke_primitive(ctx, conv_index);
                    };
                    functors.emplace_back(functor);
                }
                else
                {
                    std::function<decltype(runtime::cpu::kernel::convolution<float>)> kernel;

                    SELECT_KERNEL(
                        kernel, out[0].get_element_type(), runtime::cpu::kernel::convolution);

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
                                    data_dilation_strides](CPURuntimeContext* ctx,
                                                           CPUExecutionContext* ectx) {
                        kernel(arg0_tensor,
                               arg1_tensor,
                               out_tensor,
                               arg0_shape,
                               arg1_shape,
                               result_shape,
                               window_movement_strides,
                               window_dilation_strides,
                               padding_below,
                               padding_above,
                               data_dilation_strides,
                               0,
                               1,
                               1,
                               0,
                               0,
                               1,
                               false);
                    };
                    functors.emplace_back(functor);
                }
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::ConvolutionRelu)
            {
                auto& functors = external_function->get_functors();

                auto& arg0_tensor = external_function->get_tensor_data(args[0].get_name());
                auto& arg1_tensor = external_function->get_tensor_data(args[1].get_name());
                auto& out_tensor = external_function->get_tensor_data(out[0].get_name());

                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto conv_desc =
                        mkldnn_emitter->get_convolution_forward_desc<ngraph::op::ConvolutionRelu>(
                            node);
                    auto conv_attr =
                        mkldnn_emitter->get_convolution_forward_attr<ngraph::op::ConvolutionRelu>(
                            node);
                    size_t conv_index = mkldnn_emitter->convolution_forward_init();
                    auto& deps = mkldnn_emitter->get_primitive_deps(conv_index);

                    auto functor = [&, conv_desc, conv_attr, conv_index](
                        CPURuntimeContext* ctx, CPUExecutionContext* ectx) {
                        if (ctx->first_iteration)
                        {
                            mkldnn_emitter->build_convolution_forward<false>(
                                conv_desc, conv_attr, executor::global_cpu_engine, conv_index);
                        }
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[0], arg0_tensor);
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[1], arg1_tensor);
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[2], out_tensor);
                        cpu::mkldnn_utils::mkldnn_invoke_primitive(ctx, conv_index);
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

                auto& arg0_tensor = external_function->get_tensor_data(args[0].get_name());
                auto& arg1_tensor = external_function->get_tensor_data(args[1].get_name());
                auto& arg2_tensor = external_function->get_tensor_data(args[2].get_name());
                auto& out_tensor = external_function->get_tensor_data(out[0].get_name());

                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto conv_desc =
                        mkldnn_emitter->get_convolution_forward_desc<ngraph::op::ConvolutionBias>(
                            node);
                    auto conv_attr =
                        mkldnn_emitter->get_convolution_forward_attr<ngraph::op::ConvolutionBias>(
                            node);
                    size_t conv_index = mkldnn_emitter->convolution_forward_init(true);
                    auto& deps = mkldnn_emitter->get_primitive_deps(conv_index);

                    auto functor = [&, conv_desc, conv_attr, conv_index](
                        CPURuntimeContext* ctx, CPUExecutionContext* ectx) {
                        if (ctx->first_iteration)
                        {
                            mkldnn_emitter->build_convolution_forward<true>(
                                conv_desc, conv_attr, executor::global_cpu_engine, conv_index);
                        }
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[0], arg0_tensor);
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[1], arg1_tensor);
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[2], arg2_tensor);
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[3], out_tensor);
                        cpu::mkldnn_utils::mkldnn_invoke_primitive(ctx, conv_index);
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

                auto& arg0_tensor = external_function->get_tensor_data(args[0].get_name());
                auto& arg1_tensor = external_function->get_tensor_data(args[1].get_name());
                auto& arg2_tensor = external_function->get_tensor_data(args[2].get_name());
                auto& arg3_tensor = external_function->get_tensor_data(args[3].get_name());
                auto& out_tensor = external_function->get_tensor_data(out[0].get_name());
                size_t arg3_size = node->get_inputs()[3].get_tensor().size();

                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto conv_desc =
                        mkldnn_emitter
                            ->get_convolution_forward_desc<ngraph::op::ConvolutionBiasAdd>(node);
                    auto conv_attr =
                        mkldnn_emitter
                            ->get_convolution_forward_attr<ngraph::op::ConvolutionBiasAdd>(node);
                    size_t conv_index = mkldnn_emitter->convolution_forward_init(true);
                    auto& deps = mkldnn_emitter->get_primitive_deps(conv_index);

                    auto functor = [&, conv_desc, conv_attr, conv_index, arg3_size](
                        CPURuntimeContext* ctx, CPUExecutionContext* ectx) {
                        if (ctx->first_iteration)
                        {
                            mkldnn_emitter->build_convolution_forward<true>(
                                conv_desc, conv_attr, executor::global_cpu_engine, conv_index);
                        }
                        if (out_tensor != arg3_tensor)
                        {
                            memcpy(static_cast<char*>(out_tensor),
                                   static_cast<char*>(arg3_tensor),
                                   arg3_size);
                        }
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[0], arg0_tensor);
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[1], arg1_tensor);
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[2], arg2_tensor);
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[3], out_tensor);
                        cpu::mkldnn_utils::mkldnn_invoke_primitive(ctx, conv_index);
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

                auto& arg0_tensor = external_function->get_tensor_data(args[0].get_name());
                auto& arg1_tensor = external_function->get_tensor_data(args[1].get_name());
                auto& arg2_tensor = external_function->get_tensor_data(args[2].get_name());
                auto& out_tensor = external_function->get_tensor_data(out[0].get_name());
                size_t arg2_size = node->get_inputs()[2].get_tensor().size();

                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto conv_desc =
                        mkldnn_emitter->get_convolution_forward_desc<ngraph::op::ConvolutionAdd>(
                            node);
                    auto conv_attr =
                        mkldnn_emitter->get_convolution_forward_attr<ngraph::op::ConvolutionAdd>(
                            node);
                    size_t conv_index = mkldnn_emitter->convolution_forward_init(false);
                    auto& deps = mkldnn_emitter->get_primitive_deps(conv_index);

                    auto functor = [&, conv_desc, conv_attr, conv_index, arg2_size](
                        CPURuntimeContext* ctx, CPUExecutionContext* ectx) {
                        if (ctx->first_iteration)
                        {
                            mkldnn_emitter->build_convolution_forward<false>(
                                conv_desc, conv_attr, executor::global_cpu_engine, conv_index);
                        }
                        if (out_tensor != arg2_tensor)
                        {
                            memcpy(static_cast<char*>(out_tensor),
                                   static_cast<char*>(arg2_tensor),
                                   arg2_size);
                        }
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[0], arg0_tensor);
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[1], arg1_tensor);
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[2], out_tensor);
                        cpu::mkldnn_utils::mkldnn_invoke_primitive(ctx, conv_index);
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
                auto result_shape = out[0].get_shape();

                auto& arg0_tensor = external_function->get_tensor_data(args[0].get_name());
                auto& arg1_tensor = external_function->get_tensor_data(args[1].get_name());
                auto& out_tensor = external_function->get_tensor_data(out[0].get_name());

                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto bwd_desc = mkldnn_emitter->get_convolution_backward_data_desc<
                        ngraph::op::ConvolutionBackpropData>(node);
                    auto fwd_desc = mkldnn_emitter->get_convolution_forward_desc_for_backward_op<
                        ngraph::op::ConvolutionBackpropData>(node);
                    // ConvolutionBackpropData needs 4 primitives: weights, delta, result,
                    // and convolution_backward.
                    auto conv_index = mkldnn_emitter->reserve_primitive_space(4);
                    auto& deps = mkldnn_emitter->get_primitive_deps(conv_index);

                    auto functor = [&, bwd_desc, fwd_desc, conv_index](CPURuntimeContext* ctx,
                                                                       CPUExecutionContext* ectx) {
                        if (ctx->first_iteration)
                        {
                            mkldnn_emitter->build_convolution_backward_data(
                                bwd_desc, fwd_desc, conv_index);
                        }
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[0], arg0_tensor);
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[1], arg1_tensor);
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[2], out_tensor);
                        cpu::mkldnn_utils::mkldnn_invoke_primitive(ctx, conv_index);
                    };
                    functors.emplace_back(functor);
                }
                else
                {
                    std::function<decltype(runtime::cpu::kernel::convolution<float>)> kernel;

                    SELECT_KERNEL(
                        kernel, out[0].get_element_type(), runtime::cpu::kernel::convolution);

                    auto window_movement_strides =
                        convolution->get_window_movement_strides_backward();
                    auto window_dilation_strides =
                        convolution->get_window_dilation_strides_backward();
                    auto padding_below = convolution->get_padding_below_backward();
                    auto padding_above = convolution->get_padding_above_backward();
                    auto data_dilation_strides = convolution->get_data_dilation_strides_backward();

                    auto functor = [&,
                                    kernel,
                                    arg0_shape,
                                    arg1_shape,
                                    result_shape,
                                    window_movement_strides,
                                    window_dilation_strides,
                                    padding_below,
                                    padding_above,
                                    data_dilation_strides](CPURuntimeContext* ctx,
                                                           CPUExecutionContext* ectx) {
                        kernel(arg1_tensor,
                               arg0_tensor,
                               out_tensor,
                               arg1_shape,
                               arg0_shape,
                               result_shape,
                               window_movement_strides,
                               window_dilation_strides,
                               padding_below,
                               padding_above,
                               data_dilation_strides,
                               0,
                               1,
                               0,
                               1,
                               0,
                               1,
                               true);
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
                auto result_shape = out[0].get_shape();

                auto& arg0_tensor = external_function->get_tensor_data(args[0].get_name());
                auto& arg1_tensor = external_function->get_tensor_data(args[1].get_name());
                auto& out_tensor = external_function->get_tensor_data(out[0].get_name());

                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto bwd_desc = mkldnn_emitter->get_convolution_backward_weights_desc<
                        ngraph::op::ConvolutionBackpropFilters>(node);
                    auto fwd_desc = mkldnn_emitter->get_convolution_forward_desc_for_backward_op<
                        ngraph::op::ConvolutionBackpropFilters>(node);
                    // ConvolutionBackpropFilter needs 4 primitives: input, delta, weights_delta,
                    // and convolution_backward_weights.
                    auto conv_index = mkldnn_emitter->reserve_primitive_space(4);
                    auto& deps = mkldnn_emitter->get_primitive_deps(conv_index);

                    auto functor = [&, bwd_desc, fwd_desc, conv_index](CPURuntimeContext* ctx,
                                                                       CPUExecutionContext* ectx) {
                        if (ctx->first_iteration)
                        {
                            mkldnn_emitter->build_convolution_backward_weights(
                                bwd_desc, fwd_desc, conv_index);
                        }
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[0], arg0_tensor);
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[1], arg1_tensor);
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[2], out_tensor);
                        cpu::mkldnn_utils::mkldnn_invoke_primitive(ctx, conv_index);
                    };
                    functors.emplace_back(functor);
                }
                else
                {
                    std::function<decltype(runtime::cpu::kernel::convolution<float>)> kernel;

                    SELECT_KERNEL(
                        kernel, out[0].get_element_type(), runtime::cpu::kernel::convolution);

                    auto window_movement_strides =
                        convolution->get_window_movement_strides_backward();
                    auto window_dilation_strides =
                        convolution->get_window_dilation_strides_backward();
                    auto padding_below = convolution->get_padding_below_backward();
                    auto padding_above = convolution->get_padding_above_backward();
                    auto data_dilation_strides = convolution->get_data_dilation_strides_backward();

                    auto functor = [&,
                                    kernel,
                                    arg0_shape,
                                    arg1_shape,
                                    result_shape,
                                    window_movement_strides,
                                    window_dilation_strides,
                                    padding_below,
                                    padding_above,
                                    data_dilation_strides](CPURuntimeContext* ctx,
                                                           CPUExecutionContext* ectx) {
                        kernel(arg0_tensor,
                               arg1_tensor,
                               out_tensor,
                               arg0_shape,
                               arg1_shape,
                               result_shape,
                               window_movement_strides,
                               window_dilation_strides,
                               padding_below,
                               padding_above,
                               data_dilation_strides,
                               1,
                               0,
                               0,
                               1,
                               1,
                               0,
                               false);
                    };
                    functors.emplace_back(functor);
                }
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::ConvolutionBiasBackpropFiltersBias)
            {
                auto& functors = external_function->get_functors();

                auto& arg0_tensor = external_function->get_tensor_data(args[0].get_name());
                auto& arg1_tensor = external_function->get_tensor_data(args[1].get_name());
                auto& out0_tensor = external_function->get_tensor_data(out[0].get_name());
                auto& out1_tensor = external_function->get_tensor_data(out[1].get_name());

                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto bwd_desc = mkldnn_emitter->get_convolution_backward_weights_desc<
                        ngraph::op::ConvolutionBiasBackpropFiltersBias>(node);
                    auto fwd_desc = mkldnn_emitter->get_convolution_forward_desc_for_backward_op<
                        ngraph::op::ConvolutionBiasBackpropFiltersBias>(node);
                    // ConvolutionBiasBackpropFilter needs 5 primitives: input, delta, weights_delta,
                    // bias_delta, and convolution_backward_weights.
                    auto conv_index = mkldnn_emitter->reserve_primitive_space(5);
                    auto& deps = mkldnn_emitter->get_primitive_deps(conv_index);

                    auto functor = [&, bwd_desc, fwd_desc, conv_index](CPURuntimeContext* ctx,
                                                                       CPUExecutionContext* ectx) {
                        if (ctx->first_iteration)
                        {
                            mkldnn_emitter->build_convolution_backward_weights_bias(
                                bwd_desc, fwd_desc, conv_index);
                        }
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[0], arg0_tensor);
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[1], arg1_tensor);
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[2], out0_tensor);
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[3], out1_tensor);
                        cpu::mkldnn_utils::mkldnn_invoke_primitive(ctx, conv_index);
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

                auto& arg0_tensor = external_function->get_tensor_data(args[0].get_name());
                auto& arg1_tensor = external_function->get_tensor_data(args[1].get_name());
                auto& out_tensor = external_function->get_tensor_data(out[0].get_name());

                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto conv_desc =
                        mkldnn_emitter->get_convolution_forward_desc<ngraph::op::GroupConvolution>(
                            node);
                    auto conv_attr =
                        mkldnn_emitter->get_convolution_forward_attr<ngraph::op::GroupConvolution>(
                            node);
                    size_t conv_index = mkldnn_emitter->convolution_forward_init();
                    auto& deps = mkldnn_emitter->get_primitive_deps(conv_index);

                    auto functor = [&, conv_desc, conv_attr, conv_index](
                        CPURuntimeContext* ctx, CPUExecutionContext* ectx) {
                        if (ctx->first_iteration)
                        {
                            mkldnn_emitter->build_convolution_forward<false>(
                                conv_desc, conv_attr, executor::global_cpu_engine, conv_index);
                        }

                        // group convolution
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[0], arg0_tensor);
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[1], arg1_tensor);
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[2], out_tensor);
                        cpu::mkldnn_utils::mkldnn_invoke_primitive(ctx, conv_index);
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

                auto& arg0_tensor = external_function->get_tensor_data(args[0].get_name());
                auto& arg1_tensor = external_function->get_tensor_data(args[1].get_name());
                auto& arg2_tensor = external_function->get_tensor_data(args[2].get_name());
                auto& out_tensor = external_function->get_tensor_data(out[0].get_name());

                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto conv_desc =
                        mkldnn_emitter
                            ->get_convolution_forward_desc<ngraph::op::GroupConvolutionBias>(node);
                    auto conv_attr =
                        mkldnn_emitter
                            ->get_convolution_forward_attr<ngraph::op::GroupConvolutionBias>(node);
                    size_t conv_index = mkldnn_emitter->convolution_forward_init(true);
                    auto& deps = mkldnn_emitter->get_primitive_deps(conv_index);

                    auto functor = [&, conv_desc, conv_attr, conv_index](
                        CPURuntimeContext* ctx, CPUExecutionContext* ectx) {
                        if (ctx->first_iteration)
                        {
                            mkldnn_emitter->build_convolution_forward<true>(
                                conv_desc, conv_attr, executor::global_cpu_engine, conv_index);
                        }
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[0], arg0_tensor);
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[1], arg1_tensor);
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[2], arg2_tensor);
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[3], out_tensor);
                        cpu::mkldnn_utils::mkldnn_invoke_primitive(ctx, conv_index);
                    };
                    functors.emplace_back(functor);
                }
                else
                {
                    throw ngraph_error("unsupported parameters for GroupConvolutionBias");
                }
            }

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
        }
    }
}
