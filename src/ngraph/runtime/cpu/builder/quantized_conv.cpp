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

#include "ngraph/op/experimental/quantized_conv.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/experimental/quantized_conv_bias.hpp"
#include "ngraph/op/experimental/quantized_conv_relu.hpp"
#include "ngraph/runtime/cpu/cpu_builder.hpp"
#include "ngraph/runtime/cpu/cpu_executor.hpp"
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
                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    auto& functors = external_function->get_functors();
                    auto& arg0_tensor = external_function->get_tensor_data(args[0].get_name());
                    auto& arg1_tensor = external_function->get_tensor_data(args[1].get_name());
                    auto& arg2_tensor = external_function->get_tensor_data(args[2].get_name());
                    auto& out0_tensor = external_function->get_tensor_data(out[0].get_name());

                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto scales_size = shape_size(args[2].get_shape());

                    auto conv_desc =
                        mkldnn_emitter
                            ->get_convolution_forward_desc<ngraph::op::QuantizedConvolution>(
                                node, args, out);
                    auto conv_attr =
                        mkldnn_emitter
                            ->get_convolution_forward_attr<ngraph::op::QuantizedConvolution>(node);
                    size_t conv_index = mkldnn_emitter->convolution_forward_init();
                    auto& deps = mkldnn_emitter->get_primitive_deps(conv_index);

                    auto functor = [&, scales_size, conv_desc, conv_attr, deps, conv_index](
                        CPURuntimeContext* ctx, CPUExecutionContext* ectx) mutable {
                        // Create MKLDNN convolution primitive during the first iteration.
                        // Assumes the scales dont change for the duration of the graph
                        if (ctx->first_iteration)
                        {
                            vector<float> dyn_scales;
                            dyn_scales.assign(static_cast<float*>(arg2_tensor),
                                              static_cast<float*>(arg2_tensor) + scales_size);
                            conv_attr.set_output_scales(0, dyn_scales);
                            mkldnn_emitter->convolution_forward<false>(
                                conv_desc, conv_attr, executor::global_cpu_engine, conv_index);
                        }
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[0], arg0_tensor);
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[1], arg1_tensor);
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[2], out0_tensor);
                        cpu::mkldnn_utils::mkldnn_invoke_primitive(ctx, conv_index);
                    };
                    functors.emplace_back(functor);
                }
                else
                {
                    throw ngraph_error("unsupported parameters for QuantizedConvolution via DEX");
                }
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::QuantizedConvolutionRelu)
            {
                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    auto& functors = external_function->get_functors();
                    auto& arg0_tensor = external_function->get_tensor_data(args[0].get_name());
                    auto& arg1_tensor = external_function->get_tensor_data(args[1].get_name());
                    auto& arg2_tensor = external_function->get_tensor_data(args[2].get_name());
                    auto& out0_tensor = external_function->get_tensor_data(out[0].get_name());

                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto scales_size = shape_size(args[2].get_shape());

                    auto conv_desc =
                        mkldnn_emitter
                            ->get_convolution_forward_desc<ngraph::op::QuantizedConvolutionRelu>(
                                node, args, out);
                    auto conv_attr =
                        mkldnn_emitter
                            ->get_convolution_forward_attr<ngraph::op::QuantizedConvolutionRelu>(
                                node);
                    size_t conv_index = mkldnn_emitter->convolution_forward_init();
                    auto& deps = mkldnn_emitter->get_primitive_deps(conv_index);

                    auto functor = [&, scales_size, conv_desc, conv_attr, deps, conv_index](
                        CPURuntimeContext* ctx, CPUExecutionContext* ectx) mutable {
                        if (ctx->first_iteration)
                        {
                            vector<float> dyn_scales;
                            dyn_scales.assign(static_cast<float*>(arg2_tensor),
                                              static_cast<float*>(arg2_tensor) + scales_size);
                            conv_attr.set_output_scales(0, dyn_scales);
                            mkldnn_emitter->convolution_forward<false>(
                                conv_desc, conv_attr, executor::global_cpu_engine, conv_index);
                        }
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[0], arg0_tensor);
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[1], arg1_tensor);
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[2], out0_tensor);
                        cpu::mkldnn_utils::mkldnn_invoke_primitive(ctx, conv_index);
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
                    auto& arg0_tensor = external_function->get_tensor_data(args[0].get_name());
                    auto& arg1_tensor = external_function->get_tensor_data(args[1].get_name());
                    auto& arg2_tensor = external_function->get_tensor_data(args[2].get_name());
                    auto& arg3_tensor = external_function->get_tensor_data(args[3].get_name());
                    auto& out0_tensor = external_function->get_tensor_data(out[0].get_name());

                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto scales_size = shape_size(args[3].get_shape());

                    auto conv_desc =
                        mkldnn_emitter
                            ->get_convolution_forward_desc<ngraph::op::QuantizedConvolutionBias>(
                                node, args, out);
                    auto conv_attr =
                        mkldnn_emitter
                            ->get_convolution_forward_attr<ngraph::op::QuantizedConvolutionBias>(
                                node);
                    size_t conv_index = mkldnn_emitter->convolution_forward_init(true);
                    auto& deps = mkldnn_emitter->get_primitive_deps(conv_index);

                    auto functor = [&, scales_size, conv_desc, conv_attr, deps, conv_index](
                        CPURuntimeContext* ctx, CPUExecutionContext* ectx) mutable {
                        if (ctx->first_iteration)
                        {
                            vector<float> dyn_scales;
                            dyn_scales.assign(static_cast<float*>(arg3_tensor),
                                              static_cast<float*>(arg3_tensor) + scales_size);
                            conv_attr.set_output_scales(0, dyn_scales);
                            mkldnn_emitter->convolution_forward<true>(
                                conv_desc, conv_attr, executor::global_cpu_engine, conv_index);
                        }
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[0], arg0_tensor);
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[1], arg1_tensor);
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[2], arg2_tensor);
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[3], out0_tensor);
                        cpu::mkldnn_utils::mkldnn_invoke_primitive(ctx, conv_index);
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
                    auto& arg0_tensor = external_function->get_tensor_data(args[0].get_name());
                    auto& arg1_tensor = external_function->get_tensor_data(args[1].get_name());
                    auto& arg2_tensor = external_function->get_tensor_data(args[2].get_name());
                    auto& arg3_tensor = external_function->get_tensor_data(args[3].get_name());
                    auto& arg4_tensor = external_function->get_tensor_data(args[4].get_name());
                    auto& arg5_tensor = external_function->get_tensor_data(args[5].get_name());
                    auto& out0_tensor = external_function->get_tensor_data(out[0].get_name());

                    size_t arg3_size = args[3].get_size();
                    auto scales_size = shape_size(args[4].get_shape());
                    auto sum_scales_size = shape_size(args[5].get_shape());

                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();

                    auto conv_desc =
                        mkldnn_emitter
                            ->get_convolution_forward_desc<ngraph::op::QuantizedConvolutionBiasAdd>(
                                node, args, out);
                    auto conv_attr =
                        mkldnn_emitter
                            ->get_convolution_forward_attr<ngraph::op::QuantizedConvolutionBiasAdd>(
                                node);
                    size_t conv_index = mkldnn_emitter->convolution_forward_init(true);
                    auto& deps = mkldnn_emitter->get_primitive_deps(conv_index);

                    auto functor = [&,
                                    scales_size,
                                    sum_scales_size,
                                    conv_desc,
                                    conv_attr,
                                    deps,
                                    conv_index,
                                    arg3_size](CPURuntimeContext* ctx,
                                               CPUExecutionContext* ectx) mutable {
                        if (ctx->first_iteration)
                        {
                            vector<float> dyn_scales;
                            vector<float> dyn_post_op_scales;
                            dyn_scales.assign(static_cast<float*>(arg4_tensor),
                                              static_cast<float*>(arg4_tensor) + scales_size);
                            dyn_post_op_scales.assign(static_cast<float*>(arg5_tensor),
                                                      static_cast<float*>(arg5_tensor) +
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
                            conv_attr.set_output_scales(0, dyn_scales);
                            conv_attr.set_post_ops(new_pops);
                            mkldnn_emitter->convolution_forward<true>(
                                conv_desc, conv_attr, executor::global_cpu_engine, conv_index);
                        }

                        if (out0_tensor != arg3_tensor)
                        {
                            memcpy(static_cast<char*>(out0_tensor),
                                   static_cast<char*>(arg3_tensor),
                                   arg3_size);
                        }
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[0], arg0_tensor);
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[1], arg1_tensor);
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[2], arg2_tensor);
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[3], out0_tensor);
                        cpu::mkldnn_utils::mkldnn_invoke_primitive(ctx, conv_index);
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
                    auto& arg0_tensor = external_function->get_tensor_data(args[0].get_name());
                    auto& arg1_tensor = external_function->get_tensor_data(args[1].get_name());
                    auto& arg2_tensor = external_function->get_tensor_data(args[2].get_name());
                    auto& arg3_tensor = external_function->get_tensor_data(args[3].get_name());
                    auto& arg4_tensor = external_function->get_tensor_data(args[4].get_name());
                    auto& arg5_tensor = external_function->get_tensor_data(args[5].get_name());
                    auto& out0_tensor = external_function->get_tensor_data(out[0].get_name());

                    size_t arg3_size = args[3].get_size();
                    auto scales_size = shape_size(args[4].get_shape());
                    auto sum_scales_size = shape_size(args[5].get_shape());

                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();

                    auto conv_desc = mkldnn_emitter->get_convolution_forward_desc<
                        ngraph::op::QuantizedConvolutionBiasSignedAdd>(node, args, out);
                    auto conv_attr = mkldnn_emitter->get_convolution_forward_attr<
                        ngraph::op::QuantizedConvolutionBiasSignedAdd>(node);
                    size_t conv_index = mkldnn_emitter->convolution_forward_init(true);
                    auto& deps = mkldnn_emitter->get_primitive_deps(conv_index);

                    auto functor = [&,
                                    scales_size,
                                    sum_scales_size,
                                    conv_desc,
                                    conv_attr,
                                    deps,
                                    conv_index,
                                    arg3_size](CPURuntimeContext* ctx,
                                               CPUExecutionContext* ectx) mutable {
                        if (ctx->first_iteration)
                        {
                            vector<float> dyn_scales;
                            vector<float> dyn_post_op_scales;
                            dyn_scales.assign(static_cast<float*>(arg4_tensor),
                                              static_cast<float*>(arg4_tensor) + scales_size);
                            dyn_post_op_scales.assign(static_cast<float*>(arg5_tensor),
                                                      static_cast<float*>(arg5_tensor) +
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
                            conv_attr.set_output_scales(0, dyn_scales);
                            mkldnn_emitter->convolution_forward<true>(
                                conv_desc, conv_attr, executor::global_cpu_engine, conv_index);
                        }

                        if (out0_tensor != arg3_tensor)
                        {
                            memcpy(static_cast<char*>(out0_tensor),
                                   static_cast<char*>(arg3_tensor),
                                   arg3_size);
                        }
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[0], arg0_tensor);
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[1], arg1_tensor);
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[2], arg2_tensor);
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[3], out0_tensor);
                        cpu::mkldnn_utils::mkldnn_invoke_primitive(ctx, conv_index);
                    };
                    functors.emplace_back(functor);
                }
                else
                {
                    throw ngraph_error(
                        "unsupported parameters for QuantizedConvolutionBiasSignedAdd via DEX");
                }
            }

            REGISTER_OP_BUILDER(QuantizedConvolution);
            REGISTER_OP_BUILDER(QuantizedConvolutionRelu);
            REGISTER_OP_BUILDER(QuantizedConvolutionBias);
            REGISTER_OP_BUILDER(QuantizedConvolutionBiasAdd);
            REGISTER_OP_BUILDER(QuantizedConvolutionBiasSignedAdd);
        }
    }
}
