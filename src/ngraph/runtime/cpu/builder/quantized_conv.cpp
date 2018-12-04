//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
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
                    auto qconv = static_cast<const ngraph::op::QuantizedConvolution*>(node);
                    auto& functors = external_function->get_functors();
                    auto& arg0_tensor = external_function->get_tensor_data(args[0].get_name());
                    auto& arg1_tensor = external_function->get_tensor_data(args[1].get_name());
                    auto& out0_tensor = external_function->get_tensor_data(out[0].get_name());

                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto scale_const_op =
                        std::dynamic_pointer_cast<ngraph::op::Constant>(qconv->get_argument(2));
                    std::vector<float> scales;
                    if (scale_const_op == nullptr)
                    {
                        auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                        auto weights_desc = mkldnn_utils::get_input_mkldnn_md(node, 1);
                        auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);
                        auto padding_below = qconv->get_padding_below();
                        auto padding_above = qconv->get_padding_above();
                        auto strides = qconv->get_window_movement_strides();
                        Strides dilation_strides_adjusted;

                        for (size_t s : qconv->get_window_dilation_strides())
                        {
                            dilation_strides_adjusted.push_back(s - 1);
                        }

                        auto& arg2_tensor = external_function->get_tensor_data(args[2].get_name());
                        auto scales_size = shape_size(args[2].get_shape());

                        auto conv_index =
                            mkldnn_emitter->build_convolution<ngraph::op::QuantizedConvolution>(
                                node, args, out);
                        auto& deps = mkldnn_emitter->get_primitive_deps(conv_index);

                        auto functor = [&,
                                        input_desc,
                                        weights_desc,
                                        result_desc,
                                        scales_size,
                                        padding_below,
                                        padding_above,
                                        strides,
                                        dilation_strides_adjusted,
                                        conv_index](CPURuntimeContext* ctx,
                                                    CPUExecutionContext* ectx) {
                            // Create MKLDNN convolution primitive during the first iteration.
                            // Assumes the scales dont change for the duration of the graph
                            if (ctx->first_iteration)
                            {
                                mkldnn::primitive_attr attr;
                                vector<float> dyn_scales;
                                dyn_scales.assign(static_cast<float*>(arg2_tensor),
                                                  static_cast<float*>(arg2_tensor) + scales_size);
                                attr.set_output_scales(0, dyn_scales);
                                attr.set_int_output_round_mode(mkldnn::round_mode::round_nearest);
                                *ctx->mkldnn_primitives[conv_index] = mkldnn::convolution_forward(
                                    {{mkldnn::prop_kind::forward,
                                      mkldnn::algorithm::convolution_direct,
                                      input_desc,
                                      weights_desc,
                                      result_desc,
                                      mkldnn::memory::dims(strides.begin(), strides.end()),
                                      mkldnn::memory::dims(dilation_strides_adjusted.begin(),
                                                           dilation_strides_adjusted.end()),
                                      mkldnn::memory::dims(padding_below.begin(),
                                                           padding_below.end()),
                                      mkldnn::memory::dims(padding_above.begin(),
                                                           padding_above.end()),
                                      mkldnn::padding_kind::zero},
                                     attr,
                                     executor::global_cpu_engine},
                                    *ctx->mkldnn_primitives[deps[0]],
                                    *ctx->mkldnn_primitives[deps[1]],
                                    *ctx->mkldnn_primitives[deps[2]]);
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
                        auto conv_index =
                            mkldnn_emitter->build_convolution<ngraph::op::QuantizedConvolution>(
                                node, args, out);
                        auto& deps = mkldnn_emitter->get_primitive_deps(conv_index);

                        auto functor = [&, conv_index](CPURuntimeContext* ctx,
                                                       CPUExecutionContext* ectx) {
                            cpu::mkldnn_utils::set_memory_ptr(ctx, deps[0], arg0_tensor);
                            cpu::mkldnn_utils::set_memory_ptr(ctx, deps[1], arg1_tensor);
                            cpu::mkldnn_utils::set_memory_ptr(ctx, deps[2], out0_tensor);
                            cpu::mkldnn_utils::mkldnn_invoke_primitive(ctx, conv_index);
                        };
                        functors.emplace_back(functor);
                    }
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
                    auto qconv = static_cast<const ngraph::op::QuantizedConvolutionRelu*>(node);
                    auto& functors = external_function->get_functors();
                    auto& arg0_tensor = external_function->get_tensor_data(args[0].get_name());
                    auto& arg1_tensor = external_function->get_tensor_data(args[1].get_name());
                    auto& out0_tensor = external_function->get_tensor_data(out[0].get_name());

                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto scale_const_op =
                        std::dynamic_pointer_cast<ngraph::op::Constant>(qconv->get_argument(2));
                    std::vector<float> scales;
                    if (scale_const_op == nullptr)
                    {
                        auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                        auto weights_desc = mkldnn_utils::get_input_mkldnn_md(node, 1);
                        auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);
                        auto padding_below = qconv->get_padding_below();
                        auto padding_above = qconv->get_padding_above();
                        auto strides = qconv->get_window_movement_strides();
                        Strides dilation_strides_adjusted;

                        for (size_t s : qconv->get_window_dilation_strides())
                        {
                            dilation_strides_adjusted.push_back(s - 1);
                        }

                        auto& arg2_tensor = external_function->get_tensor_data(args[2].get_name());
                        auto scales_size = shape_size(args[2].get_shape());

                        auto conv_index =
                            mkldnn_emitter->build_convolution<ngraph::op::QuantizedConvolutionRelu>(
                                node, args, out);
                        auto& deps = mkldnn_emitter->get_primitive_deps(conv_index);

                        auto functor = [&,
                                        input_desc,
                                        weights_desc,
                                        result_desc,
                                        scales_size,
                                        padding_below,
                                        padding_above,
                                        strides,
                                        dilation_strides_adjusted,
                                        conv_index](CPURuntimeContext* ctx,
                                                    CPUExecutionContext* ectx) {
                            // Create MKLDNN convolution primitive during the first iteration.
                            // Assumes the scales dont change for the duration of the graph
                            if (ctx->first_iteration)
                            {
                                mkldnn::primitive_attr attr;
                                vector<float> dyn_scales;
                                dyn_scales.assign(static_cast<float*>(arg2_tensor),
                                                  static_cast<float*>(arg2_tensor) + scales_size);
                                attr.set_output_scales(0, dyn_scales);
                                attr.set_int_output_round_mode(mkldnn::round_mode::round_nearest);
                                *ctx->mkldnn_primitives[conv_index] = mkldnn::convolution_forward(
                                    {{mkldnn::prop_kind::forward,
                                      mkldnn::algorithm::convolution_direct,
                                      input_desc,
                                      weights_desc,
                                      result_desc,
                                      mkldnn::memory::dims(strides.begin(), strides.end()),
                                      mkldnn::memory::dims(dilation_strides_adjusted.begin(),
                                                           dilation_strides_adjusted.end()),
                                      mkldnn::memory::dims(padding_below.begin(),
                                                           padding_below.end()),
                                      mkldnn::memory::dims(padding_above.begin(),
                                                           padding_above.end()),
                                      mkldnn::padding_kind::zero},
                                     attr,
                                     executor::global_cpu_engine},
                                    *ctx->mkldnn_primitives[deps[0]],
                                    *ctx->mkldnn_primitives[deps[1]],
                                    *ctx->mkldnn_primitives[deps[2]]);
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
                        auto conv_index =
                            mkldnn_emitter->build_convolution<ngraph::op::QuantizedConvolutionRelu>(
                                node, args, out);
                        auto& deps = mkldnn_emitter->get_primitive_deps(conv_index);

                        auto functor = [&, conv_index](CPURuntimeContext* ctx,
                                                       CPUExecutionContext* ectx) {
                            cpu::mkldnn_utils::set_memory_ptr(ctx, deps[0], arg0_tensor);
                            cpu::mkldnn_utils::set_memory_ptr(ctx, deps[1], arg1_tensor);
                            cpu::mkldnn_utils::set_memory_ptr(ctx, deps[2], out0_tensor);
                            cpu::mkldnn_utils::mkldnn_invoke_primitive(ctx, conv_index);
                        };
                        functors.emplace_back(functor);
                    }
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
                    auto qconv = static_cast<const ngraph::op::QuantizedConvolutionBias*>(node);
                    auto& functors = external_function->get_functors();
                    auto& arg0_tensor = external_function->get_tensor_data(args[0].get_name());
                    auto& arg1_tensor = external_function->get_tensor_data(args[1].get_name());
                    auto& arg2_tensor = external_function->get_tensor_data(args[2].get_name());
                    auto& out0_tensor = external_function->get_tensor_data(out[0].get_name());

                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto scale_const_op =
                        std::dynamic_pointer_cast<ngraph::op::Constant>(qconv->get_argument(3));
                    std::vector<float> scales;
                    if (scale_const_op == nullptr)
                    {
                        auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                        auto weights_desc = mkldnn_utils::get_input_mkldnn_md(node, 1);
                        auto bias_desc = mkldnn_utils::get_input_mkldnn_md(node, 2);
                        auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);
                        auto padding_below = qconv->get_padding_below();
                        auto padding_above = qconv->get_padding_above();
                        auto strides = qconv->get_window_movement_strides();
                        Strides dilation_strides_adjusted;

                        for (size_t s : qconv->get_window_dilation_strides())
                        {
                            dilation_strides_adjusted.push_back(s - 1);
                        }

                        auto& arg3_tensor = external_function->get_tensor_data(args[3].get_name());
                        auto scales_size = shape_size(args[3].get_shape());

                        auto conv_index =
                            mkldnn_emitter->build_convolution<ngraph::op::QuantizedConvolutionBias>(
                                node, args, out);
                        auto& deps = mkldnn_emitter->get_primitive_deps(conv_index);

                        auto functor = [&,
                                        input_desc,
                                        weights_desc,
                                        bias_desc,
                                        result_desc,
                                        scales_size,
                                        padding_below,
                                        padding_above,
                                        strides,
                                        dilation_strides_adjusted,
                                        conv_index](CPURuntimeContext* ctx,
                                                    CPUExecutionContext* ectx) {
                            // Create MKLDNN convolution primitive during the first iteration.
                            // Assumes the scales dont change for the duration of the graph
                            if (ctx->first_iteration)
                            {
                                mkldnn::primitive_attr attr;
                                vector<float> dyn_scales;
                                dyn_scales.assign(static_cast<float*>(arg3_tensor),
                                                  static_cast<float*>(arg3_tensor) + scales_size);
                                attr.set_output_scales(0, dyn_scales);
                                attr.set_int_output_round_mode(mkldnn::round_mode::round_nearest);
                                *ctx->mkldnn_primitives[conv_index] = mkldnn::convolution_forward(
                                    {{mkldnn::prop_kind::forward,
                                      mkldnn::algorithm::convolution_direct,
                                      input_desc,
                                      weights_desc,
                                      bias_desc,
                                      result_desc,
                                      mkldnn::memory::dims(strides.begin(), strides.end()),
                                      mkldnn::memory::dims(dilation_strides_adjusted.begin(),
                                                           dilation_strides_adjusted.end()),
                                      mkldnn::memory::dims(padding_below.begin(),
                                                           padding_below.end()),
                                      mkldnn::memory::dims(padding_above.begin(),
                                                           padding_above.end()),
                                      mkldnn::padding_kind::zero},
                                     attr,
                                     executor::global_cpu_engine},
                                    *ctx->mkldnn_primitives[deps[0]],
                                    *ctx->mkldnn_primitives[deps[1]],
                                    *ctx->mkldnn_primitives[deps[2]],
                                    *ctx->mkldnn_primitives[deps[3]]);
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
                        auto conv_index =
                            mkldnn_emitter->build_convolution<ngraph::op::QuantizedConvolutionBias>(
                                node, args, out);
                        auto& deps = mkldnn_emitter->get_primitive_deps(conv_index);

                        auto functor = [&, conv_index](CPURuntimeContext* ctx,
                                                       CPUExecutionContext* ectx) {
                            cpu::mkldnn_utils::set_memory_ptr(ctx, deps[0], arg0_tensor);
                            cpu::mkldnn_utils::set_memory_ptr(ctx, deps[1], arg1_tensor);
                            cpu::mkldnn_utils::set_memory_ptr(ctx, deps[2], arg2_tensor);
                            cpu::mkldnn_utils::set_memory_ptr(ctx, deps[3], out0_tensor);
                            cpu::mkldnn_utils::mkldnn_invoke_primitive(ctx, conv_index);
                        };
                        functors.emplace_back(functor);
                    }
                }
                else
                {
                    throw ngraph_error(
                        "unsupported parameters for QuantizedConvolutionBias via DEX");
                }
            }

            REGISTER_OP_BUILDER(QuantizedConvolution);
            REGISTER_OP_BUILDER(QuantizedConvolutionRelu);
            REGISTER_OP_BUILDER(QuantizedConvolutionBias);
        }
    }
}
