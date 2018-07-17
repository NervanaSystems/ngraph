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

#include "ngraph/op/convolution.hpp"
#include "ngraph/runtime/cpu/cpu_builder.hpp"
#include "ngraph/runtime/cpu/kernel/convolution.hpp"
#include "ngraph/runtime/cpu/mkldnn_invoke.hpp"
#include "ngraph/runtime/cpu/mkldnn_utils.hpp"
#include "ngraph/runtime/cpu/op/conv_bias.hpp"
#include "ngraph/runtime/cpu/op/conv_relu.hpp"

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
                auto& tensor_data = external_function->get_tensor_data();

                auto arg0_shape = args[0].get_shape();
                auto arg1_shape = args[1].get_shape();
                auto result_shape = out[0].get_shape();

                auto& arg0_tensor = tensor_data[args[0].get_name()];
                auto& arg1_tensor = tensor_data[args[1].get_name()];
                auto& out_tensor = tensor_data[out[0].get_name()];

                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto conv_index =
                        mkldnn_emitter->build_convolution<ngraph::op::Convolution>(node, args, out);
                    auto& deps = mkldnn_emitter->get_primitive_deps(conv_index);

                    auto functor = [&, conv_index](CPURuntimeContext* ctx) {
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
                                    data_dilation_strides](CPURuntimeContext* ctx) {
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
                auto& tensor_data = external_function->get_tensor_data();

                auto& arg0_tensor = tensor_data[args[0].get_name()];
                auto& arg1_tensor = tensor_data[args[1].get_name()];
                auto& out_tensor = tensor_data[out[0].get_name()];

                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto conv_index =
                        mkldnn_emitter->build_convolution<ngraph::op::ConvolutionRelu>(
                            node, args, out);
                    auto& deps = mkldnn_emitter->get_primitive_deps(conv_index);

                    auto functor = [&, conv_index](CPURuntimeContext* ctx) {
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
                auto& tensor_data = external_function->get_tensor_data();

                auto& arg0_tensor = tensor_data[args[0].get_name()];
                auto& arg1_tensor = tensor_data[args[1].get_name()];
                auto& arg2_tensor = tensor_data[args[2].get_name()];
                auto& out_tensor = tensor_data[out[0].get_name()];

                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto conv_index =
                        mkldnn_emitter->build_convolution<ngraph::op::ConvolutionBias>(
                            node, args, out);
                    auto& deps = mkldnn_emitter->get_primitive_deps(conv_index);

                    auto functor = [&, conv_index](CPURuntimeContext* ctx) {
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
                auto& tensor_data = external_function->get_tensor_data();

                auto& arg0_tensor = tensor_data[args[0].get_name()];
                auto& arg1_tensor = tensor_data[args[1].get_name()];
                auto& arg2_tensor = tensor_data[args[2].get_name()];
                auto& out_tensor = tensor_data[out[0].get_name()];

                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto conv_index =
                        mkldnn_emitter->build_convolution<ngraph::op::ConvolutionBiasAdd>(
                            node, args, out);
                    auto& deps = mkldnn_emitter->get_primitive_deps(conv_index);

                    auto functor = [&, conv_index](CPURuntimeContext* ctx) {
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
            void Builder::BUILDER_DECL(ngraph::op::ConvolutionBackpropData)
            {
                auto convolution = static_cast<const ngraph::op::ConvolutionBackpropData*>(node);

                auto& functors = external_function->get_functors();
                auto& tensor_data = external_function->get_tensor_data();

                auto arg0_shape = args[0].get_shape();
                auto arg1_shape = args[1].get_shape();
                auto result_shape = out[0].get_shape();

                auto& arg0_tensor = tensor_data[args[0].get_name()];
                auto& arg1_tensor = tensor_data[args[1].get_name()];
                auto& out_tensor = tensor_data[out[0].get_name()];

                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    Strides window_dilation_strides_adjusted;

                    for (size_t s : convolution->get_window_dilation_strides_forward())
                    {
                        window_dilation_strides_adjusted.push_back(s - 1);
                    }

                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    // HACK to help MKLDNN pick the right implementation
                    auto weights_format =
                        runtime::cpu::mkldnn_utils::get_input_mkldnn_format(node, 0);
                    if (weights_format == mkldnn::memory::format::nchw)
                    {
                        weights_format = mkldnn::memory::format::oihw;
                    }
                    auto weights_desc =
                        mkldnn_emitter->build_memory_descriptor(args[0], weights_format);
                    auto delta_desc = mkldnn_emitter->build_memory_descriptor(
                        args[1], runtime::cpu::mkldnn_utils::get_input_mkldnn_format(node, 1));
                    auto result_desc = mkldnn_emitter->build_memory_descriptor(
                        out[0], runtime::cpu::mkldnn_utils::get_output_mkldnn_format(node, 0));

                    size_t conv_bwd_data_index = mkldnn_emitter->build_convolution_backward_data(
                        weights_desc,
                        delta_desc,
                        result_desc,
                        convolution->get_window_movement_strides_forward(),
                        window_dilation_strides_adjusted,
                        convolution->get_padding_below_forward(),
                        convolution->get_padding_above_forward());

                    auto& deps = mkldnn_emitter->get_primitive_deps(conv_bwd_data_index);

                    auto functor = [&, conv_bwd_data_index](CPURuntimeContext* ctx) {
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[0], arg0_tensor);
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[1], arg1_tensor);
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[2], out_tensor);
                        cpu::mkldnn_utils::mkldnn_invoke_primitive(ctx, conv_bwd_data_index);
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
                                    data_dilation_strides](CPURuntimeContext* ctx) {
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
                auto& tensor_data = external_function->get_tensor_data();

                auto arg0_shape = args[0].get_shape();
                auto arg1_shape = args[1].get_shape();
                auto result_shape = out[0].get_shape();

                auto& arg0_tensor = tensor_data[args[0].get_name()];
                auto& arg1_tensor = tensor_data[args[1].get_name()];
                auto& out_tensor = tensor_data[out[0].get_name()];

                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    Strides window_dilation_strides_adjusted;

                    for (size_t s : convolution->get_window_dilation_strides_forward())
                    {
                        window_dilation_strides_adjusted.push_back(s - 1);
                    }

                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto input_desc = mkldnn_emitter->build_memory_descriptor(
                        args[0], runtime::cpu::mkldnn_utils::get_input_mkldnn_format(node, 0));
                    auto delta_desc = mkldnn_emitter->build_memory_descriptor(
                        args[1], runtime::cpu::mkldnn_utils::get_input_mkldnn_format(node, 1));
                    auto result_desc = mkldnn_emitter->build_memory_descriptor(
                        out[0], runtime::cpu::mkldnn_utils::get_output_mkldnn_format(node, 0));

                    size_t conv_bwd_weights_index =
                        mkldnn_emitter->build_convolution_backward_weights(
                            input_desc,
                            delta_desc,
                            result_desc,
                            convolution->get_window_movement_strides_forward(),
                            window_dilation_strides_adjusted,
                            convolution->get_padding_below_forward(),
                            convolution->get_padding_above_forward());

                    auto& deps = mkldnn_emitter->get_primitive_deps(conv_bwd_weights_index);

                    auto functor = [&, conv_bwd_weights_index](CPURuntimeContext* ctx) {
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[0], arg0_tensor);
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[1], arg1_tensor);
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[2], out_tensor);
                        cpu::mkldnn_utils::mkldnn_invoke_primitive(ctx, conv_bwd_weights_index);
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
                                    data_dilation_strides](CPURuntimeContext* ctx) {
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
        }
    }
}
