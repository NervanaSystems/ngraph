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

#include "ngraph/op/convolution.hpp"
#include "ngraph/runtime/cpu/cpu_builder.hpp"
#include "ngraph/runtime/cpu/kernel/convolution.hpp"
#include "ngraph/runtime/cpu/mkldnn_invoke.hpp"
#include "ngraph/runtime/cpu/mkldnn_utils.hpp"
#include "ngraph/runtime/cpu/op/conv_bias.hpp"
#include "ngraph/runtime/cpu/op/conv_relu.hpp"
#include "ngraph/runtime/cpu/op/group_conv.hpp"

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

                auto& arg0_tensor = external_function->get_tensor_data(args[0].get_name());
                auto& arg1_tensor = external_function->get_tensor_data(args[1].get_name());
                auto& out_tensor = external_function->get_tensor_data(out[0].get_name());

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

                auto& arg0_tensor = external_function->get_tensor_data(args[0].get_name());
                auto& arg1_tensor = external_function->get_tensor_data(args[1].get_name());
                auto& arg2_tensor = external_function->get_tensor_data(args[2].get_name());
                auto& out_tensor = external_function->get_tensor_data(out[0].get_name());

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

                auto& arg0_tensor = external_function->get_tensor_data(args[0].get_name());
                auto& arg1_tensor = external_function->get_tensor_data(args[1].get_name());
                auto& arg2_tensor = external_function->get_tensor_data(args[2].get_name());
                auto& out_tensor = external_function->get_tensor_data(out[0].get_name());

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

                auto arg0_shape = args[0].get_shape();
                auto arg1_shape = args[1].get_shape();
                auto result_shape = out[0].get_shape();

                auto& arg0_tensor = external_function->get_tensor_data(args[0].get_name());
                auto& arg1_tensor = external_function->get_tensor_data(args[1].get_name());
                auto& out_tensor = external_function->get_tensor_data(out[0].get_name());

                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto conv_index =
                        mkldnn_emitter
                            ->build_convolution_backward<ngraph::op::ConvolutionBackpropData>(
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

                auto arg0_shape = args[0].get_shape();
                auto arg1_shape = args[1].get_shape();
                auto result_shape = out[0].get_shape();

                auto& arg0_tensor = external_function->get_tensor_data(args[0].get_name());
                auto& arg1_tensor = external_function->get_tensor_data(args[1].get_name());
                auto& out_tensor = external_function->get_tensor_data(out[0].get_name());

                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto conv_index =
                        mkldnn_emitter
                            ->build_convolution_backward<ngraph::op::ConvolutionBackpropFilters>(
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
                    auto conv_index = mkldnn_emitter->build_convolution_backward<
                        ngraph::op::ConvolutionBiasBackpropFiltersBias>(node, args, out);
                    auto& deps = mkldnn_emitter->get_primitive_deps(conv_index);

                    auto functor = [&, conv_index](CPURuntimeContext* ctx) {
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

                auto convolution = static_cast<const ngraph::op::GroupConvolution*>(node);

                auto arg0_shape = args[0].get_shape();
                auto arg1_shape = args[1].get_shape();
                auto result_shape = out[0].get_shape();

                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    Strides window_dilation_strides_adjusted;
                    for (size_t s : convolution->get_window_dilation_strides())
                    {
                        window_dilation_strides_adjusted.push_back(s - 1);
                    }

                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto input_data_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);

                    Shape weights_shape_groups = convolution->get_weights_dimensions();

                    auto weights_desc_any = mkldnn::memory::desc(
                        mkldnn::memory::dims(weights_shape_groups.begin(),
                                             weights_shape_groups.end()),
                        mkldnn_utils::get_mkldnn_data_type(args[1].get_element_type()),
                        mkldnn::memory::format::any);

                    auto padding_below = convolution->get_padding_below();
                    auto padding_above = convolution->get_padding_above();
                    auto filter_strides = convolution->get_window_movement_strides();

                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);

                    auto weights_optimized_format =
                        mkldnn_emitter->query_convolution_forward_weight_format(
                            input_data_desc,
                            weights_desc_any,
                            result_desc,
                            filter_strides,
                            window_dilation_strides_adjusted,
                            padding_below,
                            padding_above);

                    // create workspace for holding the result of converting weights layouts
                    auto ws = std::unique_ptr<MKLDNNWorkspace>(new MKLDNNWorkspace(
                        shape_size(args[1].get_shape()) * args[1].get_element_type().size()));
                    auto ws_buf_index = mkldnn_emitter->insert_workspace(ws);

                    // descriptors for reorder operation
                    auto input_reorder_desc =
                        mkldnn_emitter->build_memory_descriptor(weights_shape_groups,
                                                                args[1].get_element_type(),
                                                                mkldnn::memory::format::goihw);

                    auto result_reorder_desc = mkldnn_emitter->build_memory_descriptor(
                        weights_shape_groups, args[1].get_element_type(), weights_optimized_format);

                    auto weights_desc = mkldnn::memory::desc(
                        mkldnn::memory::dims(weights_shape_groups.begin(),
                                             weights_shape_groups.end()),
                        mkldnn_utils::get_mkldnn_data_type(args[1].get_element_type()),
                        weights_optimized_format);

                    auto prim_indices = mkldnn_emitter->build_group_convolution_forward(
                        input_reorder_desc, // weights
                        input_data_desc,
                        weights_desc,
                        result_reorder_desc,
                        result_desc,
                        convolution->get_window_movement_strides(),
                        window_dilation_strides_adjusted,
                        padding_below,
                        padding_above);

                    size_t reorder_index = prim_indices.first;
                    auto& reorder_deps = mkldnn_emitter->get_primitive_deps(reorder_index);
                    size_t conv_index = prim_indices.second;
                    auto& deps = mkldnn_emitter->get_primitive_deps(conv_index);

                    auto functor =
                        [&, conv_index, reorder_index, ws_buf_index](CPURuntimeContext* ctx) {

                            // reorder
                            cpu::mkldnn_utils::set_memory_ptr(ctx, reorder_deps[0], arg1_tensor);
                            cpu::mkldnn_utils::set_memory_ptr(
                                ctx, reorder_deps[1], ctx->mkldnn_workspaces[ws_buf_index]);
                            cpu::mkldnn_utils::mkldnn_invoke_primitive(ctx, reorder_index);

                            // group convolution
                            cpu::mkldnn_utils::set_memory_ptr(ctx, deps[0], arg0_tensor);
                            cpu::mkldnn_utils::set_memory_ptr(
                                ctx, deps[1], ctx->mkldnn_workspaces[ws_buf_index]);
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

            REGISTER_OP_BUILDER(Convolution);
            REGISTER_OP_BUILDER(ConvolutionRelu);
            REGISTER_OP_BUILDER(ConvolutionBias);
            REGISTER_OP_BUILDER(ConvolutionBiasAdd);
            REGISTER_OP_BUILDER(ConvolutionBackpropData);
            REGISTER_OP_BUILDER(ConvolutionBackpropFilters);
            REGISTER_OP_BUILDER(ConvolutionBiasBackpropFiltersBias);
            REGISTER_OP_BUILDER(GroupConvolution);
        }
    }
}
