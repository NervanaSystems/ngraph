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

#include "cpu_mkldnn_primitive_build.hpp"

#include "ngraph/op/add.hpp"
#include "ngraph/op/avg_pool.hpp"
#include "ngraph/op/batch_norm.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/convert.hpp"
#include "ngraph/op/convolution.hpp"
#include "ngraph/op/dequantize.hpp"
#include "ngraph/op/dequantize.hpp"
#include "ngraph/op/experimental/quantized_avg_pool.hpp"
#include "ngraph/op/experimental/quantized_avg_pool.hpp"
#include "ngraph/op/experimental/quantized_concat.hpp"
#include "ngraph/op/experimental/quantized_conv.hpp"
#include "ngraph/op/experimental/quantized_conv_bias.hpp"
#include "ngraph/op/experimental/quantized_conv_relu.hpp"
#include "ngraph/op/experimental/quantized_max_pool.hpp"
#include "ngraph/op/experimental/quantized_max_pool.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/op/lrn.hpp"
#include "ngraph/op/max_pool.hpp"
#include "ngraph/op/quantize.hpp"
#include "ngraph/op/relu.hpp"
#include "ngraph/op/replace_slice.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/op/softmax.hpp"
#include "ngraph/runtime/cpu/cpu_executor.hpp"
#include "ngraph/runtime/cpu/cpu_tensor_view_wrapper.hpp"
#include "ngraph/runtime/cpu/mkldnn_emitter.hpp"
#include "ngraph/runtime/cpu/mkldnn_utils.hpp"
#include "ngraph/runtime/cpu/op/convert_layout.hpp"
#include "ngraph/runtime/cpu/op/lstm.hpp"
#include "ngraph/runtime/cpu/op/max_pool_with_indices.hpp"
#include "ngraph/runtime/cpu/op/rnn.hpp"
#include "ngraph/runtime/cpu/op/sigmoid.hpp"
#include "ngraph/runtime/cpu/op/update_slice.hpp"

using namespace ngraph;
using namespace ngraph::op;
using namespace ngraph::runtime::cpu;

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            namespace pass
            {
                // The following functions build the MKLDNN primitive for each type of nGraph Node.

                template <>
                size_t MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(Add)
                {
                    std::vector<float> scale_vector(2, 1);
                    std::vector<mkldnn::memory::primitive_desc> inputs_pd;

                    auto input0_data_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto input1_data_desc = mkldnn_utils::get_input_mkldnn_md(node, 1);
                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);
                    inputs_pd.push_back(mkldnn::memory::primitive_desc(
                        input0_data_desc, executor::global_cpu_engine));
                    inputs_pd.push_back(mkldnn::memory::primitive_desc(
                        input1_data_desc, executor::global_cpu_engine));

                    return mkldnn_emitter.build_elementwise_add(
                        input0_data_desc, input1_data_desc, result_desc, scale_vector, inputs_pd);
                }

                template <>
                size_t MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(Lstm)
                {
                    return mkldnn_emitter.build_rnn<Lstm>(node);
                }

                template <>
                size_t MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(Rnn)
                {
                    return mkldnn_emitter.build_rnn<Rnn>(node);
                }

                template <>
                size_t MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(BatchNormTraining)
                {
                    return mkldnn_emitter.build_batch_norm_primitive<BatchNormInference>(
                        node, false /*Append relu*/, true /*Training*/);
                }

                template <>
                size_t MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(BatchNormInference)
                {
                    return mkldnn_emitter.build_batch_norm_primitive<BatchNormInference>(
                        node, false /*Append relu*/, false /*Training*/);
                }

                template <>
                size_t MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(BatchNormTrainingRelu)
                {
                    return mkldnn_emitter.build_batch_norm_primitive<BatchNormTrainingRelu>(
                        node, true /*Append relu*/, true /*Training*/);
                }

                template <>
                size_t MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(BatchNormInferenceRelu)
                {
                    return mkldnn_emitter.build_batch_norm_primitive<BatchNormInferenceRelu>(
                        node, true /*Append relu*/, false /*Training*/);
                }

                template <>
                size_t MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(BatchNormTrainingBackprop)
                {
                    const auto& args = node->get_inputs();
                    auto weights_shape =
                        Shape{2, args[0].get_tensor().get_tensor_layout()->get_size()};
                    auto weights_desc = mkldnn_emitter.build_memory_descriptor(
                        weights_shape, args[0].get_element_type(), mkldnn::memory::format::nc);
                    auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 2);
                    auto mean_desc = mkldnn_utils::get_input_mkldnn_md(node, 3);
                    auto variance_desc = mkldnn_utils::get_input_mkldnn_md(node, 4);
                    auto delta_desc = mkldnn_utils::get_input_mkldnn_md(node, 5);
                    auto dinput_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);
                    auto dweights_desc = mkldnn_emitter.build_memory_descriptor(
                        weights_shape, args[0].get_element_type(), mkldnn::memory::format::nc);

                    const auto* batchnorm = static_cast<const BatchNormTrainingBackprop*>(node);
                    return mkldnn_emitter.build_batchnorm_backward(weights_desc,
                                                                   input_desc,
                                                                   mean_desc,
                                                                   variance_desc,
                                                                   delta_desc,
                                                                   dinput_desc,
                                                                   dweights_desc,
                                                                   batchnorm->get_eps_value());
                }

                template <>
                size_t MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(Concat)
                {
                    std::vector<mkldnn::memory::desc> inputs_data_desc;
                    for (size_t i = 0, end = node->get_inputs().size(); i < end; i++)
                    {
                        inputs_data_desc.push_back(mkldnn_utils::get_input_mkldnn_md(node, i));
                    }

                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);

                    size_t concat_dim =
                        (static_cast<const Concat*>(node))->get_concatenation_axis();
                    return mkldnn_emitter.build_concat(inputs_data_desc, result_desc, concat_dim);
                }

                template <>
                size_t MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(LRN)
                {
                    auto input_data_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);

                    const auto* lrn = static_cast<const LRN*>(node);

                    return mkldnn_emitter.build_lrn_forward(input_data_desc,
                                                            result_desc,
                                                            static_cast<float>(lrn->get_alpha()),
                                                            static_cast<float>(lrn->get_beta()),
                                                            static_cast<float>(lrn->get_bias()),
                                                            static_cast<int>(lrn->get_nsize()));
                }

                template <>
                size_t MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(Slice)
                {
                    const auto& out = node->get_outputs();
                    const Slice* slice = static_cast<const Slice*>(node);
                    auto out_shape = out[0].get_shape();
                    auto lower_bounds = slice->get_lower_bounds();
                    auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);

                    return mkldnn_emitter.build_slice(
                        input_desc, result_desc, lower_bounds, out_shape);
                }

                template <>
                size_t MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(ConvolutionRelu)
                {
                    return mkldnn_emitter.build_convolution<ConvolutionRelu>(node);
                }

                template <>
                size_t MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(QuantizedConvolutionRelu)
                {
                    return mkldnn_emitter.build_convolution<QuantizedConvolutionRelu>(node);
                }

                template <>
                size_t MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(QuantizedConvolution)
                {
                    return mkldnn_emitter.build_convolution<QuantizedConvolution>(node);
                }

                template <>
                size_t MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(GroupConvolution)
                {
                    Strides window_dilation_strides_adjusted;
                    auto convolution = static_cast<const ngraph::op::GroupConvolution*>(node);
                    for (size_t s : convolution->get_window_dilation_strides())
                    {
                        window_dilation_strides_adjusted.push_back(s - 1);
                    }

                    auto input_data_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto weights_desc = mkldnn_utils::get_input_mkldnn_md(node, 1);
                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);
                    auto padding_below = convolution->get_padding_below();
                    auto padding_above = convolution->get_padding_above();
                    auto filter_strides = convolution->get_window_movement_strides();

                    return mkldnn_emitter.build_convolution_forward(
                        input_data_desc,
                        weights_desc,
                        result_desc,
                        filter_strides,
                        window_dilation_strides_adjusted,
                        padding_below,
                        padding_above);
                }

                template <>
                size_t MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(GroupConvolutionBias)
                {
                    Strides window_dilation_strides_adjusted;
                    auto convolution = static_cast<const ngraph::op::GroupConvolutionBias*>(node);
                    for (size_t s : convolution->get_window_dilation_strides())
                    {
                        window_dilation_strides_adjusted.push_back(s - 1);
                    }

                    auto input_data_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto weights_desc = mkldnn_utils::get_input_mkldnn_md(node, 1);
                    auto bias_desc = mkldnn_utils::get_input_mkldnn_md(node, 2);
                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);
                    auto padding_below = convolution->get_padding_below();
                    auto padding_above = convolution->get_padding_above();
                    auto filter_strides = convolution->get_window_movement_strides();

                    const float ops_scale = 1.f;
                    const float ops_alpha = -0.f; // relu negative slope
                    const float ops_beta = 0.f;

                    mkldnn::post_ops ops;
                    if (convolution->with_relu())
                    {
                        ops.append_eltwise(
                            ops_scale, mkldnn::algorithm::eltwise_relu, ops_alpha, ops_beta);
                    }

                    return mkldnn_emitter.build_convolution_forward(
                        input_data_desc,
                        weights_desc,
                        bias_desc,
                        result_desc,
                        filter_strides,
                        window_dilation_strides_adjusted,
                        padding_below,
                        padding_above,
                        ops);
                }

                template <>
                size_t MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(Convolution)
                {
                    return mkldnn_emitter.build_convolution<Convolution>(node);
                }

                template <typename OpTy>
                size_t build_convolution_backward(MKLDNNEmitter& mkldnn_emitter,
                                                  const ngraph::Node* node)
                {
                    auto convolution = static_cast<const OpTy*>(node);

                    Strides window_dilation_strides_adjusted;

                    for (size_t s : convolution->get_window_dilation_strides_forward())
                    {
                        window_dilation_strides_adjusted.push_back(s - 1);
                    }

                    auto arg0_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto arg1_desc = mkldnn_utils::get_input_mkldnn_md(node, 1);
                    auto out0_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);

                    if (std::is_same<OpTy, ngraph::op::ConvolutionBackpropData>())
                    {
                        // MKLDNN relies on named formats for kernel selection
                        if (arg0_desc.data.format == mkldnn_nchw)
                        {
                            arg0_desc.data.format = mkldnn_oihw;
                        }
                        if (arg0_desc.data.format == mkldnn_ncdhw)
                        {
                            arg0_desc.data.format = mkldnn_oidhw;
                        }

                        return mkldnn_emitter.build_convolution_backward_data(
                            arg0_desc,
                            arg1_desc,
                            out0_desc,
                            convolution->get_window_movement_strides_forward(),
                            window_dilation_strides_adjusted,
                            convolution->get_padding_below_forward(),
                            convolution->get_padding_above_forward());
                    }
                    if (std::is_same<OpTy, ngraph::op::ConvolutionBackpropFilters>())
                    {
                        return mkldnn_emitter.build_convolution_backward_weights(
                            arg0_desc,
                            arg1_desc,
                            out0_desc,
                            convolution->get_window_movement_strides_forward(),
                            window_dilation_strides_adjusted,
                            convolution->get_padding_below_forward(),
                            convolution->get_padding_above_forward());
                    }
                    if (std::is_same<OpTy, ngraph::op::ConvolutionBiasBackpropFiltersBias>())
                    {
                        auto out1_desc = mkldnn_utils::get_output_mkldnn_md(node, 1);
                        return mkldnn_emitter.build_convolution_backward_weights_bias(
                            arg0_desc,
                            arg1_desc,
                            out0_desc,
                            out1_desc,
                            convolution->get_window_movement_strides_forward(),
                            window_dilation_strides_adjusted,
                            convolution->get_padding_below_forward(),
                            convolution->get_padding_above_forward());
                    }

                    throw ngraph_error(std::string("Unknown op ") + convolution->get_name());
                }

                template <>
                size_t MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(ConvolutionBackpropFilters)
                {
                    return build_convolution_backward<ConvolutionBackpropFilters>(mkldnn_emitter,
                                                                                  node);
                }

                template <>
                size_t MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(ConvolutionBackpropData)
                {
                    return build_convolution_backward<ConvolutionBackpropData>(mkldnn_emitter,
                                                                               node);
                }

                template <>
                size_t MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(QuantizedConvolutionBias)
                {
                    return mkldnn_emitter.build_convolution<QuantizedConvolutionBias>(node);
                }

                template <>
                size_t MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(QuantizedConvolutionBiasAdd)
                {
                    return mkldnn_emitter.build_convolution<QuantizedConvolutionBiasAdd>(node);
                }

                template <>
                size_t MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(
                    QuantizedConvolutionBiasSignedAdd)
                {
                    return mkldnn_emitter.build_convolution<QuantizedConvolutionBiasSignedAdd>(
                        node);
                }

                template <>
                size_t MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(ConvolutionBias)
                {
                    return mkldnn_emitter.build_convolution<ConvolutionBias>(node);
                }

                template <>
                size_t MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(ConvolutionBiasAdd)
                {
                    return mkldnn_emitter.build_convolution<ConvolutionBiasAdd>(node);
                }

                template <>
                size_t MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(ConvolutionAdd)
                {
                    return mkldnn_emitter.build_convolution<ConvolutionAdd>(node);
                }

                template <>
                size_t MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(
                    ConvolutionBiasBackpropFiltersBias)
                {
                    return build_convolution_backward<ConvolutionBiasBackpropFiltersBias>(
                        mkldnn_emitter, node);
                }

                template <>
                size_t MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(MaxPool)
                {
                    auto max_pool = static_cast<const ngraph::op::MaxPool*>(node);
                    auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);

                    return mkldnn_emitter.build_pooling_forward(
                        mkldnn::algorithm::pooling_max,
                        input_desc,
                        result_desc,
                        max_pool->get_window_movement_strides(),
                        max_pool->get_window_shape(),
                        max_pool->get_padding_below(),
                        max_pool->get_padding_above());
                }

                template <>
                size_t MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(QuantizedMaxPool)
                {
                    return mkldnn_emitter.build_quantized_max_pool(node);
                }

                template <>
                size_t MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(QuantizedAvgPool)
                {
                    return mkldnn_emitter.build_quantized_avg_pool(node);
                }

                template <>
                size_t MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(MaxPoolWithIndices)
                {
                    auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);
                    auto max_pool = static_cast<const ngraph::op::MaxPoolWithIndices*>(node);

                    return mkldnn_emitter.build_max_pooling_with_indices_forward(
                        mkldnn::algorithm::pooling_max,
                        input_desc,
                        result_desc,
                        max_pool->get_window_movement_strides(),
                        max_pool->get_window_shape(),
                        max_pool->get_padding_below(),
                        max_pool->get_padding_above());
                }

                template <>
                size_t MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(AvgPool)
                {
                    auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);
                    auto avg_pool = static_cast<const ngraph::op::AvgPool*>(node);

                    return mkldnn_emitter.build_pooling_forward(
                        (avg_pool->get_include_padding_in_avg_computation()
                             ? mkldnn::algorithm::pooling_avg_include_padding
                             : mkldnn::algorithm::pooling_avg_exclude_padding),
                        input_desc,
                        result_desc,
                        avg_pool->get_window_movement_strides(),
                        avg_pool->get_window_shape(),
                        avg_pool->get_padding_below(),
                        avg_pool->get_padding_above());
                }

                template <>
                size_t MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(AvgPoolBackprop)
                {
                    auto diff_dst_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto diff_src_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);
                    auto apb = static_cast<const ngraph::op::AvgPoolBackprop*>(node);

                    return mkldnn_emitter.build_pooling_backward(
                        (apb->get_include_padding_in_avg_computation()
                             ? mkldnn::algorithm::pooling_avg_include_padding
                             : mkldnn::algorithm::pooling_avg_exclude_padding),
                        diff_dst_desc,
                        diff_src_desc,
                        apb->get_window_movement_strides(),
                        apb->get_window_shape(),
                        apb->get_padding_below(),
                        apb->get_padding_above());
                }

                template <>
                size_t MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(MaxPoolBackprop)
                {
                    auto fprop_src_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto diff_dst_desc = mkldnn_utils::get_input_mkldnn_md(node, 1);
                    auto diff_src_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);
                    auto mpb = static_cast<const ngraph::op::MaxPoolBackprop*>(node);

                    return mkldnn_emitter.build_max_pooling_backward(
                        mkldnn::algorithm::pooling_max,
                        fprop_src_desc,
                        diff_dst_desc,
                        diff_src_desc,
                        mpb->get_window_movement_strides(),
                        mpb->get_window_shape(),
                        mpb->get_padding_below(),
                        mpb->get_padding_above());
                }

                template <>
                size_t MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(MaxPoolWithIndicesBackprop)
                {
                    auto diff_dst_desc = mkldnn_utils::get_input_mkldnn_md(node, 1);
                    auto diff_src_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);
                    auto mpb = static_cast<const ngraph::op::MaxPoolWithIndicesBackprop*>(node);

                    return mkldnn_emitter.build_max_pooling_with_indices_backward(
                        mkldnn::algorithm::pooling_max,
                        diff_dst_desc,
                        diff_src_desc,
                        mpb->get_window_movement_strides(),
                        mpb->get_window_shape(),
                        mpb->get_padding_below(),
                        mpb->get_padding_above());
                }

                template <>
                size_t MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(
                    ngraph::runtime::cpu::op::ConvertLayout)
                {
                    const auto& args = node->get_inputs();
                    auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);

                    // This is a special case to handle nchw(oihw) to goihw/Goihw16g/Goihw8g for
                    // GroupConvolution's weights.
                    if (input_desc.data.format == mkldnn_nchw &&
                        result_desc.data.format == mkldnn_goihw)
                    {
                        input_desc = result_desc;
                    }
                    else if (input_desc.data.format == mkldnn_nchw &&
                             input_desc.data.ndims == 4 /*nchw*/ &&
                             result_desc.data.ndims == 5 /*Goihw16g/Goihw8g/etc*/ &&
                             node->get_users().size() == 1)
                    {
                        Shape weights_shape_groups;
                        if (auto gconv = std::dynamic_pointer_cast<ngraph::op::GroupConvolution>(
                                node->get_users()[0]))
                        {
                            weights_shape_groups = gconv->get_weights_dimensions();
                        }
                        else if (auto gconvb =
                                     std::dynamic_pointer_cast<ngraph::op::GroupConvolutionBias>(
                                         node->get_users()[0]))
                        {
                            weights_shape_groups = gconvb->get_weights_dimensions();
                        }
                        else
                        {
                            throw ngraph_error(
                                "Incompatible input/output shape in ConvertLayout op");
                        }
                        input_desc = mkldnn::memory::desc(
                            mkldnn::memory::dims(weights_shape_groups.begin(),
                                                 weights_shape_groups.end()),
                            mkldnn_utils::get_mkldnn_data_type(args[0].get_element_type()),
                            mkldnn::memory::format::goihw);
                    }

                    return mkldnn_emitter.build_reorder(input_desc, result_desc);
                }

                template <>
                size_t MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(ReluBackprop)
                {
                    auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto delta_desc = mkldnn_utils::get_input_mkldnn_md(node, 1);
                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);

                    return mkldnn_emitter.build_relu_backward(input_desc, delta_desc, result_desc);
                }

                template <>
                size_t MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(Relu)
                {
                    auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);
                    return mkldnn_emitter.build_relu_forward(input_desc, result_desc);
                }

                template <>
                size_t MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(LeakyRelu)
                {
                    auto leaky_relu_node = static_cast<const ngraph::op::LeakyRelu*>(node);
                    float alpha = leaky_relu_node->get_alpha();
                    auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);

                    return mkldnn_emitter.build_leaky_relu(input_desc, result_desc, alpha);
                }

                template <>
                size_t MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(BoundedRelu)
                {
                    auto bounded_relu_node = static_cast<const ngraph::op::BoundedRelu*>(node);
                    float alpha = bounded_relu_node->get_alpha();
                    auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);

                    return mkldnn_emitter.build_bounded_relu(input_desc, result_desc, alpha);
                }

                template <>
                size_t MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(Sigmoid)
                {
                    auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);
                    return mkldnn_emitter.build_sigmoid_forward(input_desc, result_desc);
                }

                template <>
                size_t MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(SigmoidBackprop)
                {
                    auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto delta_desc = mkldnn_utils::get_input_mkldnn_md(node, 1);
                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);

                    return mkldnn_emitter.build_sigmoid_backward(
                        input_desc, delta_desc, result_desc);
                }

                template <>
                size_t MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(Softmax)
                {
                    auto softmax = static_cast<const ngraph::op::Softmax*>(node);

                    if (softmax->get_axes().size() != 1)
                    {
                        throw ngraph_error("MKLDNN supports softmax only across single axis");
                    }

                    int softmax_axis = static_cast<int>(*(softmax->get_axes().begin()));
                    auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);

                    return mkldnn_emitter.build_softmax_forward(
                        input_desc, result_desc, softmax_axis);
                }

                template <>
                size_t MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(Dequantize)
                {
                    auto input_data_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);
                    return mkldnn_emitter.build_dequantization(node, input_data_desc, result_desc);
                }

                template <>
                size_t MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(Quantize)
                {
                    auto input_data_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);
                    auto quantize = static_cast<const ngraph::op::Quantize*>(node);
                    auto scale_const_op =
                        std::dynamic_pointer_cast<Constant>(quantize->get_argument(1));

                    if (scale_const_op == nullptr)
                    {
                        throw ngraph_error("Quantize scale must be a constant");
                    }

                    auto scale = scale_const_op->get_vector<float>();
                    std::vector<float> scales;
                    scales.push_back(1.0 / scale[0]);

                    return mkldnn_emitter.build_quantize_reorder(
                        input_data_desc, result_desc, scales);
                }

                template <>
                size_t MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(QuantizedConcat)
                {
                    int args_size = node->get_inputs().size();

                    std::vector<mkldnn::memory::desc> inputs_data_desc;
                    for (size_t i = 0; i < args_size; i++)
                    {
                        inputs_data_desc.push_back(mkldnn_utils::get_input_mkldnn_md(node, i));
                    }

                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);

                    size_t concat_dim =
                        (static_cast<const QuantizedConcat*>(node))->get_concatenation_axis();
                    return mkldnn_emitter.build_concat(inputs_data_desc, result_desc, concat_dim);
                }
            }
        }
    }
}

using namespace ngraph::runtime::cpu::pass;

#define TI(x) std::type_index(typeid(x))

static const PrimitiveBuildOpMap prim_build_dispatcher{
    {TI(Add), &MKLDNNPrimitiveBuildPass::build_primitive<Add>},
    {TI(Concat), &MKLDNNPrimitiveBuildPass::build_primitive<Concat>},
    {TI(Convert), &MKLDNNPrimitiveBuildPass::build_primitive<Convert>},
    {TI(runtime::cpu::op::ConvertLayout),
     &MKLDNNPrimitiveBuildPass::build_primitive<runtime::cpu::op::ConvertLayout>},
    {TI(AvgPool), &MKLDNNPrimitiveBuildPass::build_primitive<AvgPool>},
    {TI(AvgPoolBackprop), &MKLDNNPrimitiveBuildPass::build_primitive<AvgPoolBackprop>},
    {TI(BatchNormTraining), &MKLDNNPrimitiveBuildPass::build_primitive<BatchNormTraining>},
    {TI(BatchNormInference), &MKLDNNPrimitiveBuildPass::build_primitive<BatchNormInference>},
    {TI(BoundedRelu), &MKLDNNPrimitiveBuildPass::build_primitive<BoundedRelu>},
    {TI(BatchNormTrainingBackprop),
     &MKLDNNPrimitiveBuildPass::build_primitive<BatchNormTrainingBackprop>},
    {TI(Convolution), &MKLDNNPrimitiveBuildPass::build_primitive<Convolution>},
    {TI(GroupConvolution), &MKLDNNPrimitiveBuildPass::build_primitive<GroupConvolution>},
    {TI(ConvolutionRelu), &MKLDNNPrimitiveBuildPass::build_primitive<ConvolutionRelu>},
    {TI(ConvolutionBiasAdd), &MKLDNNPrimitiveBuildPass::build_primitive<ConvolutionBiasAdd>},
    {TI(BatchNormTrainingRelu), &MKLDNNPrimitiveBuildPass::build_primitive<BatchNormTrainingRelu>},
    {TI(BatchNormInferenceRelu),
     &MKLDNNPrimitiveBuildPass::build_primitive<BatchNormInferenceRelu>},
    {TI(ConvolutionBackpropData),
     &MKLDNNPrimitiveBuildPass::build_primitive<ConvolutionBackpropData>},
    {TI(ConvolutionBackpropFilters),
     &MKLDNNPrimitiveBuildPass::build_primitive<ConvolutionBackpropFilters>},
    {TI(MaxPool), &MKLDNNPrimitiveBuildPass::build_primitive<MaxPool>},
    {TI(MaxPoolWithIndices), &MKLDNNPrimitiveBuildPass::build_primitive<MaxPoolWithIndices>},
    {TI(MaxPoolBackprop), &MKLDNNPrimitiveBuildPass::build_primitive<MaxPoolBackprop>},
    {TI(MaxPoolWithIndicesBackprop),
     &MKLDNNPrimitiveBuildPass::build_primitive<MaxPoolWithIndicesBackprop>},
    {TI(ConvolutionBias), &MKLDNNPrimitiveBuildPass::build_primitive<ConvolutionBias>},
    {TI(QuantizedConvolution), &MKLDNNPrimitiveBuildPass::build_primitive<QuantizedConvolution>},
    {TI(ConvolutionBiasBackpropFiltersBias),
     &MKLDNNPrimitiveBuildPass::build_primitive<ConvolutionBiasBackpropFiltersBias>},
    {TI(LRN), &MKLDNNPrimitiveBuildPass::build_primitive<LRN>},
    {TI(Relu), &MKLDNNPrimitiveBuildPass::build_primitive<Relu>},
    {TI(ReluBackprop), &MKLDNNPrimitiveBuildPass::build_primitive<ReluBackprop>},
    {TI(LeakyRelu), &MKLDNNPrimitiveBuildPass::build_primitive<LeakyRelu>},
    {TI(Sigmoid), &MKLDNNPrimitiveBuildPass::build_primitive<Sigmoid>},
    {TI(SigmoidBackprop), &MKLDNNPrimitiveBuildPass::build_primitive<SigmoidBackprop>},
    {TI(Lstm), &MKLDNNPrimitiveBuildPass::build_primitive<Lstm>},
    {TI(Rnn), &MKLDNNPrimitiveBuildPass::build_primitive<Rnn>},
    {TI(QuantizedMaxPool), &MKLDNNPrimitiveBuildPass::build_primitive<QuantizedMaxPool>},
    {TI(QuantizedAvgPool), &MKLDNNPrimitiveBuildPass::build_primitive<QuantizedAvgPool>},
    {TI(Softmax), &MKLDNNPrimitiveBuildPass::build_primitive<Softmax>},
    {TI(Slice), &MKLDNNPrimitiveBuildPass::build_primitive<Slice>},
    {TI(ReplaceSlice), &MKLDNNPrimitiveBuildPass::build_primitive<ReplaceSlice>},
    {TI(UpdateSlice), &MKLDNNPrimitiveBuildPass::build_primitive<UpdateSlice>},
    {TI(ConvolutionAdd), &MKLDNNPrimitiveBuildPass::build_primitive<ConvolutionAdd>},
    {TI(QuantizedConvolutionRelu),
     &MKLDNNPrimitiveBuildPass::build_primitive<QuantizedConvolutionRelu>},
    {TI(QuantizedConvolutionBias),
     &MKLDNNPrimitiveBuildPass::build_primitive<QuantizedConvolutionBias>},
    {TI(QuantizedConvolutionBiasAdd),
     &MKLDNNPrimitiveBuildPass::build_primitive<QuantizedConvolutionBiasAdd>},
    {TI(QuantizedConvolutionBiasSignedAdd),
     &MKLDNNPrimitiveBuildPass::build_primitive<QuantizedConvolutionBiasSignedAdd>},
    {TI(GroupConvolutionBias), &MKLDNNPrimitiveBuildPass::build_primitive<GroupConvolutionBias>},
    {TI(Quantize), &MKLDNNPrimitiveBuildPass::build_primitive<Quantize>},
    {TI(Dequantize), &MKLDNNPrimitiveBuildPass::build_primitive<Dequantize>},
    {TI(QuantizedConcat), &MKLDNNPrimitiveBuildPass::build_primitive<QuantizedConcat>},
    {TI(GetOutputElement), &MKLDNNPrimitiveBuildPass::build_primitive<GetOutputElement>},
};

bool MKLDNNPrimitiveBuildPass::run_on_call_graph(const std::list<std::shared_ptr<Node>>& nodes)
{
    for (const auto& shp_node : nodes)
    {
        Node* node = shp_node.get();

        if (mkldnn_utils::use_mkldnn_kernel(node))
        {
            auto handler = prim_build_dispatcher.find(TI(*node));
            NGRAPH_ASSERT(handler != prim_build_dispatcher.end())
                << "Unsupported node '" << node->description() << "' in MKLDNNPrimitiveBuildPass";

            size_t primitive_idx = handler->second(m_mkldnn_emitter, node);
            m_node_primitive_idx_map[node] = primitive_idx;
        }
    }

    return false;
}

#undef TI
