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

#pragma once

#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include <mkldnn.hpp>

#include "ngraph/coordinate_diff.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/convolution.hpp"
#include "ngraph/op/experimental/quantized_conv.hpp"
#include "ngraph/op/experimental/quantized_conv_bias.hpp"
#include "ngraph/op/experimental/quantized_conv_relu.hpp"
#include "ngraph/runtime/cpu/cpu_executor.hpp"
#include "ngraph/runtime/cpu/cpu_tensor_view_wrapper.hpp"
#include "ngraph/runtime/cpu/mkldnn_utils.hpp"
#include "ngraph/runtime/cpu/op/bounded_relu.hpp"
#include "ngraph/runtime/cpu/op/conv_add.hpp"
#include "ngraph/runtime/cpu/op/conv_bias.hpp"
#include "ngraph/runtime/cpu/op/conv_relu.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/strides.hpp"
#include "ngraph/type/element_type.hpp"
#include "ngraph/util.hpp"

#define MKLDNN_DIMS(X) mkldnn::memory::dims(X.begin(), X.end())

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            class CPU_ExternalFunction;
            class TensorViewWrapper;

            class MKLDNNWorkspace
            {
            public:
                MKLDNNWorkspace(size_t size) { buf = reinterpret_cast<char*>(ngraph_malloc(size)); }
                ~MKLDNNWorkspace() { ngraph_free(buf); }
                char* buf;

                MKLDNNWorkspace(const MKLDNNWorkspace&) = delete;
                MKLDNNWorkspace(MKLDNNWorkspace&&) = delete;
                MKLDNNWorkspace& operator=(const MKLDNNWorkspace&) = delete;
            };

            class MKLDNNEmitter
            {
            public:
                MKLDNNEmitter() {}
                ~MKLDNNEmitter();

                const std::vector<mkldnn::primitive*>& get_mkldnn_primitives() const;
                const std::vector<char*>& get_mkldnn_workspaces();

                size_t insert_primitive(mkldnn::primitive* primitive);
                size_t insert_workspace(std::unique_ptr<MKLDNNWorkspace>& workspace);
                const std::vector<size_t>& get_primitive_deps(size_t index) const;

                // TODO(jmenon): Get rid of TensorViewWrappers at some point
                mkldnn::memory::desc build_memory_descriptor(const TensorViewWrapper& tvw,
                                                             mkldnn::memory::format fmt) const;
                mkldnn::memory::desc build_memory_descriptor(const Shape& shape,
                                                             const ngraph::element::Type& et,
                                                             mkldnn::memory::format fmt) const;
                mkldnn::memory::desc
                    build_blocked_memory_descriptor(const mkldnn::memory::dims& dim,
                                                    const mkldnn::memory::dims& strides,
                                                    mkldnn::memory::data_type dtype) const;
                size_t build_memory_primitive(const mkldnn::memory::desc& desc);

                size_t build_convolution_forward(const mkldnn::memory::desc& input_data_desc,
                                                 const mkldnn::memory::desc& weights_desc,
                                                 const mkldnn::memory::desc& result_desc,
                                                 const ngraph::Strides& strides,
                                                 const ngraph::Strides& dilation_strides,
                                                 const ngraph::CoordinateDiff& padding_below,
                                                 const ngraph::CoordinateDiff& padding_above,
                                                 const mkldnn::post_ops& pops = mkldnn::post_ops());

                /**
                 * Convolution + bias forward
                 */
                size_t build_convolution_forward(const mkldnn::memory::desc& input_data_desc,
                                                 const mkldnn::memory::desc& weights_desc,
                                                 const mkldnn::memory::desc& bias_desc,
                                                 const mkldnn::memory::desc& result_desc,
                                                 const ngraph::Strides& strides,
                                                 const ngraph::Strides& dilation_strides,
                                                 const ngraph::CoordinateDiff& padding_below,
                                                 const ngraph::CoordinateDiff& padding_above,
                                                 const mkldnn::post_ops& pops = mkldnn::post_ops());

                size_t build_quantized_convolution_forward(
                    const mkldnn::memory::desc& input_data_desc,
                    const mkldnn::memory::desc& weights_desc,
                    const mkldnn::memory::desc& result_desc,
                    const ngraph::Strides& strides,
                    const ngraph::Strides& dilation_strides,
                    const ngraph::CoordinateDiff& padding_below,
                    const ngraph::CoordinateDiff& padding_above,
                    const float scale,
                    const mkldnn::post_ops& pops = mkldnn::post_ops());

                size_t build_quantized_convolution_forward(
                    const mkldnn::memory::desc& input_data_desc,
                    const mkldnn::memory::desc& weights_desc,
                    const mkldnn::memory::desc& bias_desc,
                    const mkldnn::memory::desc& result_desc,
                    const ngraph::Strides& strides,
                    const ngraph::Strides& dilation_strides,
                    const ngraph::CoordinateDiff& padding_below,
                    const ngraph::CoordinateDiff& padding_above,
                    const float scale,
                    const mkldnn::post_ops& pops = mkldnn::post_ops());

                // TODO (nbpatel) Templatize the return type when we have double scales
                template <typename OP>
                std::vector<float> extract_scale_value(const ngraph::Node* node, int index)
                {
                    auto qc = static_cast<const OP*>(node);
                    std::vector<float> scale_val = {1.0f};
                    auto scale_const_op =
                        std::dynamic_pointer_cast<ngraph::op::Constant>(qc->get_arguments()[index]);
                    if (scale_const_op != nullptr)
                    {
                        scale_val = scale_const_op->template get_vector<float>();
                    }

                    return scale_val;
                }

                template <typename OP>
                size_t build_convolution(const ngraph::Node* node,
                                         const std::vector<TensorViewWrapper>& args,
                                         const std::vector<TensorViewWrapper>& out)
                {
                    auto convolution = static_cast<const OP*>(node);

                    // For dilation, MKLDNN wants to know how many elements to insert between, not how far
                    // apart to space the elements like nGraph. So we have to subtract 1 from each pos.
                    Strides window_dilation_strides_adjusted;

                    for (size_t s : convolution->get_window_dilation_strides())
                    {
                        window_dilation_strides_adjusted.push_back(s - 1);
                    }

                    auto data_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto weights_desc = mkldnn_utils::get_input_mkldnn_md(node, 1);

                    // MKLDNN relies on named formats for kernel selection
                    if (weights_desc.data.format == mkldnn_nchw)
                        weights_desc.data.format = mkldnn_oihw;
                    if (weights_desc.data.format == mkldnn_ncdhw)
                        weights_desc.data.format = mkldnn_oidhw;

                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);

                    mkldnn::post_ops ops;

                    if (std::is_same<OP, ngraph::op::ConvolutionBiasAdd>() ||
                        std::is_same<OP, ngraph::op::ConvolutionAdd>())
                    {
                        ops.append_sum(1.f);
                    }

                    if (std::is_same<OP, ngraph::op::QuantizedConvolutionBiasAdd>() ||
                        std::is_same<OP, ngraph::op::QuantizedConvolutionBiasSignedAdd>())
                    {
                        auto sum_scale_val =
                            extract_scale_value<ngraph::op::QuantizedConvolutionBiasAdd>(node, 5);
                        ops.append_sum(sum_scale_val[0]);
                    }

                    if (has_relu<OP>(node))
                    {
                        const float ops_scale = 1.f;
                        const float ops_alpha = -0.f; // relu negative slope
                        const float ops_beta = 0.f;
                        ops.append_eltwise(
                            ops_scale, mkldnn::algorithm::eltwise_relu, ops_alpha, ops_beta);
                    }

                    if (std::is_same<OP, ngraph::op::ConvolutionBias>() ||
                        std::is_same<OP, ngraph::op::ConvolutionBiasAdd>())
                    {
                        auto bias_desc = mkldnn_utils::get_input_mkldnn_md(node, 2);
                        return build_convolution_forward(data_desc,
                                                         weights_desc,
                                                         bias_desc,
                                                         result_desc,
                                                         convolution->get_window_movement_strides(),
                                                         window_dilation_strides_adjusted,
                                                         convolution->get_padding_below(),
                                                         convolution->get_padding_above(),
                                                         ops);
                    }
                    else if (std::is_same<OP, ngraph::op::QuantizedConvolution>() ||
                             std::is_same<OP, ngraph::op::QuantizedConvolutionRelu>())
                    {
                        auto scale_val = extract_scale_value<OP>(node, 2);
                        return build_quantized_convolution_forward(
                            data_desc,
                            weights_desc,
                            result_desc,
                            convolution->get_window_movement_strides(),
                            window_dilation_strides_adjusted,
                            convolution->get_padding_below(),
                            convolution->get_padding_above(),
                            scale_val[0],
                            ops);
                    }
                    else if (std::is_same<OP, ngraph::op::QuantizedConvolutionBias>() ||
                             std::is_same<OP, ngraph::op::QuantizedConvolutionBiasAdd>() ||
                             std::is_same<OP, ngraph::op::QuantizedConvolutionBiasSignedAdd>())
                    {
                        int index =
                            std::is_same<OP, ngraph::op::QuantizedConvolutionBias>() ? 3 : 4;
                        auto scale_val = extract_scale_value<OP>(node, index);
                        auto bias_desc = mkldnn_utils::get_input_mkldnn_md(node, 2);
                        return build_quantized_convolution_forward(
                            data_desc,
                            weights_desc,
                            bias_desc,
                            result_desc,
                            convolution->get_window_movement_strides(),
                            window_dilation_strides_adjusted,
                            convolution->get_padding_below(),
                            convolution->get_padding_above(),
                            scale_val[0],
                            ops);
                    }
                    else
                    {
                        return build_convolution_forward(data_desc,
                                                         weights_desc,
                                                         result_desc,
                                                         convolution->get_window_movement_strides(),
                                                         window_dilation_strides_adjusted,
                                                         convolution->get_padding_below(),
                                                         convolution->get_padding_above(),
                                                         ops);
                    }
                }

                mkldnn::memory::format query_convolution_forward_weight_format(
                    const mkldnn::memory::desc& input_data_desc,
                    const mkldnn::memory::desc& weights_desc_any,
                    const mkldnn::memory::desc& result_desc,
                    const ngraph::Strides& filter_strides,
                    const ngraph::Strides& window_dilation_strides_adjusted,
                    const ngraph::CoordinateDiff& padding_below,
                    const ngraph::CoordinateDiff& padding_above);

                size_t
                    build_convolution_backward_weights(const mkldnn::memory::desc& input_desc,
                                                       const mkldnn::memory::desc& delta_desc,
                                                       const mkldnn::memory::desc& result_desc,
                                                       const ngraph::Strides& strides,
                                                       const ngraph::Strides& dilation_strides,
                                                       const ngraph::CoordinateDiff& padding_below,
                                                       const ngraph::CoordinateDiff& padding_above);

                size_t build_convolution_backward_data(const mkldnn::memory::desc& weights_desc,
                                                       const mkldnn::memory::desc& delta_desc,
                                                       const mkldnn::memory::desc& result_desc,
                                                       const ngraph::Strides& strides,
                                                       const ngraph::Strides& dilation_strides,
                                                       const ngraph::CoordinateDiff& padding_below,
                                                       const ngraph::CoordinateDiff& padding_above);
                /**
                 * Convolution + bias backprop for weights and bias
                 */
                size_t build_convolution_backward_weights_bias(
                    const mkldnn::memory::desc& in_data_desc,
                    const mkldnn::memory::desc& in_delta_desc,
                    const mkldnn::memory::desc& out_weights_delta_desc,
                    const mkldnn::memory::desc& out_bias_delta_desc,
                    const ngraph::Strides& ng_strides,
                    const ngraph::Strides& ng_dilation_strides,
                    const ngraph::CoordinateDiff& ng_padding_below,
                    const ngraph::CoordinateDiff& ng_padding_above);

                template <typename OP>
                size_t build_convolution_backward(const ngraph::Node* node,
                                                  const std::vector<TensorViewWrapper>& args,
                                                  const std::vector<TensorViewWrapper>& out)
                {
                    auto convolution = static_cast<const OP*>(node);

                    Strides window_dilation_strides_adjusted;

                    for (size_t s : convolution->get_window_dilation_strides_forward())
                    {
                        window_dilation_strides_adjusted.push_back(s - 1);
                    }

                    auto arg0_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto arg1_desc = mkldnn_utils::get_input_mkldnn_md(node, 1);
                    auto out0_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);

                    if (std::is_same<OP, ngraph::op::ConvolutionBackpropData>())
                    {
                        // MKLDNN relies on named formats for kernel selection
                        if (arg0_desc.data.format == mkldnn_nchw)
                            arg0_desc.data.format = mkldnn_oihw;
                        if (arg0_desc.data.format == mkldnn_ncdhw)
                            arg0_desc.data.format = mkldnn_oidhw;

                        return build_convolution_backward_data(
                            arg0_desc,
                            arg1_desc,
                            out0_desc,
                            convolution->get_window_movement_strides_forward(),
                            window_dilation_strides_adjusted,
                            convolution->get_padding_below_forward(),
                            convolution->get_padding_above_forward());
                    }
                    if (std::is_same<OP, ngraph::op::ConvolutionBackpropFilters>())
                    {
                        return build_convolution_backward_weights(
                            arg0_desc,
                            arg1_desc,
                            out0_desc,
                            convolution->get_window_movement_strides_forward(),
                            window_dilation_strides_adjusted,
                            convolution->get_padding_below_forward(),
                            convolution->get_padding_above_forward());
                    }
                    if (std::is_same<OP, ngraph::op::ConvolutionBiasBackpropFiltersBias>())
                    {
                        auto out1_desc = mkldnn_utils::get_output_mkldnn_md(node, 1);
                        return build_convolution_backward_weights_bias(
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

                size_t build_pooling_forward(mkldnn::algorithm pooling_algorithm,
                                             const mkldnn::memory::desc& input_desc,
                                             const mkldnn::memory::desc& result_desc,
                                             const ngraph::Strides& window_strides,
                                             const ngraph::Shape& window_shape,
                                             const ngraph::Shape& padding_below,
                                             const ngraph::Shape& padding_above);

                size_t build_pooling_backward(mkldnn::algorithm pooling_algorithm,
                                              const mkldnn::memory::desc& diff_dst_desc,
                                              const mkldnn::memory::desc& diff_src_desc,
                                              const ngraph::Strides& window_strides,
                                              const ngraph::Shape& window_shape,
                                              const ngraph::Shape& padding_below,
                                              const ngraph::Shape& padding_above);

                size_t build_max_pooling_with_indices_forward(mkldnn::algorithm pooling_algorithm,
                                                              const mkldnn::memory::desc& src_desc,
                                                              const mkldnn::memory::desc& dst_desc,
                                                              const ngraph::Strides& window_strides,
                                                              const ngraph::Shape& window_shape,
                                                              const ngraph::Shape& padding_below,
                                                              const ngraph::Shape& padding_above);

                size_t build_max_pooling_backward(mkldnn::algorithm pooling_algorithm,
                                                  const mkldnn::memory::desc& fprop_src_desc,
                                                  const mkldnn::memory::desc& diff_dst_desc,
                                                  const mkldnn::memory::desc& diff_src_desc,
                                                  const ngraph::Strides& window_strides,
                                                  const ngraph::Shape& window_shape,
                                                  const ngraph::Shape& padding_below,
                                                  const ngraph::Shape& padding_above);

                size_t build_max_pooling_with_indices_backward(
                    mkldnn::algorithm pooling_algorithm,
                    const mkldnn::memory::desc& diff_dst_desc,
                    const mkldnn::memory::desc& diff_src_desc,
                    const ngraph::Strides& window_strides,
                    const ngraph::Shape& window_shape,
                    const ngraph::Shape& padding_below,
                    const ngraph::Shape& padding_above);

                size_t build_reorder(const mkldnn::memory::desc& input_desc,
                                     const mkldnn::memory::desc& result_desc);

                size_t build_lrn_forward(const mkldnn::memory::desc& input_desc,
                                         const mkldnn::memory::desc& result_desc,
                                         float alpha,
                                         float beta,
                                         float bias,
                                         int nsize);

                size_t build_relu_forward(const mkldnn::memory::desc& input_desc,
                                          const mkldnn::memory::desc& result_desc);

                size_t build_relu_backward(const mkldnn::memory::desc& input_desc,
                                           const mkldnn::memory::desc& delta_desc,
                                           const mkldnn::memory::desc& result_desc);

                size_t build_sigmoid_forward(const mkldnn::memory::desc& input_desc,
                                             const mkldnn::memory::desc& result_desc);

                size_t build_sigmoid_backward(const mkldnn::memory::desc& input_desc,
                                              const mkldnn::memory::desc& delta_desc,
                                              const mkldnn::memory::desc& result_desc);

                size_t build_elementwise_add(
                    const mkldnn::memory::desc& input0_data_desc,
                    const mkldnn::memory::desc& input1_data_desc,
                    const mkldnn::memory::desc& result_desc,
                    const std::vector<float>& scale_vector,
                    const std::vector<mkldnn::memory::primitive_desc>& input_pd);

                size_t build_batchnorm_forward(const mkldnn::memory::desc& input_desc,
                                               const mkldnn::memory::desc& weights_desc,
                                               const mkldnn::memory::desc& result_desc,
                                               const mkldnn::memory::desc& mean_desc,
                                               const mkldnn::memory::desc& variance_desc,
                                               const double eps,
                                               bool use_global_stats,
                                               bool bn_training_flag,
                                               const mkldnn::post_ops& pops = mkldnn::post_ops());

                size_t build_batchnorm_backward(const mkldnn::memory::desc& weights_desc,
                                                const mkldnn::memory::desc& input_desc,
                                                const mkldnn::memory::desc& mean_desc,
                                                const mkldnn::memory::desc& variance_desc,
                                                const mkldnn::memory::desc& delta_desc,
                                                const mkldnn::memory::desc& dinput_desc,
                                                const mkldnn::memory::desc& dweights_desc,
                                                const double eps);

                template <typename OP>
                size_t build_rnn(const ngraph::Node* node,
                                 const std::vector<TensorViewWrapper>& args,
                                 const std::vector<TensorViewWrapper>& out)
                {
                    auto rnn_node = static_cast<const OP*>(node);
                    auto src_sequence_length_max =
                        static_cast<unsigned long>(rnn_node->get_src_sequence_length());
                    auto direction = static_cast<unsigned long>(rnn_node->get_direction());
                    auto num_fused_layers =
                        static_cast<unsigned long>(rnn_node->get_num_fused_layers());
                    auto feature_size =
                        static_cast<unsigned long>(rnn_node->get_src_iter_feature_size());
                    auto batch = static_cast<unsigned long>(rnn_node->get_batch_size());
                    auto rnn_cell_n_gates =
                        static_cast<unsigned long>(rnn_node->get_gates_per_cell());
                    auto rnn_cell_n_states =
                        static_cast<unsigned long>(rnn_node->get_num_cell_states());

                    if (out[0].get_shape().size() == 2 && (out[0].get_shape()[1] != feature_size))
                    {
                        throw ngraph_error(
                            "input slc{ht} feature size is not equal to output dlc{ht} feature "
                            "size ");
                    }

                    if (out[1].get_shape().size() == 2 && (out[1].get_shape()[1] != feature_size) &&
                        rnn_node->get_num_timesteps() != 1)
                    {
                        throw ngraph_error(
                            "input sic{ht_1|ct_1} feature size is not equal to output "
                            "dlc{ht_1|ct_1} "
                            "feature size ");
                    }

                    Shape src_layer_tz{
                        src_sequence_length_max,
                        batch,
                        static_cast<unsigned long>(rnn_node->get_src_layer_feature_size())};
                    Shape src_iter_tz{
                        num_fused_layers, direction, rnn_cell_n_states, batch, feature_size};
                    Shape wei_layer_tz{
                        num_fused_layers,
                        direction,
                        static_cast<unsigned long>(rnn_node->get_src_layer_feature_size()),
                        rnn_cell_n_gates,
                        feature_size};
                    Shape wei_iter_tz{
                        num_fused_layers, direction, feature_size, rnn_cell_n_gates, feature_size};
                    Shape bias_tz{num_fused_layers, direction, rnn_cell_n_gates, feature_size};
                    Shape dst_layer_tz{src_sequence_length_max, batch, feature_size};
                    Shape dst_iter_tz{
                        num_fused_layers, direction, rnn_cell_n_states, batch, feature_size};

                    // We create the memory descriptors used by the user
                    auto src_layer_md = build_memory_descriptor(
                        src_layer_tz, args[0].get_element_type(), mkldnn::memory::format::tnc);
                    auto src_iter_md = build_memory_descriptor(
                        src_iter_tz, args[1].get_element_type(), mkldnn::memory::format::ldsnc);
                    auto wei_layer_md = build_memory_descriptor(
                        wei_layer_tz, args[2].get_element_type(), mkldnn::memory::format::ldigo);
                    auto wei_iter_md = build_memory_descriptor(
                        wei_iter_tz, args[3].get_element_type(), mkldnn::memory::format::ldigo);
                    auto bias_md = build_memory_descriptor(
                        bias_tz, args[4].get_element_type(), mkldnn::memory::format::ldgo);
                    auto dst_layer_md = build_memory_descriptor(
                        dst_layer_tz, out[0].get_element_type(), mkldnn::memory::format::tnc);
                    auto dst_iter_md = build_memory_descriptor(
                        dst_iter_tz, out[1].get_element_type(), mkldnn::memory::format::ldsnc);

                    return build_rnn_forward(src_layer_md,
                                             src_iter_md,
                                             wei_layer_md,
                                             wei_iter_md,
                                             bias_md,
                                             dst_layer_md,
                                             dst_iter_md);
                }

                size_t build_rnn_forward(const mkldnn::memory::desc& src_layer_desc,
                                         const mkldnn::memory::desc& src_iter_desc,
                                         const mkldnn::memory::desc& weights_layer_desc,
                                         const mkldnn::memory::desc& weights_iter_desc,
                                         const mkldnn::memory::desc& bias_desc,
                                         const mkldnn::memory::desc& dst_layer_desc,
                                         const mkldnn::memory::desc& dst_iter_desc);

                size_t build_concat(const std::vector<mkldnn::memory::desc>& inputs_data_desc,
                                    const mkldnn::memory::desc& result_desc,
                                    const size_t concat_dim);

                size_t build_slice(const mkldnn::memory::desc& input_desc,
                                   const mkldnn::memory::desc& result_desc,
                                   const ngraph::Coordinate& lower_bounds,
                                   const ngraph::Shape& result_shape);

                size_t build_softmax_forward(const mkldnn::memory::desc& input_desc,
                                             const mkldnn::memory::desc& result_desc,
                                             int softmax_axis);

                size_t build_leaky_relu(const mkldnn::memory::desc& input_desc,
                                        const mkldnn::memory::desc& result_desc,
                                        float alpha);

                size_t build_bounded_relu(const mkldnn::memory::desc& input_desc,
                                          const mkldnn::memory::desc& result_desc,
                                          float alpha);

                size_t build_quantized_max_pool(const ngraph::Node* node);

                size_t build_quantized_avg_pool(const ngraph::Node* node);

                size_t build_dequantization(const ngraph::Node* node,
                                            const mkldnn::memory::desc& input_desc,
                                            const mkldnn::memory::desc& result_desc);

                size_t build_quantize_reorder(const mkldnn::memory::desc& input_desc,
                                              const mkldnn::memory::desc& result_desc,
                                              const std::vector<float>& scales);
                template <typename OP>
                size_t get_scale_index()
                {
                    if (std::is_same<OP, ngraph::op::QuantizedConvolution>() ||
                        std::is_same<OP, ngraph::op::QuantizedConvolutionRelu>())
                    {
                        return 2;
                    }
                    if (std::is_same<OP, ngraph::op::QuantizedConvolutionBias>())
                    {
                        return 3;
                    }
                    if (std::is_same<OP, ngraph::op::QuantizedConvolutionBiasAdd>() ||
                        std::is_same<OP, ngraph::op::QuantizedConvolutionBiasSignedAdd>())
                    {
                        return 4;
                    }
                }

                template <typename OP, typename T>
                std::vector<T> get_output_scale(const ngraph::Node* node)
                {
                    auto index = get_scale_index<OP>();
                    std::vector<T> scale_val = {0};
                    auto scale_const_op = std::dynamic_pointer_cast<ngraph::op::Constant>(
                        node->get_arguments()[index]);
                    if (scale_const_op != nullptr)
                    {
                        scale_val = scale_const_op->template get_vector<T>();
                    }

                    return scale_val;
                }

                template <typename OP,
                          typename std::enable_if<
                              (std::is_same<OP, ngraph::op::Convolution>::value ||
                               std::is_same<OP, ngraph::op::QuantizedConvolution>::value),
                              std::nullptr_t>::type = nullptr>
                bool has_relu(const ngraph::Node* node)
                {
                    return false;
                }

                template <typename OP,
                          typename std::enable_if<
                              (!std::is_same<OP, ngraph::op::Convolution>::value &&
                               !std::is_same<OP, ngraph::op::QuantizedConvolution>::value),
                              std::nullptr_t>::type = nullptr>
                bool has_relu(const ngraph::Node* node)
                {
                    return static_cast<const OP*>(node)->with_relu();
                }

                template <typename OP>
                bool has_bias()
                {
                    if (std::is_same<OP, ngraph::op::ConvolutionBias>() ||
                        std::is_same<OP, ngraph::op::ConvolutionBiasAdd>() ||
                        std::is_same<OP, ngraph::op::QuantizedConvolutionBias>() ||
                        std::is_same<OP, ngraph::op::QuantizedConvolutionBiasAdd>() ||
                        std::is_same<OP, ngraph::op::QuantizedConvolutionBiasSignedAdd>())
                    {
                        return true;
                    }
                    else
                    {
                        return false;
                    }
                }

                template <typename OP>
                bool is_quantized_conv()
                {
                    if (std::is_same<OP, ngraph::op::QuantizedConvolution>() ||
                        std::is_same<OP, ngraph::op::QuantizedConvolutionRelu>() ||
                        std::is_same<OP, ngraph::op::QuantizedConvolutionBias>() ||
                        std::is_same<OP, ngraph::op::QuantizedConvolutionBiasAdd>() ||
                        std::is_same<OP, ngraph::op::QuantizedConvolutionBiasSignedAdd>())
                    {
                        return true;
                    }
                    else
                    {
                        return false;
                    }
                }

                template <typename OP>
                mkldnn::convolution_forward::desc
                    get_convolution_forward_desc(const ngraph::Node* node,
                                                 const std::vector<TensorViewWrapper>& args,
                                                 const std::vector<TensorViewWrapper>& out)
                {
                    auto convolution = static_cast<const OP*>(node);
                    // For dilation, MKLDNN wants to know how many elements to insert between, not how far
                    // apart to space the elements like nGraph. So we have to subtract 1 from each pos.
                    Strides window_dilation_strides_adjusted;

                    for (size_t s : convolution->get_window_dilation_strides())
                    {
                        window_dilation_strides_adjusted.push_back(s - 1);
                    }

                    auto data_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto weights_desc = mkldnn_utils::get_input_mkldnn_md(node, 1);
                    // MKLDNN relies on named formats for kernel selection
                    if (weights_desc.data.format == mkldnn_nchw)
                        weights_desc.data.format = mkldnn_oihw;
                    if (weights_desc.data.format == mkldnn_ncdhw)
                        weights_desc.data.format = mkldnn_oidhw;
                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);

                    if (has_bias<OP>())
                    {
                        auto bias_desc = mkldnn_utils::get_input_mkldnn_md(node, 2);
                        return mkldnn::convolution_forward::desc(
                            mkldnn::prop_kind::forward,
                            mkldnn::algorithm::convolution_direct,
                            data_desc,
                            weights_desc,
                            bias_desc,
                            result_desc,
                            MKLDNN_DIMS(convolution->get_window_movement_strides()),
                            MKLDNN_DIMS(window_dilation_strides_adjusted),
                            MKLDNN_DIMS(convolution->get_padding_below()),
                            MKLDNN_DIMS(convolution->get_padding_above()),
                            mkldnn::padding_kind::zero);
                    }
                    else
                    {
                        return mkldnn::convolution_forward::desc(
                            mkldnn::prop_kind::forward,
                            mkldnn::algorithm::convolution_direct,
                            data_desc,
                            weights_desc,
                            result_desc,
                            MKLDNN_DIMS(convolution->get_window_movement_strides()),
                            MKLDNN_DIMS(window_dilation_strides_adjusted),
                            MKLDNN_DIMS(convolution->get_padding_below()),
                            MKLDNN_DIMS(convolution->get_padding_above()),
                            mkldnn::padding_kind::zero);
                    }
                }

                template <typename OP>
                mkldnn::primitive_attr get_convolution_forward_attr(const ngraph::Node* node)
                {
                    mkldnn::post_ops ops;

                    if (std::is_same<OP, ngraph::op::ConvolutionBiasAdd>() ||
                        std::is_same<OP, ngraph::op::ConvolutionAdd>())
                    {
                        ops.append_sum(1.f);
                    }

                    if (std::is_same<OP, ngraph::op::QuantizedConvolutionBiasAdd>() ||
                        std::is_same<OP, ngraph::op::QuantizedConvolutionBiasSignedAdd>())
                    {
                        auto sum_scale_val =
                            extract_scale_value<ngraph::op::QuantizedConvolutionBiasAdd>(node, 5);
                        ops.append_sum(sum_scale_val[0]);
                    }

                    if (has_relu<OP>(node))
                    {
                        const float ops_scale = 1.f;
                        const float ops_alpha = -0.f; // relu negative slope
                        const float ops_beta = 0.f;
                        ops.append_eltwise(
                            ops_scale, mkldnn::algorithm::eltwise_relu, ops_alpha, ops_beta);
                    }

                    mkldnn::primitive_attr conv_attr;
                    conv_attr.set_post_ops(ops);
                    if (is_quantized_conv<OP>())
                    {
                        conv_attr.set_int_output_round_mode(mkldnn::round_mode::round_nearest);
                        conv_attr.set_output_scales(0, get_output_scale<OP, float>(node));
                    }
                    return conv_attr;
                }

                size_t convolution_forward_init(bool with_bias = false);

                template <bool with_bias>
                void convolution_forward(const mkldnn::convolution_forward::desc& desc,
                                         const mkldnn::primitive_attr& attr,
                                         const mkldnn::engine& engine,
                                         size_t& conv_idx)
                {
                    size_t input_idx, weights_idx, results_idx, bias_idx;
                    input_idx = m_primitive_deps[conv_idx][0];
                    weights_idx = m_primitive_deps[conv_idx][1];
                    m_mkldnn_primitives[input_idx] =
                        new mkldnn::memory({{desc.data.src_desc}, engine}, nullptr);
                    m_mkldnn_primitives[weights_idx] =
                        new mkldnn::memory({{desc.data.weights_desc}, engine}, nullptr);
                    if (with_bias)
                    {
                        bias_idx = m_primitive_deps[conv_idx][2];
                        results_idx = m_primitive_deps[conv_idx][3];
                        m_mkldnn_primitives[bias_idx] =
                            new mkldnn::memory({{desc.data.bias_desc}, engine}, nullptr);
                    }
                    else
                    {
                        results_idx = m_primitive_deps[conv_idx][2];
                    }
                    m_mkldnn_primitives[results_idx] =
                        new mkldnn::memory({{desc.data.dst_desc}, engine}, nullptr);

                    mkldnn::primitive* prim;
                    if (with_bias)
                    {
                        prim = new mkldnn::convolution_forward({desc, attr, engine},
                                                               *m_mkldnn_primitives[input_idx],
                                                               *m_mkldnn_primitives[weights_idx],
                                                               *m_mkldnn_primitives[bias_idx],
                                                               *m_mkldnn_primitives[results_idx]);
                    }
                    else
                    {
                        prim = new mkldnn::convolution_forward({desc, attr, engine},
                                                               *m_mkldnn_primitives[input_idx],
                                                               *m_mkldnn_primitives[weights_idx],
                                                               *m_mkldnn_primitives[results_idx]);
                    }

                    m_mkldnn_primitives[conv_idx] = prim;
                }

            private:
                std::vector<mkldnn::primitive*> m_mkldnn_primitives;
                std::vector<mkldnn::stream> m_mkldnn_streams;
                std::unordered_map<size_t, std::vector<size_t>> m_primitive_deps;
                std::vector<std::unique_ptr<MKLDNNWorkspace>> m_workspaces;
                std::vector<char*> m_workspace_bufs;
            };
        }
    }
}
