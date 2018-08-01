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

#pragma once

#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include <mkldnn.hpp>

#include "ngraph/coordinate_diff.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/convolution.hpp"
#include "ngraph/runtime/cpu/mkldnn_utils.hpp"
#include "ngraph/runtime/cpu/op/conv_bias.hpp"
#include "ngraph/runtime/cpu/op/conv_relu.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/strides.hpp"
#include "ngraph/type/element_type.hpp"

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
                MKLDNNWorkspace(size_t size) { buf = reinterpret_cast<char*>(malloc(size)); }
                ~MKLDNNWorkspace() { free(buf); }
                char* buf;
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
                mkldnn::memory::desc build_memory_descriptor(const TensorViewWrapper& tvw) const;
                mkldnn::memory::desc build_memory_descriptor(const Shape& shape,
                                                             const ngraph::element::Type& et,
                                                             mkldnn::memory::format fmt) const;
                mkldnn::memory::desc
                    build_blocked_memory_descriptor(const mkldnn::memory::dims& dim,
                                                    const mkldnn::memory::dims& strides,
                                                    mkldnn::memory::data_type dtype) const;
                mkldnn::memory build_memory_primitive(const TensorViewWrapper& tvw) const;
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

                    auto data_format = mkldnn_utils::get_input_mkldnn_format(node, 0);
                    auto weights_format = mkldnn_utils::get_input_mkldnn_format(node, 1);

                    // HACK to help MKLDNN pick the right implementation
                    if (weights_format == mkldnn::memory::format::nchw)
                    {
                        weights_format = mkldnn::memory::format::oihw;
                    }
                    auto result_format = mkldnn_utils::get_output_mkldnn_format(node, 0);

                    auto data_desc = build_memory_descriptor(args[0], data_format);
                    auto weights_desc = build_memory_descriptor(args[1], weights_format);
                    auto result_desc = build_memory_descriptor(out[0], result_format);

                    mkldnn::post_ops ops;

                    if (std::is_same<OP, ngraph::op::ConvolutionBiasAdd>())
                    {
                        ops.append_sum(1.f);
                    }

                    auto add_relu = [&]() {
                        if (dynamic_cast<const ngraph::op::ConvolutionBias*>(node))
                        {
                            return (dynamic_cast<const ngraph::op::ConvolutionBias*>(node))
                                ->with_relu();
                        }
                        if (dynamic_cast<const ngraph::op::ConvolutionBiasAdd*>(node))
                        {
                            return (dynamic_cast<const ngraph::op::ConvolutionBiasAdd*>(node))
                                ->with_relu();
                        }
                        if (dynamic_cast<const ngraph::op::ConvolutionRelu*>(node))
                        {
                            return true;
                        }
                        return false;
                    };

                    if (add_relu())
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
                        auto bias_format = mkldnn_utils::get_input_mkldnn_format(node, 2);
                        auto bias_desc = build_memory_descriptor(args[2], bias_format);
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

                std::pair<size_t, size_t> build_group_convolution_forward(
                    const mkldnn::memory::desc& input_reorder_desc,
                    const mkldnn::memory::desc& input_conv_desc,
                    const mkldnn::memory::desc& weights_desc,
                    const mkldnn::memory::desc& result_reorder_desc,
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

                    auto arg0_format = mkldnn_utils::get_input_mkldnn_format(node, 0);
                    if (std::is_same<OP, ngraph::op::ConvolutionBackpropData>())
                    {
                        // HACK to help MKLDNN pick the right implementation
                        arg0_format = (arg0_format == mkldnn::memory::format::nchw)
                                          ? mkldnn::memory::format::oihw
                                          : arg0_format;
                    }
                    auto arg0_desc = build_memory_descriptor(args[0], arg0_format);
                    auto arg1_format = mkldnn_utils::get_input_mkldnn_format(node, 1);
                    auto arg1_desc = build_memory_descriptor(args[1], arg1_format);
                    auto out0_format = mkldnn_utils::get_output_mkldnn_format(node, 0);
                    auto out0_desc = build_memory_descriptor(out[0], out0_format);

                    if (std::is_same<OP, ngraph::op::ConvolutionBackpropData>())
                    {
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
                        auto out1_format = mkldnn_utils::get_output_mkldnn_format(node, 1);
                        auto out1_desc = build_memory_descriptor(out[1], out1_format);
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

                size_t build_softmax_forward(const mkldnn::memory::desc& input_desc,
                                             const mkldnn::memory::desc& result_desc,
                                             int softmax_axis);

                size_t build_bounded_relu(const mkldnn::memory::desc& input_desc,
                                          const mkldnn::memory::desc& result_desc,
                                          float alpha);

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
