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

// For direct execution, we reserve space for primitives then create those primitives the first
// time functor is called. This could be extended to create primitives when shapes are changed.
// Different ops need different numbers of primitives.

#pragma once

#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include <mkldnn.hpp>

#include "ngraph/coordinate_diff.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/avg_pool.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/convolution.hpp"
#include "ngraph/op/experimental/quantized_avg_pool.hpp"
#include "ngraph/op/experimental/quantized_conv.hpp"
#include "ngraph/op/experimental/quantized_conv_bias.hpp"
#include "ngraph/op/experimental/quantized_conv_relu.hpp"
#include "ngraph/op/experimental/quantized_dot.hpp"
#include "ngraph/op/experimental/quantized_dot_bias.hpp"
#include "ngraph/op/experimental/quantized_max_pool.hpp"
#include "ngraph/op/fused/conv_fused.hpp"
#include "ngraph/op/fused/group_conv.hpp"
#include "ngraph/op/lrn.hpp"
#include "ngraph/op/max_pool.hpp"
#include "ngraph/op/softmax.hpp"
#include "ngraph/runtime/cpu/cpu_executor.hpp"
#include "ngraph/runtime/cpu/cpu_tensor_view_wrapper.hpp"
#include "ngraph/runtime/cpu/mkldnn_utils.hpp"
#include "ngraph/runtime/cpu/op/bounded_relu.hpp"
#include "ngraph/runtime/cpu/op/conv_add.hpp"
#include "ngraph/runtime/cpu/op/conv_relu.hpp"
#include "ngraph/runtime/cpu/op/deconv.hpp"
#include "ngraph/runtime/cpu/op/group_conv_bias.hpp"
#include "ngraph/runtime/cpu/op/leaky_relu.hpp"
#include "ngraph/runtime/cpu/op/rnn_utils.hpp"
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

            // TODO (nbpatel) Templatize the return type when we have double scales
            template <typename OP>
            inline std::vector<float> extract_scale_value(const ngraph::Node* node, int index)
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

            template <typename OP,
                      typename std::enable_if<
                          (std::is_same<OP, ngraph::op::Convolution>::value ||
                           std::is_same<OP, ngraph::op::QuantizedConvolution>::value ||
                           std::is_same<OP, ngraph::op::GroupConvolution>::value),
                          std::nullptr_t>::type = nullptr>
            bool has_relu(const ngraph::Node* node)
            {
                return false;
            }

            template <typename OP,
                      typename std::enable_if<
                          (!std::is_same<OP, ngraph::op::Convolution>::value &&
                           !std::is_same<OP, ngraph::op::QuantizedConvolution>::value &&
                           !std::is_same<OP, ngraph::op::GroupConvolution>::value),
                          std::nullptr_t>::type = nullptr>
            bool has_relu(const ngraph::Node* node)
            {
                return static_cast<const OP*>(node)->with_relu();
            }

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
                const std::vector<mkldnn::primitive*>& get_mkldnn_primitives_cg() const;
                std::vector<mkldnn::primitive*>& get_mkldnn_primitives();
                const std::vector<char*>& get_mkldnn_workspaces();

                // reserve the space for primitives for each op, different op requires different number of primitives.
                // some ops require a new workspace.
                size_t reserve_primitive_space(size_t count, bool new_workspace = false);
                size_t reserve_primitive_space_cg(size_t count, bool new_workspace = false);
                size_t insert_primitive(mkldnn::primitive* primitive);
                size_t insert_workspace(std::unique_ptr<MKLDNNWorkspace>& workspace);
                size_t insert_workspace(std::vector<char*>& mkldnn_workspaces,
                                        std::unique_ptr<MKLDNNWorkspace>& workspace);
                const std::vector<size_t>& get_primitive_deps(size_t index) const;
                const std::vector<size_t>& get_primitive_deps_cg(size_t index) const;
                size_t reserve_workspace();
                void reserve_descriptor_space(size_t count);
                size_t get_mkldnn_descriptors_size();
                std::vector<size_t>& get_primitive_deps(size_t index);

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
                void build_memory_primitive(const mkldnn::memory::desc& desc, size_t index);
                void build_memory_primitive(std::vector<mkldnn::primitive*>& mkldnn_primitives,
                                            const mkldnn::memory::desc& desc,
                                            size_t index);

                size_t build_quantized_inner_product_forward(
                    const mkldnn::memory::desc& input_data_desc,
                    const mkldnn::memory::desc& weights_desc,
                    const mkldnn::memory::desc& result_desc,
                    const float scale,
                    const mkldnn::post_ops& pops = mkldnn::post_ops());

                size_t build_quantized_inner_product_forward(
                    const mkldnn::memory::desc& input_data_desc,
                    const mkldnn::memory::desc& weights_desc,
                    const mkldnn::memory::desc& bias_desc,
                    const mkldnn::memory::desc& result_desc,
                    const float scale,
                    const mkldnn::post_ops& pops = mkldnn::post_ops());

                void build_deconvolutionbias_forward(
                    std::vector<mkldnn::primitive*>& mkldnn_primitives,
                    const mkldnn::deconvolution_forward::desc& fwd_desc,
                    const std::vector<size_t>& deps,
                    size_t conv_index,
                    const mkldnn::memory::desc& weights_desc);

                size_t build_deconvolutionbias_forward(
                    const mkldnn::memory::desc& input_data_desc,
                    const mkldnn::memory::desc& weights_desc,
                    const mkldnn::memory::desc& bias_desc,
                    const mkldnn::memory::desc& result_desc,
                    const ngraph::Strides& strides,
                    const ngraph::Strides& dilation_strides,
                    const ngraph::CoordinateDiff& padding_below,
                    const ngraph::CoordinateDiff& padding_above,
                    const mkldnn::post_ops& pops = mkldnn::post_ops());

                template <typename OP>
                size_t build_deconvolution(const ngraph::Node* node,
                                           const std::vector<TensorViewWrapper>& args,
                                           const std::vector<TensorViewWrapper>& out)
                {
                    auto convolution = static_cast<const OP*>(node);

                    // For dilation, MKLDNN wants to know how many elements to insert between, not how far
                    // apart to space the elements like nGraph. So we have to subtract 1 from each pos.
                    Strides window_dilation_strides_adjusted;

                    for (size_t s : convolution->get_window_dilation_strides_forward())
                    {
                        window_dilation_strides_adjusted.push_back(s - 1);
                    }

                    auto weights_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto data_desc = mkldnn_utils::get_input_mkldnn_md(node, 1);

                    // MKLDNN relies on named formats for kernel selection
                    if (weights_desc.data.format == mkldnn_nchw)
                        weights_desc.data.format = mkldnn_oihw;
                    if (weights_desc.data.format == mkldnn_ncdhw)
                        weights_desc.data.format = mkldnn_oidhw;

                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);

                    mkldnn::post_ops ops;

                    auto add_relu = [&]() {
                        if (dynamic_cast<const ngraph::op::DeconvolutionBias*>(node))
                        {
                            return (dynamic_cast<const ngraph::op::DeconvolutionBias*>(node))
                                ->with_relu();
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

                    if (std::is_same<OP, ngraph::op::DeconvolutionBias>())
                    {
                        auto bias_desc = mkldnn_utils::get_input_mkldnn_md(node, 2);
                        return build_deconvolutionbias_forward(
                            data_desc,
                            weights_desc,
                            bias_desc,
                            result_desc,
                            convolution->get_window_movement_strides_forward(),
                            window_dilation_strides_adjusted,
                            convolution->get_padding_below_forward(),
                            convolution->get_padding_above_forward(),
                            ops);
                    }
                    else
                    {
                        throw ngraph_error("Unsupported Op.");
                    }
                }

                template <typename OP>
                size_t build_inner_product(const ngraph::Node* node,
                                           const std::vector<TensorViewWrapper>& args,
                                           const std::vector<TensorViewWrapper>& out)
                {
                    auto data_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto weights_desc = mkldnn_utils::get_input_mkldnn_md(node, 1);

                    // MKLDNN relies on named formats for kernel selection
                    if (weights_desc.data.format == mkldnn_nchw)
                    {
                        weights_desc.data.format = mkldnn_oihw;
                    }
                    if (weights_desc.data.format == mkldnn_ncdhw)
                    {
                        weights_desc.data.format = mkldnn_oidhw;
                    }

                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);

                    mkldnn::post_ops ops;

                    if (has_relu<OP>(node))
                    {
                        const float ops_scale = 1.f;
                        const float ops_alpha = -0.f; // relu negative slope
                        const float ops_beta = 0.f;
                        ops.append_eltwise(
                            ops_scale, mkldnn::algorithm::eltwise_relu, ops_alpha, ops_beta);
                    }

                    if (std::is_same<OP, ngraph::op::QuantizedDot>())
                    {
                        auto scale_val = extract_scale_value<OP>(node, 2);
                        return build_quantized_inner_product_forward(
                            data_desc, weights_desc, result_desc, scale_val[0], ops);
                    }
                    else if (std::is_same<OP, ngraph::op::QuantizedDotBias>())
                    {
                        auto scale_val = extract_scale_value<OP>(node, 3);
                        auto bias_desc = mkldnn_utils::get_input_mkldnn_md(node, 2);
                        return build_quantized_inner_product_forward(
                            data_desc, weights_desc, bias_desc, result_desc, scale_val[0], ops);
                    }
                    else
                    {
                        throw ngraph_error("unsupported inner_product");
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

                void build_convolution_backward_weights(
                    std::vector<mkldnn::primitive*>& mkldnn_primitives,
                    const mkldnn::convolution_backward_weights::desc& bwd_desc,
                    const mkldnn::convolution_forward::desc& fwd_desc,
                    const std::vector<size_t>& deps,
                    size_t conv_index);

                size_t build_convolution_backward_data(const mkldnn::memory::desc& weights_desc,
                                                       const mkldnn::memory::desc& delta_desc,
                                                       const mkldnn::memory::desc& result_desc,
                                                       const ngraph::Strides& strides,
                                                       const ngraph::Strides& dilation_strides,
                                                       const ngraph::CoordinateDiff& padding_below,
                                                       const ngraph::CoordinateDiff& padding_above);

                void build_convolution_backward_data(
                    std::vector<mkldnn::primitive*>& mkldnn_primitives,
                    const mkldnn::convolution_backward_data::desc& bwd_desc,
                    const mkldnn::convolution_forward::desc& fwd_desc,
                    const std::vector<size_t>& deps,
                    size_t conv_index);

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

                void build_convolution_backward_weights_bias(
                    std::vector<mkldnn::primitive*>& mkldnn_primitives,
                    const mkldnn::convolution_backward_weights::desc& bwd_desc,
                    const mkldnn::convolution_forward::desc& fwd_desc,
                    const std::vector<size_t>& deps,
                    size_t conv_index);

                template <typename OP>
                mkldnn::pooling_forward::desc get_avg_pooling_forward_desc(const ngraph::Node* node,
                                                                           bool training)
                {
                    auto pool = static_cast<const OP*>(node);

                    auto window_shape = pool->get_window_shape();
                    auto window_strides = pool->get_window_movement_strides();
                    auto padding_below = pool->get_padding_below();
                    auto padding_above = pool->get_padding_above();
                    auto include_padding_in_avg_computation =
                        pool->get_include_padding_in_avg_computation();

                    auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);

                    if (training)
                    {
                        return mkldnn::pooling_forward::desc(
                            mkldnn::prop_kind::forward_training,
                            (include_padding_in_avg_computation
                                 ? mkldnn::algorithm::pooling_avg_include_padding
                                 : mkldnn::algorithm::pooling_avg_exclude_padding),
                            result_desc,
                            input_desc,
                            mkldnn::memory::dims(window_strides.begin(), window_strides.end()),
                            mkldnn::memory::dims(window_shape.begin(), window_shape.end()),
                            mkldnn::memory::dims(padding_below.begin(), padding_below.end()),
                            mkldnn::memory::dims(padding_above.begin(), padding_above.end()),
                            mkldnn::padding_kind::zero);
                    }
                    else
                    {
                        return mkldnn::pooling_forward::desc(
                            mkldnn::prop_kind::forward_inference,
                            (include_padding_in_avg_computation
                                 ? mkldnn::algorithm::pooling_avg_include_padding
                                 : mkldnn::algorithm::pooling_avg_exclude_padding),
                            input_desc,
                            result_desc,
                            mkldnn::memory::dims(window_strides.begin(), window_strides.end()),
                            mkldnn::memory::dims(window_shape.begin(), window_shape.end()),
                            mkldnn::memory::dims(padding_below.begin(), padding_below.end()),
                            mkldnn::memory::dims(padding_above.begin(), padding_above.end()),
                            mkldnn::padding_kind::zero);
                    }
                }

                template <typename OP>
                mkldnn::pooling_forward::desc get_max_pooling_forward_desc(const ngraph::Node* node,
                                                                           bool training)
                {
                    auto pool = static_cast<const OP*>(node);

                    auto window_shape = pool->get_window_shape();
                    auto window_strides = pool->get_window_movement_strides();
                    auto padding_below = pool->get_padding_below();
                    auto padding_above = pool->get_padding_above();

                    if (training)
                    {
                        auto diff_dst_desc = mkldnn_utils::get_input_mkldnn_md(node, 1);
                        auto diff_src_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);

                        return mkldnn::pooling_forward::desc(
                            mkldnn::prop_kind::forward_training,
                            mkldnn::algorithm::pooling_max,
                            diff_src_desc,
                            diff_dst_desc,
                            mkldnn::memory::dims(window_strides.begin(), window_strides.end()),
                            mkldnn::memory::dims(window_shape.begin(), window_shape.end()),
                            mkldnn::memory::dims(padding_below.begin(), padding_below.end()),
                            mkldnn::memory::dims(padding_above.begin(), padding_above.end()),
                            mkldnn::padding_kind::zero);
                    }
                    else
                    {
                        auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                        auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);

                        return mkldnn::pooling_forward::desc(
                            mkldnn::prop_kind::forward_inference,
                            mkldnn::algorithm::pooling_max,
                            input_desc,
                            result_desc,
                            mkldnn::memory::dims(window_strides.begin(), window_strides.end()),
                            mkldnn::memory::dims(window_shape.begin(), window_shape.end()),
                            mkldnn::memory::dims(padding_below.begin(), padding_below.end()),
                            mkldnn::memory::dims(padding_above.begin(), padding_above.end()),
                            mkldnn::padding_kind::zero);
                    }
                }

                void build_pooling_forward(std::vector<mkldnn::primitive*>& mkldnn_primitives,
                                           const mkldnn::pooling_forward::desc& pool_desc,
                                           const std::vector<size_t>& deps,
                                           size_t pool_index);

                size_t build_pooling_backward(mkldnn::algorithm pooling_algorithm,
                                              const mkldnn::memory::desc& diff_dst_desc,
                                              const mkldnn::memory::desc& diff_src_desc,
                                              const ngraph::Strides& window_strides,
                                              const ngraph::Shape& window_shape,
                                              const ngraph::Shape& padding_below,
                                              const ngraph::Shape& padding_above);

                template <typename OP>
                mkldnn::pooling_backward::desc
                    get_avg_pooling_backward_desc(const ngraph::Node* node)
                {
                    auto pool = static_cast<const OP*>(node);

                    auto window_shape = pool->get_window_shape();
                    auto window_strides = pool->get_window_movement_strides();
                    auto padding_below = pool->get_padding_below();
                    auto padding_above = pool->get_padding_above();
                    auto include_padding_in_avg_computation =
                        pool->get_include_padding_in_avg_computation();

                    auto diff_dst_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto diff_src_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);

                    return mkldnn::pooling_backward::desc(
                        (include_padding_in_avg_computation
                             ? mkldnn::algorithm::pooling_avg_include_padding
                             : mkldnn::algorithm::pooling_avg_exclude_padding),
                        diff_src_desc,
                        diff_dst_desc,
                        mkldnn::memory::dims(window_strides.begin(), window_strides.end()),
                        mkldnn::memory::dims(window_shape.begin(), window_shape.end()),
                        mkldnn::memory::dims(padding_below.begin(), padding_below.end()),
                        mkldnn::memory::dims(padding_above.begin(), padding_above.end()),
                        mkldnn::padding_kind::zero);
                }

                void build_pooling_backward(std::vector<mkldnn::primitive*>& mkldnn_primitives,
                                            const mkldnn::pooling_backward::desc& pool_desc,
                                            const mkldnn::pooling_forward::desc& pool_fwd_desc,
                                            const std::vector<size_t>& deps,
                                            size_t pool_index);

                size_t build_max_pooling_with_indices_forward(mkldnn::algorithm pooling_algorithm,
                                                              const mkldnn::memory::desc& src_desc,
                                                              const mkldnn::memory::desc& dst_desc,
                                                              const ngraph::Strides& window_strides,
                                                              const ngraph::Shape& window_shape,
                                                              const ngraph::Shape& padding_below,
                                                              const ngraph::Shape& padding_above);

                template <typename OP>
                mkldnn::pooling_forward::desc
                    get_max_pooling_with_indices_forward_desc(const ngraph::Node* node)
                {
                    auto pool = static_cast<const OP*>(node);

                    auto window_shape = pool->get_window_shape();
                    auto window_strides = pool->get_window_movement_strides();
                    auto padding_below = pool->get_padding_below();
                    auto padding_above = pool->get_padding_above();

                    auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);

                    return mkldnn::pooling_forward::desc(
                        mkldnn::prop_kind::forward_training,
                        mkldnn::algorithm::pooling_max,
                        input_desc,
                        result_desc,
                        mkldnn::memory::dims(window_strides.begin(), window_strides.end()),
                        mkldnn::memory::dims(window_shape.begin(), window_shape.end()),
                        mkldnn::memory::dims(padding_below.begin(), padding_below.end()),
                        mkldnn::memory::dims(padding_above.begin(), padding_above.end()),
                        mkldnn::padding_kind::zero);
                }

                void build_max_pooling_with_indices_forward(
                    std::vector<mkldnn::primitive*>& mkldnn_primitives,
                    const mkldnn::pooling_forward::desc& max_pool_desc,
                    const std::vector<size_t>& deps,
                    size_t max_pool_index);

                size_t build_max_pooling_backward(mkldnn::algorithm pooling_algorithm,
                                                  const mkldnn::memory::desc& fprop_src_desc,
                                                  const mkldnn::memory::desc& diff_dst_desc,
                                                  const mkldnn::memory::desc& diff_src_desc,
                                                  const ngraph::Strides& window_strides,
                                                  const ngraph::Shape& window_shape,
                                                  const ngraph::Shape& padding_below,
                                                  const ngraph::Shape& padding_above);

                template <typename OP>
                mkldnn::pooling_backward::desc
                    get_max_pooling_backward_desc(const ngraph::Node* node)
                {
                    auto pool = static_cast<const OP*>(node);

                    auto window_shape = pool->get_window_shape();
                    auto window_strides = pool->get_window_movement_strides();
                    auto padding_below = pool->get_padding_below();
                    auto padding_above = pool->get_padding_above();

                    auto diff_dst_desc = mkldnn_utils::get_input_mkldnn_md(node, 1);
                    auto diff_src_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);

                    return mkldnn::pooling_backward::desc(
                        mkldnn::algorithm::pooling_max,
                        diff_src_desc,
                        diff_dst_desc,
                        mkldnn::memory::dims(window_strides.begin(), window_strides.end()),
                        mkldnn::memory::dims(window_shape.begin(), window_shape.end()),
                        mkldnn::memory::dims(padding_below.begin(), padding_below.end()),
                        mkldnn::memory::dims(padding_above.begin(), padding_above.end()),
                        mkldnn::padding_kind::zero);
                }

                void build_max_pooling_backward(std::vector<mkldnn::primitive*>& mkldnn_primitives,
                                                std::vector<char*>& mkldnn_workspaces,
                                                const mkldnn::pooling_backward::desc& bwd_pool_desc,
                                                const mkldnn::pooling_forward::desc& fwd_pool_desc,
                                                const mkldnn::memory::desc& fprop_src_desc,
                                                std::vector<size_t>& fdeps,
                                                std::vector<size_t>& bdeps,
                                                size_t fwd_pool_index,
                                                size_t bwd_pool_index);

                size_t build_max_pooling_with_indices_backward(
                    mkldnn::algorithm pooling_algorithm,
                    const mkldnn::memory::desc& diff_dst_desc,
                    const mkldnn::memory::desc& diff_src_desc,
                    const ngraph::Strides& window_strides,
                    const ngraph::Shape& window_shape,
                    const ngraph::Shape& padding_below,
                    const ngraph::Shape& padding_above);

                void build_max_pooling_with_indices_backward(
                    std::vector<mkldnn::primitive*>& mkldnn_primitives,
                    const mkldnn::pooling_backward::desc& bwd_pool_desc,
                    const mkldnn::pooling_forward::desc& fwd_pool_desc,
                    const std::vector<size_t>& deps,
                    size_t max_pool_index);

                size_t build_reorder(const mkldnn::memory::desc& input_desc,
                                     const mkldnn::memory::desc& result_desc);

                void build_reorder(std::vector<mkldnn::primitive*>& mkldnn_primitives,
                                   const mkldnn::memory::desc& input_desc,
                                   const mkldnn::memory::desc& result_desc,
                                   const std::vector<size_t>& deps,
                                   size_t reorder_index);

                mkldnn::lrn_forward::desc get_lrn_forward_desc(const ngraph::Node* node);

                void build_lrn_forward(std::vector<mkldnn::primitive*>& mkldnn_primitives,
                                       const mkldnn::lrn_forward::desc& lrn_desc,
                                       const std::vector<size_t>& deps,
                                       size_t lrn_index);

                size_t build_relu_forward(const mkldnn::memory::desc& input_desc,
                                          const mkldnn::memory::desc& result_desc);

                mkldnn::eltwise_forward::desc get_relu_forward_desc(const ngraph::Node* node);

                void build_relu_forward(std::vector<mkldnn::primitive*>& mkldnn_primitives,
                                        const mkldnn::eltwise_forward::desc& relu_desc,
                                        const std::vector<size_t>& deps,
                                        size_t relu_index);

                size_t build_relu_backward(const mkldnn::memory::desc& input_desc,
                                           const mkldnn::memory::desc& delta_desc,
                                           const mkldnn::memory::desc& result_desc);

                mkldnn::eltwise_backward::desc get_relu_backward_desc(const ngraph::Node* node);

                void build_relu_backward(std::vector<mkldnn::primitive*>& mkldnn_primitives,
                                         const mkldnn::eltwise_backward::desc& bwd_desc,
                                         const mkldnn::eltwise_forward::desc& fwd_desc,
                                         const std::vector<size_t>& deps,
                                         size_t relu_index);

                size_t build_sigmoid_forward(const mkldnn::memory::desc& input_desc,
                                             const mkldnn::memory::desc& result_desc);

                mkldnn::eltwise_forward::desc get_sigmoid_forward_desc(const ngraph::Node* node,
                                                                       bool backward_op);

                void build_sigmoid_forward(std::vector<mkldnn::primitive*>& mkldnn_primitives,
                                           const mkldnn::eltwise_forward::desc& sigmoid_desc,
                                           const std::vector<size_t>& deps,
                                           size_t sigmoid_index);

                size_t build_sigmoid_backward(const mkldnn::memory::desc& input_desc,
                                              const mkldnn::memory::desc& delta_desc,
                                              const mkldnn::memory::desc& result_desc);

                mkldnn::eltwise_backward::desc get_sigmoid_backward_desc(const ngraph::Node* node);

                void build_sigmoid_backward(std::vector<mkldnn::primitive*>& mkldnn_primitives,
                                            const mkldnn::eltwise_backward::desc& bwd_desc,
                                            const mkldnn::eltwise_forward::desc& fwd_desc,
                                            const std::vector<size_t>& deps,
                                            size_t sigmoid_index);

                mkldnn::sum::primitive_desc get_elementwise_add_desc(const ngraph::Node* node);

                void build_elementwise_add(std::vector<mkldnn::primitive*>& mkldnn_primitives,
                                           const mkldnn::sum::primitive_desc& sum_pd,
                                           const std::vector<size_t>& deps,
                                           size_t add_index);

                template <typename OP>
                mkldnn::batch_normalization_forward::desc
                    get_batchnorm_forward_desc(const ngraph::Node* node, bool training_with_3args)
                {
                    const OP* batchnorm = static_cast<const OP*>(node);
                    auto eps = batchnorm->get_eps_value();

                    if (training_with_3args)
                    {
                        auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 2);
                        return mkldnn::batch_normalization_forward::desc(
                            mkldnn::prop_kind::forward_training,
                            input_desc,
                            eps,
                            mkldnn::batch_normalization_flag::use_scale_shift);
                    }
                    else
                    {
                        auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 2);
                        return mkldnn::batch_normalization_forward::desc(
                            mkldnn::prop_kind::forward_training,
                            input_desc,
                            eps,
                            mkldnn::batch_normalization_flag::use_scale_shift |
                                mkldnn::batch_normalization_flag::use_global_stats);
                    }
                }

                void build_batchnorm_forward(
                    std::vector<mkldnn::primitive*>& mkldnn_primitives,
                    const mkldnn::batch_normalization_forward::desc& batchnorm_desc,
                    const mkldnn::memory::desc& weights_desc,
                    bool bn_training_flag,
                    const std::vector<size_t>& deps,
                    size_t batchnorm_index,
                    const mkldnn::post_ops& pops = mkldnn::post_ops());

                mkldnn::batch_normalization_backward::desc
                    get_batchnorm_backward_desc(const ngraph::Node* node);

                void build_batchnorm_backward(
                    std::vector<mkldnn::primitive*>& mkldnn_primitives,
                    const mkldnn::batch_normalization_backward::desc& batchnorm_desc,
                    const mkldnn::memory::desc& weights_desc,
                    const mkldnn::memory::desc& dweights_desc,
                    const std::vector<size_t>& deps,
                    size_t batchnorm_index);

                void build_rnn_forward(std::vector<mkldnn::primitive*>& mkldnn_primitives,
                                       std::vector<char*>& mkldnn_workspaces,
                                       const mkldnn::rnn_forward::desc& desc,
                                       std::vector<size_t>& deps,
                                       size_t rnn_idx);

                size_t build_concat(const std::vector<mkldnn::memory::desc>& inputs_data_desc,
                                    const mkldnn::memory::desc& result_desc,
                                    const size_t concat_dim);

                template <typename OP>
                mkldnn::concat::primitive_desc get_concat_desc(const ngraph::Node* node,
                                                               size_t nargs)
                {
                    auto concat = static_cast<const OP*>(node);

                    std::vector<mkldnn::memory::primitive_desc> inputs_pd;
                    for (size_t i = 0; i < nargs; i++)
                    {
                        inputs_pd.push_back(mkldnn::memory::primitive_desc(
                            mkldnn_utils::get_input_mkldnn_md(node, i),
                            runtime::cpu::executor::global_cpu_engine));
                    }

                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);

                    size_t concat_dim = concat->get_concatenation_axis();

                    // concat primtive descriptor
                    return mkldnn::concat::primitive_desc(
                        result_desc, static_cast<int>(concat_dim), inputs_pd);
                }
                void build_concat(std::vector<mkldnn::primitive*>& mkldnn_primitives,
                                  const mkldnn::concat::primitive_desc& concat_pd,
                                  const std::vector<mkldnn::memory::desc>& inputs_data_desc,
                                  const std::vector<size_t>& deps,
                                  size_t concat_index);

                size_t build_slice(const mkldnn::memory::desc& input_desc,
                                   const mkldnn::memory::desc& result_desc,
                                   const ngraph::Coordinate& lower_bounds,
                                   const ngraph::Shape& result_shape);

                void build_slice(std::vector<mkldnn::primitive*>& mkldnn_primitives,
                                 const mkldnn::memory::desc& input_desc,
                                 const mkldnn::memory::desc& result_desc,
                                 const ngraph::Coordinate& lower_bounds,
                                 const ngraph::Shape& result_shape,
                                 const std::vector<size_t>& deps,
                                 size_t slice_index);

                size_t build_softmax_forward(const mkldnn::memory::desc& input_desc,
                                             const mkldnn::memory::desc& result_desc,
                                             int softmax_axis);

                mkldnn::softmax_forward::desc get_softmax_forward_desc(const ngraph::Node* node);

                void build_softmax_forward(std::vector<mkldnn::primitive*>& mkldnn_primitives,
                                           const mkldnn::softmax_forward::desc& sigmoid_desc,
                                           const std::vector<size_t>& deps,
                                           size_t softmax_index);

                size_t build_leaky_relu(const mkldnn::memory::desc& input_desc,
                                        const mkldnn::memory::desc& result_desc,
                                        float alpha);

                mkldnn::eltwise_forward::desc get_leaky_relu_desc(const ngraph::Node* node);

                void build_leaky_relu(std::vector<mkldnn::primitive*>& mkldnn_primitives,
                                      const mkldnn::eltwise_forward::desc& leaky_relu_desc,
                                      const std::vector<size_t>& deps,
                                      size_t leaky_relu_index);

                size_t build_bounded_relu(const mkldnn::memory::desc& input_desc,
                                          const mkldnn::memory::desc& result_desc,
                                          float alpha);

                mkldnn::eltwise_forward::desc get_bounded_relu_desc(const ngraph::Node* node);

                void build_bounded_relu(std::vector<mkldnn::primitive*>& mkldnn_primitives,
                                        const mkldnn::eltwise_forward::desc& bounded_relu_desc,
                                        const std::vector<size_t>& deps,
                                        size_t bounded_relu_index);

                size_t build_dequantization(const ngraph::Node* node,
                                            const mkldnn::memory::desc& input_desc,
                                            const mkldnn::memory::desc& result_desc);

                size_t build_quantize_reorder(const mkldnn::memory::desc& input_desc,
                                              const mkldnn::memory::desc& result_desc,
                                              const std::vector<float>& scales);

                void build_quantize_reorder(std::vector<mkldnn::primitive*>& mkldnn_primitives,
                                            const mkldnn::memory::desc& input_desc,
                                            const mkldnn::memory::desc& result_desc,
                                            const std::vector<float>& scales,
                                            const std::vector<size_t>& deps,
                                            size_t quantize_index,
                                            const int mask = 0);

                template <typename OP>
                size_t get_scale_index()
                {
                    size_t index = 0;
                    if (std::is_same<OP, ngraph::op::QuantizedConvolution>() ||
                        std::is_same<OP, ngraph::op::QuantizedDot>() ||
                        std::is_same<OP, ngraph::op::QuantizedConvolutionRelu>())
                    {
                        index = 2;
                    }
                    else if (std::is_same<OP, ngraph::op::QuantizedConvolutionBias>() ||
                             std::is_same<OP, ngraph::op::QuantizedDotBias>())
                    {
                        index = 3;
                    }
                    else if (std::is_same<OP, ngraph::op::QuantizedConvolutionBiasAdd>() ||
                             std::is_same<OP, ngraph::op::QuantizedConvolutionBiasSignedAdd>())
                    {
                        index = 4;
                    }
                    NGRAPH_CHECK(index != 0);
                    return index;
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

                template <typename OP>
                bool has_bias()
                {
                    if (std::is_same<OP, ngraph::op::ConvolutionBias>() ||
                        std::is_same<OP, ngraph::op::ConvolutionBiasAdd>() ||
                        std::is_same<OP, ngraph::op::ConvolutionBiasBackpropFiltersBias>() ||
                        std::is_same<OP, ngraph::op::QuantizedConvolutionBias>() ||
                        std::is_same<OP, ngraph::op::QuantizedConvolutionBiasAdd>() ||
                        std::is_same<OP, ngraph::op::QuantizedConvolutionBiasSignedAdd>() ||
                        std::is_same<OP, ngraph::op::QuantizedDotBias>() ||
                        std::is_same<OP, ngraph::op::GroupConvolutionBias>())
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
                bool is_quantized_inner_product()
                {
                    if (std::is_same<OP, ngraph::op::QuantizedDot>() ||
                        std::is_same<OP, ngraph::op::QuantizedDotBias>())
                    {
                        return true;
                    }
                    else
                    {
                        return false;
                    }
                }

                template <typename OP>
                mkldnn::rnn_forward::desc
                    get_rnn_forward_desc(const ngraph::Node* node,
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

                    auto get_mkldnn_rnn_cell_type = [&]() {
                        switch (rnn_node->get_rnn_type())
                        {
                        case rnn_utils::rnntype::vanilla_rnn: return mkldnn::algorithm::vanilla_rnn;
                        case rnn_utils::rnntype::vanilla_gru: return mkldnn::algorithm::vanilla_gru;
                        case rnn_utils::rnntype::vanilla_lstm:
                            return mkldnn::algorithm::vanilla_lstm;
                        default: throw ngraph_error("unsupported mkldnn rnn algorithm");
                        }
                    };

                    auto get_mkldnn_rnn_direction = [&]() {
                        switch (direction)
                        {
                        case 1: return mkldnn::rnn_direction::unidirectional_left2right;
                        case 2: return mkldnn::rnn_direction::bidirectional_concat;
                        default: throw ngraph_error("unsupported mkldnn rnn direction");
                        }
                    };

                    if (out[0].get_shape().size() == 2 &&
                        (out[0].get_shape()[1] != direction * feature_size))
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
                    Shape dst_layer_tz{src_sequence_length_max, batch, direction * feature_size};
                    Shape dst_iter_tz{
                        num_fused_layers, direction, rnn_cell_n_states, batch, feature_size};

                    // We create the memory descriptors used by the user
                    auto src_layer_desc = build_memory_descriptor(
                        src_layer_tz, args[0].get_element_type(), mkldnn::memory::format::tnc);
                    auto src_iter_desc = build_memory_descriptor(
                        src_iter_tz, args[1].get_element_type(), mkldnn::memory::format::ldsnc);
                    auto weights_layer_desc = build_memory_descriptor(
                        wei_layer_tz, args[2].get_element_type(), mkldnn::memory::format::ldigo);
                    auto weights_iter_desc = build_memory_descriptor(
                        wei_iter_tz, args[3].get_element_type(), mkldnn::memory::format::ldigo);
                    auto bias_desc = build_memory_descriptor(
                        bias_tz, args[4].get_element_type(), mkldnn::memory::format::ldgo);
                    auto dst_layer_desc = build_memory_descriptor(
                        dst_layer_tz, out[0].get_element_type(), mkldnn::memory::format::tnc);
                    auto dst_iter_desc = build_memory_descriptor(
                        dst_iter_tz, out[1].get_element_type(), mkldnn::memory::format::ldsnc);

                    mkldnn::rnn_cell::desc rnn_cell_desc(get_mkldnn_rnn_cell_type());
                    return mkldnn::rnn_forward::desc(mkldnn::prop_kind::forward_training,
                                                     rnn_cell_desc,
                                                     get_mkldnn_rnn_direction(),
                                                     src_layer_desc,
                                                     src_iter_desc,
                                                     weights_layer_desc,
                                                     weights_iter_desc,
                                                     bias_desc,
                                                     dst_layer_desc,
                                                     dst_iter_desc);
                }

                template <typename OP>
                mkldnn::convolution_forward::desc
                    get_convolution_forward_desc(const ngraph::Node* node)
                {
                    auto convolution = static_cast<const OP*>(node);
                    // For dilation, MKLDNN wants to know how many elements to insert between, not how far
                    // apart to space the elements like nGraph. So we have to subtract 1 from each pos.
                    Strides window_dilation_strides_adjusted;

                    mkldnn::algorithm convolution_algo = mkldnn_utils::get_conv_algo();

                    if (node->get_input_element_type(0) != element::f32 &&
                        convolution_algo != mkldnn::algorithm::convolution_direct)
                    {
                        convolution_algo = mkldnn::algorithm::convolution_direct;
                    }

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
                            mkldnn::prop_kind::forward_inference,
                            convolution_algo,
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
                            mkldnn::prop_kind::forward_inference,
                            convolution_algo,
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
                size_t inner_product_forward_init(bool with_bias = false);

                template <bool with_bias>
                void build_convolution_forward(std::vector<mkldnn::primitive*>& mkldnn_primitives,
                                               const mkldnn::convolution_forward::desc& desc,
                                               const mkldnn::primitive_attr& attr,
                                               const mkldnn::engine& engine,
                                               const std::vector<size_t>& deps,
                                               size_t conv_idx)
                {
                    size_t input_idx, weights_idx, results_idx, bias_idx;
                    input_idx = deps[0];
                    weights_idx = deps[1];
                    mkldnn_primitives[input_idx] =
                        new mkldnn::memory({{desc.data.src_desc}, engine}, nullptr);
                    mkldnn_primitives[weights_idx] =
                        new mkldnn::memory({{desc.data.weights_desc}, engine}, nullptr);
                    if (with_bias)
                    {
                        bias_idx = deps[2];
                        results_idx = deps[3];
                        mkldnn_primitives[bias_idx] =
                            new mkldnn::memory({{desc.data.bias_desc}, engine}, nullptr);
                    }
                    else
                    {
                        results_idx = deps[2];
                    }
                    mkldnn_primitives[results_idx] =
                        new mkldnn::memory({{desc.data.dst_desc}, engine}, nullptr);

                    mkldnn::primitive* prim;
                    if (with_bias)
                    {
                        prim = new mkldnn::convolution_forward({desc, attr, engine},
                                                               *mkldnn_primitives[input_idx],
                                                               *mkldnn_primitives[weights_idx],
                                                               *mkldnn_primitives[bias_idx],
                                                               *mkldnn_primitives[results_idx]);
                    }
                    else
                    {
                        prim = new mkldnn::convolution_forward({desc, attr, engine},
                                                               *mkldnn_primitives[input_idx],
                                                               *mkldnn_primitives[weights_idx],
                                                               *mkldnn_primitives[results_idx]);
                    }

                    mkldnn_primitives[conv_idx] = prim;
                }

                template <typename OP>
                mkldnn::inner_product_forward::desc
                    get_inner_product_forward_desc(const ngraph::Node* node)
                {
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
                        return mkldnn::inner_product_forward::desc(mkldnn::prop_kind::forward,
                                                                   data_desc,
                                                                   weights_desc,
                                                                   bias_desc,
                                                                   result_desc);
                    }
                    else
                    {
                        return mkldnn::inner_product_forward::desc(
                            mkldnn::prop_kind::forward, data_desc, weights_desc, result_desc);
                    }
                }

                template <typename OP>
                mkldnn::primitive_attr get_inner_product_forward_attr(const ngraph::Node* node)
                {
                    mkldnn::post_ops ops;

                    if (has_relu<OP>(node))
                    {
                        const float ops_scale = 1.f;
                        const float ops_alpha = -0.f; // relu negative slope
                        const float ops_beta = 0.f;
                        ops.append_eltwise(
                            ops_scale, mkldnn::algorithm::eltwise_relu, ops_alpha, ops_beta);
                    }

                    mkldnn::primitive_attr ip_attr;
                    ip_attr.set_post_ops(ops);
                    if (is_quantized_inner_product<OP>())
                    {
                        ip_attr.set_int_output_round_mode(mkldnn::round_mode::round_nearest);
                        ip_attr.set_output_scales(0, get_output_scale<OP, float>(node));
                    }
                    return ip_attr;
                }

                template <bool with_bias>
                void build_inner_product_forward(std::vector<mkldnn::primitive*>& mkldnn_primitives,
                                                 const mkldnn::inner_product_forward::desc& desc,
                                                 const mkldnn::primitive_attr& attr,
                                                 const mkldnn::engine& engine,
                                                 const std::vector<size_t>& deps,
                                                 size_t ip_idx)
                {
                    size_t input_idx, weights_idx, results_idx, bias_idx;
                    input_idx = deps[0];
                    weights_idx = deps[1];
                    mkldnn_primitives[input_idx] =
                        new mkldnn::memory({{desc.data.src_desc}, engine}, nullptr);
                    mkldnn_primitives[weights_idx] =
                        new mkldnn::memory({{desc.data.weights_desc}, engine}, nullptr);
                    if (with_bias)
                    {
                        bias_idx = deps[2];
                        results_idx = deps[3];
                        mkldnn_primitives[bias_idx] =
                            new mkldnn::memory({{desc.data.bias_desc}, engine}, nullptr);
                    }
                    else
                    {
                        results_idx = deps[2];
                    }
                    mkldnn_primitives[results_idx] =
                        new mkldnn::memory({{desc.data.dst_desc}, engine}, nullptr);

                    mkldnn::primitive* prim;
                    if (with_bias)
                    {
                        prim = new mkldnn::inner_product_forward({desc, attr, engine},
                                                                 *mkldnn_primitives[input_idx],
                                                                 *mkldnn_primitives[weights_idx],
                                                                 *mkldnn_primitives[bias_idx],
                                                                 *mkldnn_primitives[results_idx]);
                    }
                    else
                    {
                        prim = new mkldnn::inner_product_forward({desc, attr, engine},
                                                                 *mkldnn_primitives[input_idx],
                                                                 *mkldnn_primitives[weights_idx],
                                                                 *mkldnn_primitives[results_idx]);
                    }

                    mkldnn_primitives[ip_idx] = prim;
                }

                template <typename OP>
                mkldnn::deconvolution_forward::desc
                    get_deconvolutionbias_forward_data(const ngraph::Node* node)
                {
                    auto convolution = static_cast<const OP*>(node);
                    // For dilation, MKLDNN wants to know how many elements to insert between, not how far
                    // apart to space the elements like nGraph. So we have to subtract 1 from each pos.
                    Strides window_dilation_strides_adjusted;

                    for (size_t s : convolution->get_window_dilation_strides_forward())
                    {
                        window_dilation_strides_adjusted.push_back(s - 1);
                    }

                    auto weights_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    // MKLDNN relies on named formats for kernel selection
                    if (weights_desc.data.format == mkldnn_nchw)
                    {
                        weights_desc.data.format = mkldnn_oihw;
                    }
                    if (weights_desc.data.format == mkldnn_ncdhw)
                    {
                        weights_desc.data.format = mkldnn_oidhw;
                    }
                    // MKLDNN deconvolution primivtive needs weights format to be "mkldnn_any"
                    // with any other format it picks reference kernel which is very slow
                    // TODO: check if there's change in MKLDNN primitive format req.
                    weights_desc.data.format = mkldnn_any;

                    auto delta_desc = mkldnn_utils::get_input_mkldnn_md(node, 1);
                    auto bias_desc = mkldnn_utils::get_input_mkldnn_md(node, 2);
                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);
                    mkldnn::algorithm deconvolution_algo = mkldnn_utils::get_deconv_algo();

                    mkldnn::post_ops ops;
                    return mkldnn::deconvolution_forward::desc(
                        mkldnn::prop_kind::forward,
                        deconvolution_algo,
                        delta_desc,
                        weights_desc,
                        bias_desc,
                        result_desc,
                        MKLDNN_DIMS(convolution->get_window_movement_strides_forward()),
                        MKLDNN_DIMS(window_dilation_strides_adjusted),
                        MKLDNN_DIMS(convolution->get_padding_below_forward()),
                        MKLDNN_DIMS(convolution->get_padding_above_forward()),
                        mkldnn::padding_kind::zero);
                }

                template <typename OP>
                mkldnn::convolution_backward_data::desc
                    get_convolution_backward_data_desc(const ngraph::Node* node)
                {
                    auto convolution = static_cast<const OP*>(node);
                    // For dilation, MKLDNN wants to know how many elements to insert between, not how far
                    // apart to space the elements like nGraph. So we have to subtract 1 from each pos.
                    Strides window_dilation_strides_adjusted;

                    for (size_t s : convolution->get_window_dilation_strides_forward())
                    {
                        window_dilation_strides_adjusted.push_back(s - 1);
                    }

                    auto weights_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    // MKLDNN relies on named formats for kernel selection
                    if (weights_desc.data.format == mkldnn_nchw)
                    {
                        weights_desc.data.format = mkldnn_oihw;
                    }
                    if (weights_desc.data.format == mkldnn_ncdhw)
                    {
                        weights_desc.data.format = mkldnn_oidhw;
                    }
                    auto delta_desc = mkldnn_utils::get_input_mkldnn_md(node, 1);
                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);
                    mkldnn::algorithm convolution_algo = mkldnn_utils::get_conv_algo();

                    return mkldnn::convolution_backward_data::desc(
                        convolution_algo,
                        result_desc,
                        weights_desc,
                        delta_desc,
                        MKLDNN_DIMS(convolution->get_window_movement_strides_forward()),
                        MKLDNN_DIMS(window_dilation_strides_adjusted),
                        MKLDNN_DIMS(convolution->get_padding_below_forward()),
                        MKLDNN_DIMS(convolution->get_padding_above_forward()),
                        mkldnn::padding_kind::zero);
                }

                template <typename OP>
                mkldnn::convolution_backward_weights::desc
                    get_convolution_backward_weights_desc(const ngraph::Node* node)
                {
                    auto convolution = static_cast<const OP*>(node);
                    // For dilation, MKLDNN wants to know how many elements to insert between, not how far
                    // apart to space the elements like nGraph. So we have to subtract 1 from each pos.
                    Strides window_dilation_strides_adjusted;

                    for (size_t s : convolution->get_window_dilation_strides_forward())
                    {
                        window_dilation_strides_adjusted.push_back(s - 1);
                    }

                    auto in_data_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto in_delta_desc = mkldnn_utils::get_input_mkldnn_md(node, 1);
                    auto out_weights_delta_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);
                    mkldnn::algorithm convolution_algo = mkldnn_utils::get_conv_algo();
                    if (has_bias<OP>())
                    {
                        auto out_bias_delta_desc = mkldnn_utils::get_output_mkldnn_md(node, 1);

                        return mkldnn::convolution_backward_weights::desc(
                            convolution_algo,
                            in_data_desc,
                            out_weights_delta_desc,
                            out_bias_delta_desc,
                            in_delta_desc,
                            MKLDNN_DIMS(convolution->get_window_movement_strides_forward()),
                            MKLDNN_DIMS(window_dilation_strides_adjusted),
                            MKLDNN_DIMS(convolution->get_padding_below_forward()),
                            MKLDNN_DIMS(convolution->get_padding_above_forward()),
                            mkldnn::padding_kind::zero);
                    }
                    else
                    {
                        return mkldnn::convolution_backward_weights::desc(
                            convolution_algo,
                            in_data_desc,
                            out_weights_delta_desc,
                            in_delta_desc,
                            MKLDNN_DIMS(convolution->get_window_movement_strides_forward()),
                            MKLDNN_DIMS(window_dilation_strides_adjusted),
                            MKLDNN_DIMS(convolution->get_padding_below_forward()),
                            MKLDNN_DIMS(convolution->get_padding_above_forward()),
                            mkldnn::padding_kind::zero);
                    }
                }

                template <typename OP>
                mkldnn::convolution_forward::desc
                    get_convolution_forward_desc_for_backward_op(const ngraph::Node* node)
                {
                    auto convolution = static_cast<const OP*>(node);
                    // For dilation, MKLDNN wants to know how many elements to insert between, not how far
                    // apart to space the elements like nGraph. So we have to subtract 1 from each pos.
                    Strides window_dilation_strides_adjusted;

                    for (size_t s : convolution->get_window_dilation_strides_forward())
                    {
                        window_dilation_strides_adjusted.push_back(s - 1);
                    }

                    mkldnn::algorithm convolution_algo = mkldnn_utils::get_conv_algo();
                    if (std::is_same<OP, ngraph::op::ConvolutionBackpropData>())
                    {
                        auto weights_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                        // MKLDNN relies on named formats for kernel selection
                        if (weights_desc.data.format == mkldnn_nchw)
                        {
                            weights_desc.data.format = mkldnn_oihw;
                        }
                        if (weights_desc.data.format == mkldnn_ncdhw)
                        {
                            weights_desc.data.format = mkldnn_oidhw;
                        }
                        auto delta_desc = mkldnn_utils::get_input_mkldnn_md(node, 1);
                        auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);

                        return mkldnn::convolution_forward::desc(
                            mkldnn::prop_kind::forward,
                            convolution_algo,
                            result_desc,
                            weights_desc,
                            delta_desc,
                            MKLDNN_DIMS(convolution->get_window_movement_strides_forward()),
                            MKLDNN_DIMS(window_dilation_strides_adjusted),
                            MKLDNN_DIMS(convolution->get_padding_below_forward()),
                            MKLDNN_DIMS(convolution->get_padding_above_forward()),
                            mkldnn::padding_kind::zero);
                    }
                    else if (std::is_same<OP, ngraph::op::ConvolutionBackpropFilters>())
                    {
                        auto in_data_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                        auto in_delta_desc = mkldnn_utils::get_input_mkldnn_md(node, 1);
                        auto out_weights_delta_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);
                        return mkldnn::convolution_forward::desc(
                            mkldnn::prop_kind::forward,
                            convolution_algo,
                            in_data_desc,
                            out_weights_delta_desc,
                            in_delta_desc,
                            MKLDNN_DIMS(convolution->get_window_movement_strides_forward()),
                            MKLDNN_DIMS(window_dilation_strides_adjusted),
                            MKLDNN_DIMS(convolution->get_padding_below_forward()),
                            MKLDNN_DIMS(convolution->get_padding_above_forward()),
                            mkldnn::padding_kind::zero);
                    }
                    else
                    {
                        auto in_data_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                        auto in_delta_desc = mkldnn_utils::get_input_mkldnn_md(node, 1);
                        auto out_weights_delta_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);
                        auto out_bias_delta_desc = mkldnn_utils::get_output_mkldnn_md(node, 1);

                        return mkldnn::convolution_forward::desc(
                            mkldnn::prop_kind::forward,
                            convolution_algo,
                            in_data_desc,
                            out_weights_delta_desc,
                            out_bias_delta_desc,
                            in_delta_desc,
                            MKLDNN_DIMS(convolution->get_window_movement_strides_forward()),
                            MKLDNN_DIMS(window_dilation_strides_adjusted),
                            MKLDNN_DIMS(convolution->get_padding_below_forward()),
                            MKLDNN_DIMS(convolution->get_padding_above_forward()),
                            mkldnn::padding_kind::zero);
                    }
                }

            private:
                std::vector<mkldnn::primitive*> m_mkldnn_primitives;
                std::vector<mkldnn::primitive*> m_mkldnn_primitives_cg;
                std::vector<mkldnn::stream> m_mkldnn_streams;
                std::unordered_map<size_t, std::vector<size_t>> m_primitive_deps;
                std::unordered_map<size_t, std::vector<size_t>> m_primitive_deps_cg;
                std::vector<std::unique_ptr<MKLDNNWorkspace>> m_workspaces;
                std::vector<char*> m_workspace_bufs;
                size_t m_workspaces_size = 0;
                size_t m_mkldnn_descriptors_size = 0;
            };
        }
    }
}
