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

// For direct execution, we reserve space for primitives then create those primitives the first
// time functor is called. This could be extended to create primitives when shapes are changed.
// Different ops need different numbers of primitives.

#pragma once

#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include <dnnl.hpp>

#include "ngraph/coordinate_diff.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/avg_pool.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/conv_fused.hpp"
#include "ngraph/op/convolution.hpp"
#include "ngraph/op/dequantize.hpp"
#include "ngraph/op/experimental/quantized_conv_bias.hpp"
#include "ngraph/op/experimental/quantized_conv_relu.hpp"
#include "ngraph/op/experimental/quantized_dot_bias.hpp"
#include "ngraph/op/gelu.hpp"
#include "ngraph/op/group_conv.hpp"
#include "ngraph/op/lrn.hpp"
#include "ngraph/op/max_pool.hpp"
#include "ngraph/op/quantize.hpp"
#include "ngraph/op/quantized_convolution.hpp"
#include "ngraph/op/quantized_dot.hpp"
#include "ngraph/op/softmax.hpp"
#include "ngraph/runtime/cpu/cpu_executor.hpp"
#include "ngraph/runtime/cpu/cpu_tensor_wrapper.hpp"
#include "ngraph/runtime/cpu/dnnl_invoke.hpp"
#include "ngraph/runtime/cpu/dnnl_utils.hpp"
#include "ngraph/runtime/cpu/op/bounded_relu.hpp"
#include "ngraph/runtime/cpu/op/conv_add.hpp"
#include "ngraph/runtime/cpu/op/conv_relu.hpp"
#include "ngraph/runtime/cpu/op/deconv.hpp"
#include "ngraph/runtime/cpu/op/group_conv_bias.hpp"
#include "ngraph/runtime/cpu/op/leaky_relu.hpp"
#include "ngraph/runtime/cpu/op/quantized_matmul.hpp"
#include "ngraph/runtime/cpu/op/rnn_utils.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/strides.hpp"
#include "ngraph/type/element_type.hpp"
#include "ngraph/util.hpp"

#define DNNL_DIMS(X) dnnl::memory::dims(X.begin(), X.end())
// DNNL relies on named formats for kernel selection

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            class CPU_ExternalFunction;
            class TensorWrapper;

            // TODO (nbpatel) Templatize the return type when we have double scales
            template <typename OP>
            inline std::vector<float> extract_scale_value(const ngraph::Node* node, int index)
            {
                auto qc = static_cast<const OP*>(node);
                std::vector<float> scale_val = {1.0f};
                auto scale_const_op =
                    as_type_ptr<ngraph::op::v0::Constant>(qc->get_arguments()[index]);
                if (scale_const_op != nullptr)
                {
                    scale_val = scale_const_op->template get_vector<float>();
                }

                return scale_val;
            }

            template <typename OP,
                      typename std::enable_if<
                          (std::is_same<OP, ngraph::op::v0::Convolution>::value ||
                           std::is_same<OP, ngraph::op::v0::QuantizedConvolution>::value ||
                           std::is_same<OP, ngraph::op::v0::GroupConvolution>::value),
                          std::nullptr_t>::type = nullptr>
            bool has_relu(const ngraph::Node* /* node */)
            {
                return false;
            }

            template <typename OP,
                      typename std::enable_if<
                          (!std::is_same<OP, ngraph::op::v0::Convolution>::value &&
                           !std::is_same<OP, ngraph::op::v0::QuantizedConvolution>::value &&
                           !std::is_same<OP, ngraph::op::v0::GroupConvolution>::value),
                          std::nullptr_t>::type = nullptr>
            bool has_relu(const ngraph::Node* node)
            {
                return static_cast<const OP*>(node)->with_relu();
            }

            class DNNLWorkspace
            {
            public:
                DNNLWorkspace(size_t size) { buf = reinterpret_cast<char*>(ngraph_malloc(size)); }
                ~DNNLWorkspace() { ngraph_free(buf); }
                char* buf;

                DNNLWorkspace(const DNNLWorkspace&) = delete;
                DNNLWorkspace(DNNLWorkspace&&) = delete;
                DNNLWorkspace& operator=(const DNNLWorkspace&) = delete;
            };

            class DNNLEmitter
            {
            public:
                DNNLEmitter() {}
                ~DNNLEmitter();

                const std::vector<dnnl::primitive*>& get_dnnl_primitives() const;
                std::vector<dnnl::primitive*>& get_dnnl_primitives();
                const std::vector<dnnl::memory*>& get_dnnl_memories() const;
                const std::vector<char*>& get_dnnl_workspaces();
                const std::vector<dnnl::memory::desc*>& get_dnnl_scratchpad_mds() const;

                // reserve the space for primitives for each op, different op requires different
                // number of primitives.
                // some ops require a new workspace.
                size_t reserve_primitive_space(size_t count,
                                               bool fwd_bwd = false,
                                               bool new_workspace = false);
                size_t insert_primitive(dnnl::primitive* primitive);
                size_t insert_memory(dnnl::memory* memory);
                size_t insert_workspace(std::unique_ptr<DNNLWorkspace>& workspace);
                size_t insert_workspace(std::vector<char*>& dnnl_workspaces,
                                        std::unique_ptr<DNNLWorkspace>& workspace);
                size_t insert_scratchpad_md(dnnl::memory::desc* md);
                const std::vector<size_t>& get_primitive_deps(size_t index) const;
                size_t reserve_workspace();
                void reserve_descriptor_space(size_t count);
                size_t get_dnnl_descriptors_size();
                std::vector<size_t>& get_primitive_deps(size_t index);
                size_t get_max_scratchpad_size() const;

                size_t build_quantized_inner_product_forward(
                    const dnnl::memory::desc& input_data_desc,
                    const dnnl::memory::desc& weights_desc,
                    const dnnl::memory::desc& result_desc,
                    const float scale,
                    const dnnl::post_ops& pops = dnnl::post_ops());

                size_t build_quantized_inner_product_forward(
                    const dnnl::memory::desc& input_data_desc,
                    const dnnl::memory::desc& weights_desc,
                    const dnnl::memory::desc& bias_desc,
                    const dnnl::memory::desc& result_desc,
                    const float scale,
                    const dnnl::post_ops& pops = dnnl::post_ops());

                dnnl::memory::desc
                    build_blocked_memory_descriptor(const dnnl::memory::dims& dim,
                                                    const dnnl::memory::dims& strides,
                                                    dnnl::memory::data_type dtype) const;

                template <typename OP>
                size_t build_deconvolution(const ngraph::Node* node,
                                           const std::vector<TensorWrapper>& /* args */,
                                           const std::vector<TensorWrapper>& /* out */)
                {
                    auto convolution = static_cast<const OP*>(node);

                    // For dilation, DNNL wants to know how many elements to insert between, not
                    // how far apart to space the elements like nGraph. So we have to subtract 1
                    // from each pos.
                    Strides window_dilation_strides_adjusted;

                    for (size_t s : convolution->get_window_dilation_strides_forward())
                    {
                        window_dilation_strides_adjusted.push_back(s - 1);
                    }

                    auto weights_desc = dnnl_utils::get_input_dnnl_md(node, 0);
                    CHANGE_FORMAT
                    auto data_desc = dnnl_utils::get_input_dnnl_md(node, 1);

                    auto result_desc = dnnl_utils::get_output_dnnl_md(node, 0);

                    dnnl::post_ops ops;

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
                            ops_scale, dnnl::algorithm::eltwise_relu, ops_alpha, ops_beta);
                    }

                    if (std::is_same<OP, ngraph::op::DeconvolutionBias>())
                    {
                        auto bias_desc = dnnl_utils::get_input_dnnl_md(node, 2);
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
                                           const std::vector<TensorWrapper>& /* args */,
                                           const std::vector<TensorWrapper>& /* out */)
                {
                    auto data_desc = dnnl_utils::get_input_dnnl_md(node, 0);
                    auto weights_desc = dnnl_utils::get_input_dnnl_md(node, 1);
                    CHANGE_FORMAT

                    auto result_desc = dnnl_utils::get_output_dnnl_md(node, 0);

                    dnnl::post_ops ops;

                    if (std::is_same<OP, ngraph::op::v0::QuantizedDotBias>() &&
                        has_relu<ngraph::op::v0::QuantizedDotBias>(node))
                    {
                        const float ops_scale = 1.f;
                        const float ops_alpha = -0.f; // relu negative slope
                        const float ops_beta = 0.f;
                        ops.append_eltwise(
                            ops_scale, dnnl::algorithm::eltwise_relu, ops_alpha, ops_beta);
                    }

                    if (std::is_same<OP, ngraph::op::v0::QuantizedDot>())
                    {
                        auto scale_val = extract_scale_value<OP>(node, 2);
                        return build_quantized_inner_product_forward(
                            data_desc, weights_desc, result_desc, scale_val[0], ops);
                    }
                    else if (std::is_same<OP, ngraph::op::v0::QuantizedDotBias>())
                    {
                        auto scale_val = extract_scale_value<OP>(node, 3);
                        auto bias_desc = dnnl_utils::get_input_dnnl_md(node, 2);
                        return build_quantized_inner_product_forward(
                            data_desc, weights_desc, bias_desc, result_desc, scale_val[0], ops);
                    }
                    else
                    {
                        throw ngraph_error("unsupported inner_product");
                    }
                }

                template <typename OP>
                dnnl::pooling_forward::desc get_avg_pooling_forward_desc(const ngraph::Node* node,
                                                                         bool training)
                {
                    auto pool = static_cast<const OP*>(node);

                    auto window_shape = pool->get_window_shape();
                    auto window_strides = pool->get_window_movement_strides();
                    auto padding_below = pool->get_padding_below();
                    auto padding_above = pool->get_padding_above();
                    auto include_padding_in_avg_computation =
                        pool->get_include_padding_in_avg_computation();

                    auto input_desc = dnnl_utils::get_input_dnnl_md(node, 0);
                    auto result_desc = dnnl_utils::get_output_dnnl_md(node, 0);

                    if (training)
                    {
                        return dnnl::pooling_forward::desc(
                            dnnl::prop_kind::forward_training,
                            (include_padding_in_avg_computation
                                 ? dnnl::algorithm::pooling_avg_include_padding
                                 : dnnl::algorithm::pooling_avg_exclude_padding),
                            result_desc,
                            input_desc,
                            dnnl::memory::dims(window_strides.begin(), window_strides.end()),
                            dnnl::memory::dims(window_shape.begin(), window_shape.end()),
                            dnnl::memory::dims(padding_below.begin(), padding_below.end()),
                            dnnl::memory::dims(padding_above.begin(), padding_above.end()) PADDING);
                    }
                    else
                    {
                        return dnnl::pooling_forward::desc(
                            dnnl::prop_kind::forward_inference,
                            (include_padding_in_avg_computation
                                 ? dnnl::algorithm::pooling_avg_include_padding
                                 : dnnl::algorithm::pooling_avg_exclude_padding),
                            input_desc,
                            result_desc,
                            dnnl::memory::dims(window_strides.begin(), window_strides.end()),
                            dnnl::memory::dims(window_shape.begin(), window_shape.end()),
                            dnnl::memory::dims(padding_below.begin(), padding_below.end()),
                            dnnl::memory::dims(padding_above.begin(), padding_above.end()) PADDING);
                    }
                }

                template <typename OP>
                dnnl::pooling_forward::desc get_max_pooling_forward_desc(const ngraph::Node* node,
                                                                         bool training)
                {
                    auto pool = static_cast<const OP*>(node);

                    auto window_shape = pool->get_window_shape();
                    auto window_strides = pool->get_window_movement_strides();
                    auto padding_below = pool->get_padding_below();
                    auto padding_above = pool->get_padding_above();

                    if (training)
                    {
                        auto diff_dst_desc = dnnl_utils::get_input_dnnl_md(node, 1);
                        auto diff_src_desc = dnnl_utils::get_output_dnnl_md(node, 0);

                        return dnnl::pooling_forward::desc(
                            dnnl::prop_kind::forward_training,
                            dnnl::algorithm::pooling_max,
                            diff_src_desc,
                            diff_dst_desc,
                            dnnl::memory::dims(window_strides.begin(), window_strides.end()),
                            dnnl::memory::dims(window_shape.begin(), window_shape.end()),
                            dnnl::memory::dims(padding_below.begin(), padding_below.end()),
                            dnnl::memory::dims(padding_above.begin(), padding_above.end()) PADDING);
                    }
                    else
                    {
                        auto input_desc = dnnl_utils::get_input_dnnl_md(node, 0);
                        auto result_desc = dnnl_utils::get_output_dnnl_md(node, 0);

                        return dnnl::pooling_forward::desc(
                            dnnl::prop_kind::forward_inference,
                            dnnl::algorithm::pooling_max,
                            input_desc,
                            result_desc,
                            dnnl::memory::dims(window_strides.begin(), window_strides.end()),
                            dnnl::memory::dims(window_shape.begin(), window_shape.end()),
                            dnnl::memory::dims(padding_below.begin(), padding_below.end()),
                            dnnl::memory::dims(padding_above.begin(), padding_above.end()) PADDING);
                    }
                }

                template <typename OP>
                dnnl::pooling_backward::desc get_avg_pooling_backward_desc(const ngraph::Node* node)
                {
                    auto pool = static_cast<const OP*>(node);

                    auto window_shape = pool->get_window_shape();
                    auto window_strides = pool->get_window_movement_strides();
                    auto padding_below = pool->get_padding_below();
                    auto padding_above = pool->get_padding_above();
                    auto include_padding_in_avg_computation =
                        pool->get_include_padding_in_avg_computation();

                    auto diff_dst_desc = dnnl_utils::get_input_dnnl_md(node, 0);
                    auto diff_src_desc = dnnl_utils::get_output_dnnl_md(node, 0);

                    return dnnl::pooling_backward::desc(
                        (include_padding_in_avg_computation
                             ? dnnl::algorithm::pooling_avg_include_padding
                             : dnnl::algorithm::pooling_avg_exclude_padding),
                        diff_src_desc,
                        diff_dst_desc,
                        dnnl::memory::dims(window_strides.begin(), window_strides.end()),
                        dnnl::memory::dims(window_shape.begin(), window_shape.end()),
                        dnnl::memory::dims(padding_below.begin(), padding_below.end()),
                        dnnl::memory::dims(padding_above.begin(), padding_above.end()) PADDING);
                }

                template <typename OP>
                dnnl::pooling_forward::desc
                    get_max_pooling_with_indices_forward_desc(const ngraph::Node* node)
                {
                    auto pool = static_cast<const OP*>(node);

                    auto window_shape = pool->get_window_shape();
                    auto window_strides = pool->get_window_movement_strides();
                    auto padding_below = pool->get_padding_below();
                    auto padding_above = pool->get_padding_above();

                    auto input_desc = dnnl_utils::get_input_dnnl_md(node, 0);
                    auto result_desc = dnnl_utils::get_output_dnnl_md(node, 0);

                    return dnnl::pooling_forward::desc(
                        dnnl::prop_kind::forward_training,
                        dnnl::algorithm::pooling_max,
                        input_desc,
                        result_desc,
                        dnnl::memory::dims(window_strides.begin(), window_strides.end()),
                        dnnl::memory::dims(window_shape.begin(), window_shape.end()),
                        dnnl::memory::dims(padding_below.begin(), padding_below.end()),
                        dnnl::memory::dims(padding_above.begin(), padding_above.end()) PADDING);
                }

                template <typename OP>
                dnnl::pooling_backward::desc get_max_pooling_backward_desc(const ngraph::Node* node)
                {
                    auto pool = static_cast<const OP*>(node);

                    auto window_shape = pool->get_window_shape();
                    auto window_strides = pool->get_window_movement_strides();
                    auto padding_below = pool->get_padding_below();
                    auto padding_above = pool->get_padding_above();

                    auto diff_dst_desc = dnnl_utils::get_input_dnnl_md(node, 1);
                    auto diff_src_desc = dnnl_utils::get_output_dnnl_md(node, 0);

                    return dnnl::pooling_backward::desc(
                        dnnl::algorithm::pooling_max,
                        diff_src_desc,
                        diff_dst_desc,
                        dnnl::memory::dims(window_strides.begin(), window_strides.end()),
                        dnnl::memory::dims(window_shape.begin(), window_shape.end()),
                        dnnl::memory::dims(padding_below.begin(), padding_below.end()),
                        dnnl::memory::dims(padding_above.begin(), padding_above.end()) PADDING);
                }

                size_t build_reorder(const dnnl::memory::desc& input_desc,
                                     const dnnl::memory::desc& result_desc);

                dnnl::lrn_forward::desc get_lrn_forward_desc(const ngraph::Node* node);

                dnnl::eltwise_forward::desc get_relu_forward_desc(const ngraph::Node* node);

                dnnl::eltwise_backward::desc get_relu_backward_desc(const ngraph::Node* node);

                dnnl::eltwise_forward::desc get_sigmoid_forward_desc(const ngraph::Node* node,
                                                                     bool backward_op);

                dnnl::eltwise_backward::desc get_sigmoid_backward_desc(const ngraph::Node* node);

                dnnl::sum::primitive_desc get_elementwise_add_desc(const ngraph::Node* node);

                template <typename OP>
                dnnl::batch_normalization_forward::desc
                    get_batchnorm_forward_desc(const ngraph::Node* node, bool training_with_3args)
                {
                    const OP* batchnorm = static_cast<const OP*>(node);
                    auto eps = batchnorm->get_eps_value();

                    if (training_with_3args)
                    {
                        auto input_desc = dnnl_utils::get_input_dnnl_md(node, 2);
                        return dnnl::batch_normalization_forward::desc(
                            dnnl::prop_kind::forward_training,
                            input_desc,
                            eps,
                            dnnl::BN_FLAG_CLASS::use_scale_shift);
                    }
                    else
                    {
                        auto input_desc = dnnl_utils::get_input_dnnl_md(node, 2);
                        return dnnl::batch_normalization_forward::desc(
                            dnnl::prop_kind::forward_training,
                            input_desc,
                            eps,
                            dnnl::BN_FLAG_CLASS::use_scale_shift |
                                dnnl::BN_FLAG_CLASS::use_global_stats);
                    }
                }

                dnnl::batch_normalization_backward::desc
                    get_batchnorm_backward_desc(const ngraph::Node* node);

                dnnl::softmax_forward::desc get_softmax_forward_desc(const ngraph::Node* node);

                dnnl::eltwise_forward::desc get_leaky_relu_desc(const ngraph::Node* node);

                dnnl::eltwise_forward::desc get_bounded_relu_desc(const ngraph::Node* node);

                dnnl::eltwise_forward::desc get_gelu_forward_desc(const ngraph::Node* node);

                dnnl::eltwise_backward::desc get_gelu_backward_desc(const ngraph::Node* node);

                size_t build_dequantization(const ngraph::Node* node,
                                            const dnnl::memory::desc& input_desc,
                                            const dnnl::memory::desc& result_desc);

                size_t build_quantize_reorder(const dnnl::memory::desc& input_desc,
                                              const dnnl::memory::desc& result_desc,
                                              const std::vector<float>& scales);

                template <typename OP>
                size_t get_scale_index()
                {
                    size_t index = 0;
                    if (std::is_same<OP, ngraph::op::v0::Quantize>() ||
                        std::is_same<OP, ngraph::op::v0::Dequantize>())
                    {
                        index = 1;
                    }
                    else if (std::is_same<OP, ngraph::op::v0::QuantizedConvolution>() ||
                             std::is_same<OP, ngraph::op::QuantizedMatmul>() ||
                             std::is_same<OP, ngraph::op::v0::QuantizedConvolutionRelu>())
                    {
                        index = 2;
                    }
                    else if (std::is_same<OP, ngraph::op::v0::QuantizedConvolutionBias>() ||
                             std::is_same<OP, ngraph::op::v0::QuantizedDotBias>())
                    {
                        index = 3;
                    }
                    else if (std::is_same<OP, ngraph::op::v0::QuantizedConvolutionBiasAdd>() ||
                             std::is_same<OP, ngraph::op::v0::QuantizedConvolutionBiasSignedAdd>())
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
                    auto scale_const_op =
                        as_type_ptr<ngraph::op::v0::Constant>(node->get_arguments()[index]);
                    if (scale_const_op != nullptr)
                    {
                        scale_val = scale_const_op->template get_vector<T>();
                    }

                    return scale_val;
                }

                template <typename OP>
                bool has_bias()
                {
                    if (std::is_same<OP, ngraph::op::v0::ConvolutionBias>() ||
                        std::is_same<OP, ngraph::op::v0::ConvolutionBiasAdd>() ||
                        std::is_same<OP, ngraph::op::v0::ConvolutionBiasBackpropFiltersBias>() ||
                        std::is_same<OP, ngraph::op::v0::QuantizedConvolutionBias>() ||
                        std::is_same<OP, ngraph::op::v0::QuantizedConvolutionBiasAdd>() ||
                        std::is_same<OP, ngraph::op::v0::QuantizedConvolutionBiasSignedAdd>() ||
                        std::is_same<OP, ngraph::op::v0::QuantizedDotBias>() ||
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
                    if (std::is_same<OP, ngraph::op::v0::QuantizedConvolution>() ||
                        std::is_same<OP, ngraph::op::v0::QuantizedConvolutionRelu>() ||
                        std::is_same<OP, ngraph::op::v0::QuantizedConvolutionBias>() ||
                        std::is_same<OP, ngraph::op::v0::QuantizedConvolutionBiasAdd>() ||
                        std::is_same<OP, ngraph::op::v0::QuantizedConvolutionBiasSignedAdd>())
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
                    if (std::is_same<OP, ngraph::op::QuantizedMatmul>() ||
                        std::is_same<OP, ngraph::op::v0::QuantizedDotBias>())
                    {
                        return true;
                    }
                    else
                    {
                        return false;
                    }
                }

                template <typename OP>
                dnnl::convolution_forward::desc
                    get_convolution_forward_desc(const ngraph::Node* node)
                {
                    auto convolution = static_cast<const OP*>(node);
                    // For dilation, DNNL wants to know how many elements to insert between, not
                    // how far apart to space the elements like nGraph. So we have to subtract 1
                    // from each pos.
                    Strides window_dilation_strides_adjusted;

                    dnnl::algorithm convolution_algo = dnnl_utils::get_conv_algo();

                    if ((node->get_input_element_type(0) != element::f32 &&
                         convolution_algo != dnnl::algorithm::convolution_direct) ||
                        convolution->get_input_shape(0)[1] <= 8)
                    {
                        convolution_algo = dnnl::algorithm::convolution_direct;
                    }

                    for (size_t s : convolution->get_window_dilation_strides())
                    {
                        window_dilation_strides_adjusted.push_back(s - 1);
                    }

                    auto data_desc = dnnl_utils::get_input_dnnl_md(node, 0);
                    auto weights_desc = dnnl_utils::get_input_dnnl_md(node, 1);
                    CHANGE_FORMAT

                    auto result_desc = dnnl_utils::get_output_dnnl_md(node, 0);

                    if (has_bias<OP>())
                    {
                        auto bias_desc = dnnl_utils::get_input_dnnl_md(node, 2);
                        return dnnl::convolution_forward::desc(
                            dnnl::prop_kind::forward_inference,
                            convolution_algo,
                            data_desc,
                            weights_desc,
                            bias_desc,
                            result_desc,
                            DNNL_DIMS(convolution->get_window_movement_strides()),
                            DNNL_DIMS(window_dilation_strides_adjusted),
                            DNNL_DIMS(convolution->get_padding_below()),
                            DNNL_DIMS(convolution->get_padding_above()) PADDING);
                    }
                    else
                    {
                        return dnnl::convolution_forward::desc(
                            dnnl::prop_kind::forward_inference,
                            convolution_algo,
                            data_desc,
                            weights_desc,
                            result_desc,
                            DNNL_DIMS(convolution->get_window_movement_strides()),
                            DNNL_DIMS(window_dilation_strides_adjusted),
                            DNNL_DIMS(convolution->get_padding_below()),
                            DNNL_DIMS(convolution->get_padding_above()) PADDING);
                    }
                }

                template <typename OP>
                dnnl::primitive_attr get_convolution_forward_attr(const ngraph::Node* node)
                {
                    dnnl::post_ops ops;

                    if (std::is_same<OP, ngraph::op::v0::ConvolutionBiasAdd>() ||
                        std::is_same<OP, ngraph::op::ConvolutionAdd>())
                    {
                        ops.append_sum(1.f);
                    }

                    if (std::is_same<OP, ngraph::op::v0::QuantizedConvolutionBiasAdd>() ||
                        std::is_same<OP, ngraph::op::v0::QuantizedConvolutionBiasSignedAdd>())
                    {
                        auto sum_scale_val =
                            extract_scale_value<ngraph::op::v0::QuantizedConvolutionBiasAdd>(node,
                                                                                             5);
                        ops.append_sum(sum_scale_val[0]);
                    }

                    if (has_relu<OP>(node))
                    {
                        const float ops_scale = 1.f;
                        const float ops_alpha = -0.f; // relu negative slope
                        const float ops_beta = 0.f;
                        ops.append_eltwise(
                            ops_scale, dnnl::algorithm::eltwise_relu, ops_alpha, ops_beta);
                    }

                    dnnl::primitive_attr attr;
                    attr.set_post_ops(ops);
                    if (is_quantized_conv<OP>())
                    {
                        SET_ROUND_MODE
                        attr.set_output_scales(0, get_output_scale<OP, float>(node));
                    }
                    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
                    return attr;
                }

                size_t convolution_forward_init(bool with_bias = false);
                size_t inner_product_forward_init(bool with_bias = false);

                template <typename OP>
                dnnl::inner_product_forward::desc
                    get_inner_product_forward_desc(const ngraph::Node* node)
                {
                    auto data_desc = dnnl_utils::get_input_dnnl_md(node, 0);
                    auto weights_desc = dnnl_utils::get_input_dnnl_md(node, 1);
                    CHANGE_FORMAT

                    auto result_desc = dnnl_utils::get_output_dnnl_md(node, 0);

                    if (has_bias<OP>())
                    {
                        auto bias_desc = dnnl_utils::get_input_dnnl_md(node, 2);
                        return dnnl::inner_product_forward::desc(dnnl::prop_kind::forward,
                                                                 data_desc,
                                                                 weights_desc,
                                                                 bias_desc,
                                                                 result_desc);
                    }
                    else
                    {
                        return dnnl::inner_product_forward::desc(
                            dnnl::prop_kind::forward, data_desc, weights_desc, result_desc);
                    }
                }

                template <typename OP>
                dnnl::primitive_attr get_inner_product_forward_attr(const ngraph::Node* node)
                {
                    dnnl::post_ops ops;

                    if (std::is_same<OP, ngraph::op::v0::QuantizedDotBias>() &&
                        has_relu<ngraph::op::v0::QuantizedDotBias>(node))
                    {
                        const float ops_scale = 1.f;
                        const float ops_alpha = -0.f; // relu negative slope
                        const float ops_beta = 0.f;
                        ops.append_eltwise(
                            ops_scale, dnnl::algorithm::eltwise_relu, ops_alpha, ops_beta);
                    }

                    dnnl::primitive_attr attr;
                    attr.set_post_ops(ops);
                    if (is_quantized_inner_product<OP>())
                    {
                        SET_ROUND_MODE
                        attr.set_output_scales(0, get_output_scale<OP, float>(node));
                    }
                    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
                    return attr;
                }

                template <typename OP>
                dnnl::deconvolution_forward::desc
                    get_deconvolutionbias_forward_data(const ngraph::Node* node)
                {
                    auto convolution = static_cast<const OP*>(node);
                    // For dilation, DNNL wants to know how many elements to insert between, not
                    // how far apart to space the elements like nGraph. So we have to subtract 1
                    // from each pos.
                    Strides window_dilation_strides_adjusted;

                    for (size_t s : convolution->get_window_dilation_strides_forward())
                    {
                        window_dilation_strides_adjusted.push_back(s - 1);
                    }

                    auto weights_desc = dnnl_utils::get_input_dnnl_md(node, 0);

                    CHANGE_FORMAT
                    // DNNL deconvolution primitive needs weights format to be "dnnl_any"
                    // with any other format it picks reference kernel which is very slow
                    // TODO: check if there's change in DNNL primitive format req.

                    weights_desc.data.FORMAT_KIND = FORMAT_ANY;

                    auto delta_desc = dnnl_utils::get_input_dnnl_md(node, 1);
                    auto bias_desc = dnnl_utils::get_input_dnnl_md(node, 2);
                    auto result_desc = dnnl_utils::get_output_dnnl_md(node, 0);
                    dnnl::algorithm deconvolution_algo = dnnl_utils::get_deconv_algo();

                    dnnl::post_ops ops;
                    return dnnl::deconvolution_forward::desc(
                        dnnl::prop_kind::forward,
                        deconvolution_algo,
                        delta_desc,
                        weights_desc,
                        bias_desc,
                        result_desc,
                        DNNL_DIMS(convolution->get_window_movement_strides_forward()),
                        DNNL_DIMS(window_dilation_strides_adjusted),
                        DNNL_DIMS(convolution->get_padding_below_forward()),
                        DNNL_DIMS(convolution->get_padding_above_forward()) PADDING);
                }

                template <typename OP>
                dnnl::convolution_backward_data::desc
                    get_convolution_backward_data_desc(const ngraph::Node* node)
                {
                    auto convolution = static_cast<const OP*>(node);
                    // For dilation, DNNL wants to know how many elements to insert between, not
                    // how far apart to space the elements like nGraph. So we have to subtract 1
                    // from each pos.
                    Strides window_dilation_strides_adjusted;

                    for (size_t s : convolution->get_window_dilation_strides_forward())
                    {
                        window_dilation_strides_adjusted.push_back(s - 1);
                    }

                    auto weights_desc = dnnl_utils::get_input_dnnl_md(node, 0);
                    CHANGE_FORMAT

                    auto diff_dst_desc = dnnl_utils::get_input_dnnl_md(node, 1);
                    auto diff_src_desc = dnnl_utils::get_output_dnnl_md(node, 0);
                    dnnl::algorithm convolution_algo = dnnl_utils::get_conv_algo();

                    return dnnl::convolution_backward_data::desc(
                        convolution_algo,
                        diff_src_desc,
                        weights_desc,
                        diff_dst_desc,
                        DNNL_DIMS(convolution->get_window_movement_strides_forward()),
                        DNNL_DIMS(window_dilation_strides_adjusted),
                        DNNL_DIMS(convolution->get_padding_below_forward()),
                        DNNL_DIMS(convolution->get_padding_above_forward()) PADDING);
                }

                template <typename OP>
                dnnl::convolution_backward_weights::desc
                    get_convolution_backward_weights_desc(const ngraph::Node* node)
                {
                    auto convolution = static_cast<const OP*>(node);
                    // For dilation, DNNL wants to know how many elements to insert between, not
                    // how far apart to space the elements like nGraph. So we have to subtract 1
                    // from each pos.
                    Strides window_dilation_strides_adjusted;

                    for (size_t s : convolution->get_window_dilation_strides_forward())
                    {
                        window_dilation_strides_adjusted.push_back(s - 1);
                    }

                    auto src_desc = dnnl_utils::get_input_dnnl_md(node, 0);
                    auto diff_dst_desc = dnnl_utils::get_input_dnnl_md(node, 1);
                    auto diff_weights_desc = dnnl_utils::get_output_dnnl_md(node, 0);
                    dnnl::algorithm convolution_algo = dnnl_utils::get_conv_algo();
                    if (has_bias<OP>())
                    {
                        auto diff_bias_desc = dnnl_utils::get_output_dnnl_md(node, 1);

                        return dnnl::convolution_backward_weights::desc(
                            convolution_algo,
                            src_desc,
                            diff_weights_desc,
                            diff_bias_desc,
                            diff_dst_desc,
                            DNNL_DIMS(convolution->get_window_movement_strides_forward()),
                            DNNL_DIMS(window_dilation_strides_adjusted),
                            DNNL_DIMS(convolution->get_padding_below_forward()),
                            DNNL_DIMS(convolution->get_padding_above_forward()) PADDING);
                    }
                    else
                    {
                        return dnnl::convolution_backward_weights::desc(
                            convolution_algo,
                            src_desc,
                            diff_weights_desc,
                            diff_dst_desc,
                            DNNL_DIMS(convolution->get_window_movement_strides_forward()),
                            DNNL_DIMS(window_dilation_strides_adjusted),
                            DNNL_DIMS(convolution->get_padding_below_forward()),
                            DNNL_DIMS(convolution->get_padding_above_forward()) PADDING);
                    }
                }

                template <typename OP>
                dnnl::convolution_forward::desc
                    get_convolution_forward_desc_for_backward_op(const ngraph::Node* node)
                {
                    auto convolution = static_cast<const OP*>(node);
                    // For dilation, DNNL wants to know how many elements to insert between, not
                    // how far apart to space the elements like nGraph. So we have to subtract 1
                    // from each pos.
                    Strides window_dilation_strides_adjusted;

                    for (size_t s : convolution->get_window_dilation_strides_forward())
                    {
                        window_dilation_strides_adjusted.push_back(s - 1);
                    }

                    dnnl::algorithm convolution_algo = dnnl_utils::get_conv_algo();
                    if (std::is_same<OP, ngraph::op::v0::ConvolutionBackpropData>())
                    {
                        auto weights_desc = dnnl_utils::get_input_dnnl_md(node, 0);
                        CHANGE_FORMAT

                        auto diff_dst_desc = dnnl_utils::get_input_dnnl_md(node, 1);
                        auto diff_src_desc = dnnl_utils::get_output_dnnl_md(node, 0);

                        return dnnl::convolution_forward::desc(
                            dnnl::prop_kind::forward,
                            convolution_algo,
                            diff_src_desc,
                            weights_desc,
                            diff_dst_desc,
                            DNNL_DIMS(convolution->get_window_movement_strides_forward()),
                            DNNL_DIMS(window_dilation_strides_adjusted),
                            DNNL_DIMS(convolution->get_padding_below_forward()),
                            DNNL_DIMS(convolution->get_padding_above_forward()) PADDING);
                    }
                    else if (std::is_same<OP, ngraph::op::v0::ConvolutionBackpropFilters>())
                    {
                        auto src_desc = dnnl_utils::get_input_dnnl_md(node, 0);
                        auto diff_dst_desc = dnnl_utils::get_input_dnnl_md(node, 1);
                        auto diff_weights_desc = dnnl_utils::get_output_dnnl_md(node, 0);
                        return dnnl::convolution_forward::desc(
                            dnnl::prop_kind::forward,
                            convolution_algo,
                            src_desc,
                            diff_weights_desc,
                            diff_dst_desc,
                            DNNL_DIMS(convolution->get_window_movement_strides_forward()),
                            DNNL_DIMS(window_dilation_strides_adjusted),
                            DNNL_DIMS(convolution->get_padding_below_forward()),
                            DNNL_DIMS(convolution->get_padding_above_forward()) PADDING);
                    }
                    else
                    {
                        auto src_desc = dnnl_utils::get_input_dnnl_md(node, 0);
                        auto diff_dst_desc = dnnl_utils::get_input_dnnl_md(node, 1);
                        auto diff_weights_desc = dnnl_utils::get_output_dnnl_md(node, 0);
                        auto diff_bias_desc = dnnl_utils::get_output_dnnl_md(node, 1);

                        return dnnl::convolution_forward::desc(
                            dnnl::prop_kind::forward,
                            convolution_algo,
                            src_desc,
                            diff_weights_desc,
                            diff_bias_desc,
                            diff_dst_desc,
                            DNNL_DIMS(convolution->get_window_movement_strides_forward()),
                            DNNL_DIMS(window_dilation_strides_adjusted),
                            DNNL_DIMS(convolution->get_padding_below_forward()),
                            DNNL_DIMS(convolution->get_padding_above_forward()) PADDING);
                    }
                }

                void build_quantize_reorder(std::vector<dnnl::memory*>& dnnl_memories,
                                            std::vector<dnnl::primitive*>& dnnl_primitives,
                                            std::vector<dnnl::memory::desc*>& dnnl_scratchpad_mds,
                                            const dnnl::memory::desc& input_desc,
                                            const dnnl::memory::desc& result_desc,
                                            const std::vector<float>& scales,
                                            const std::vector<size_t>& deps,
                                            size_t quantize_index,
                                            const int mask = 0);

                void build_convolution_backward_weights(
                    std::vector<dnnl::memory*>& dnnl_memories,
                    std::vector<dnnl::primitive*>& dnnl_primitives,
                    std::vector<dnnl::memory::desc*>& dnnl_scratchpad_mds,
                    const dnnl::convolution_backward_weights::desc& bwd_desc,
                    const dnnl::convolution_forward::desc& fwd_desc,
                    const std::vector<size_t>& deps,
                    size_t conv_index);

                void build_convolution_backward_data(
                    std::vector<dnnl::memory*>& dnnl_memories,
                    std::vector<dnnl::primitive*>& dnnl_primitives,
                    std::vector<dnnl::memory::desc*>& dnnl_scratchpad_mds,
                    const dnnl::convolution_backward_data::desc& bwd_desc,
                    const dnnl::convolution_forward::desc& fwd_desc,
                    const std::vector<size_t>& deps,
                    size_t conv_index);

                /**
                 * Convolution + bias backprop for weights and bias
                 */
                void build_convolution_backward_weights_bias(
                    std::vector<dnnl::memory*>& dnnl_memories,
                    std::vector<dnnl::primitive*>& dnnl_primitives,
                    std::vector<dnnl::memory::desc*>& dnnl_scratchpad_mds,
                    const dnnl::convolution_backward_weights::desc& bwd_desc,
                    const dnnl::convolution_forward::desc& fwd_desc,
                    const std::vector<size_t>& deps,
                    size_t conv_index);

                void build_deconvolutionbias_forward(
                    std::vector<dnnl::memory*>& dnnl_memories,
                    std::vector<dnnl::primitive*>& dnnl_primitives,
                    std::vector<dnnl::memory::desc*>& dnnl_scratchpad_mds,
                    const dnnl::deconvolution_forward::desc& fwd_desc,
                    const std::vector<size_t>& deps,
                    size_t conv_index,
                    const dnnl::memory::desc& weights_desc);

                void build_pooling_forward(std::vector<dnnl::memory*>& dnnl_memories,
                                           std::vector<dnnl::primitive*>& dnnl_primitives,
                                           std::vector<dnnl::memory::desc*>& dnnl_scratchpad_mds,
                                           const dnnl::pooling_forward::desc& pool_desc,
                                           const std::vector<size_t>& deps,
                                           size_t pool_index);

                void build_pooling_backward(std::vector<dnnl::memory*>& dnnl_memories,
                                            std::vector<dnnl::primitive*>& dnnl_primitives,
                                            std::vector<dnnl::memory::desc*>& dnnl_scratchpad_mds,
                                            const dnnl::pooling_backward::desc& pool_desc,
                                            const dnnl::pooling_forward::desc& pool_fwd_desc,
                                            const std::vector<size_t>& deps,
                                            size_t pool_index);

                void build_max_pooling_with_indices_forward(
                    std::vector<dnnl::memory*>& dnnl_memories,
                    std::vector<dnnl::primitive*>& dnnl_primitives,
                    std::vector<dnnl::memory::desc*>& dnnl_scratchpad_mds,
                    const dnnl::pooling_forward::desc& max_pool_desc,
                    const std::vector<size_t>& deps,
                    size_t max_pool_index);

                void build_max_pooling_backward(
                    std::vector<dnnl::memory*>& dnnl_memories,
                    std::vector<dnnl::primitive*>& dnnl_primitives,
                    std::vector<dnnl::memory::desc*>& dnnl_scratchpad_mds,
                    std::vector<char*>& dnnl_workspaces,
                    const dnnl::pooling_backward::desc& bwd_pool_desc,
                    const dnnl::pooling_forward::desc& fwd_pool_desc,
                    const dnnl::memory::desc& fprop_src_desc,
                    std::vector<size_t>& fdeps,
                    std::vector<size_t>& bdeps,
                    size_t fwd_pool_index,
                    size_t bwd_pool_index);

                void build_max_pooling_with_indices_backward(
                    std::vector<dnnl::memory*>& dnnl_memories,
                    std::vector<dnnl::primitive*>& dnnl_primitives,
                    std::vector<dnnl::memory::desc*>& dnnl_scratchpad_mds,
                    const dnnl::pooling_backward::desc& bwd_pool_desc,
                    const dnnl::pooling_forward::desc& fwd_pool_desc,
                    const std::vector<size_t>& deps,
                    size_t max_pool_index);

                void build_reorder(std::vector<dnnl::memory*>& dnnl_memories,
                                   std::vector<dnnl::primitive*>& dnnl_primitives,
                                   std::vector<dnnl::memory::desc*>& dnnl_scratchpad_mds,
                                   const dnnl::memory::desc& input_desc,
                                   const dnnl::memory::desc& result_desc,
                                   const std::vector<size_t>& deps,
                                   size_t reorder_index);

                void build_lrn_forward(std::vector<dnnl::memory*>& dnnl_memories,
                                       std::vector<dnnl::primitive*>& dnnl_primitives,
                                       std::vector<dnnl::memory::desc*>& dnnl_scratchpad_mds,
                                       const dnnl::lrn_forward::desc& lrn_desc,
                                       const std::vector<size_t>& deps,
                                       size_t lrn_index);

                void build_relu_forward(std::vector<dnnl::memory*>& dnnl_memories,
                                        std::vector<dnnl::primitive*>& dnnl_primitives,
                                        std::vector<dnnl::memory::desc*>& dnnl_scratchpad_mds,
                                        const dnnl::eltwise_forward::desc& relu_desc,
                                        const std::vector<size_t>& deps,
                                        size_t relu_index);

                void build_relu_backward(std::vector<dnnl::memory*>& dnnl_memories,
                                         std::vector<dnnl::primitive*>& dnnl_primitives,
                                         std::vector<dnnl::memory::desc*>& dnnl_scratchpad_mds,
                                         const dnnl::eltwise_backward::desc& bwd_desc,
                                         const dnnl::eltwise_forward::desc& fwd_desc,
                                         const std::vector<size_t>& deps,
                                         size_t relu_index);

                void build_sigmoid_forward(std::vector<dnnl::memory*>& dnnl_memories,
                                           std::vector<dnnl::primitive*>& dnnl_primitives,
                                           std::vector<dnnl::memory::desc*>& dnnl_scratchpad_mds,
                                           const dnnl::eltwise_forward::desc& sigmoid_desc,
                                           const std::vector<size_t>& deps,
                                           size_t sigmoid_index);

                void build_sigmoid_backward(std::vector<dnnl::memory*>& dnnl_memories,
                                            std::vector<dnnl::primitive*>& dnnl_primitives,
                                            std::vector<dnnl::memory::desc*>& dnnl_scratchpad_mds,
                                            const dnnl::eltwise_backward::desc& bwd_desc,
                                            const dnnl::eltwise_forward::desc& fwd_desc,
                                            const std::vector<size_t>& deps,
                                            size_t sigmoid_index);

                void build_elementwise_add(std::vector<dnnl::memory*>& dnnl_memories,
                                           std::vector<dnnl::primitive*>& dnnl_primitives,
                                           std::vector<dnnl::memory::desc*>& dnnl_scratchpad_mds,
                                           const dnnl::sum::primitive_desc& sum_pd,
                                           const std::vector<size_t>& deps,
                                           size_t add_index);

                void build_batchnorm_forward(
                    std::vector<dnnl::memory*>& dnnl_memories,
                    std::vector<dnnl::primitive*>& dnnl_primitives,
                    std::vector<dnnl::memory::desc*>& dnnl_scratchpad_mds,
                    const dnnl::batch_normalization_forward::desc& batchnorm_desc,
                    const dnnl::memory::desc& weights_desc,
                    bool bn_training_flag,
                    const std::vector<size_t>& deps,
                    size_t batchnorm_index,
                    const dnnl::post_ops& pops = dnnl::post_ops());

                void build_batchnorm_backward(
                    std::vector<dnnl::memory*>& dnnl_memories,
                    std::vector<dnnl::primitive*>& dnnl_primitives,
                    std::vector<dnnl::memory::desc*>& dnnl_scratchpad_mds,
                    const dnnl::batch_normalization_backward::desc& batchnorm_desc,
                    const dnnl::memory::desc& input_desc,
                    const dnnl::memory::desc& weights_desc,
                    const dnnl::memory::desc& dweights_desc,
                    float epsilon,
                    const std::vector<size_t>& deps,
                    size_t batchnorm_index);

                void build_concat(std::vector<dnnl::memory*>& dnnl_memories,
                                  std::vector<dnnl::primitive*>& dnnl_primitives,
                                  std::vector<dnnl::memory::desc*>& dnnl_scratchpad_mds,
                                  const dnnl::concat::primitive_desc& concat_pd,
                                  const std::vector<dnnl::memory::desc>& inputs_data_desc,
                                  const std::vector<size_t>& deps,
                                  size_t concat_index);

                void build_slice(std::vector<dnnl::memory*>& dnnl_memories,
                                 std::vector<dnnl::primitive*>& dnnl_primitives,
                                 std::vector<dnnl::memory::desc*>& dnnl_scratchpad_mds,
                                 dnnl::memory::desc input_desc,
                                 const dnnl::memory::desc& result_desc,
                                 const ngraph::Coordinate& lower_bounds,
                                 const ngraph::Shape& result_shape,
                                 const std::vector<size_t>& deps,
                                 size_t slice_index);

                void build_softmax_forward(std::vector<dnnl::memory*>& dnnl_memories,
                                           std::vector<dnnl::primitive*>& dnnl_primitives,
                                           std::vector<dnnl::memory::desc*>& dnnl_scratchpad_mds,
                                           const dnnl::softmax_forward::desc& sigmoid_desc,
                                           const std::vector<size_t>& deps,
                                           size_t softmax_index);

                void build_leaky_relu(std::vector<dnnl::memory*>& dnnl_memories,
                                      std::vector<dnnl::primitive*>& dnnl_primitives,
                                      std::vector<dnnl::memory::desc*>& dnnl_scratchpad_mds,
                                      const dnnl::eltwise_forward::desc& leaky_relu_desc,
                                      const std::vector<size_t>& deps,
                                      size_t leaky_relu_index);

                void build_bounded_relu(std::vector<dnnl::memory*>& dnnl_memories,
                                        std::vector<dnnl::primitive*>& dnnl_primitives,
                                        std::vector<dnnl::memory::desc*>& dnnl_scratchpad_mds,
                                        const dnnl::eltwise_forward::desc& bounded_relu_desc,
                                        const std::vector<size_t>& deps,
                                        size_t bounded_relu_index);

                void build_gelu(std::vector<dnnl::memory*>& dnnl_memories,
                                std::vector<dnnl::primitive*>& dnnl_primitives,
                                std::vector<dnnl::memory::desc*>& dnnl_scratchpad_mds,
                                const dnnl::eltwise_forward::desc& gelu_desc,
                                const std::vector<size_t>& deps,
                                size_t gelu_index);

                void build_gelu_backward(std::vector<dnnl::memory*>& dnnl_memories,
                                         std::vector<dnnl::primitive*>& dnnl_primitives,
                                         std::vector<dnnl::memory::desc*>& dnnl_scratchpad_mds,
                                         const dnnl::eltwise_backward::desc& bwd_desc,
                                         const dnnl::eltwise_forward::desc& fwd_desc,
                                         const std::vector<size_t>& deps,
                                         size_t gelu_index);

                // TODO(jmenon): Get rid of TensorWrappers at some point
                dnnl::memory::desc build_memory_descriptor(const TensorWrapper& tvw,
                                                           dnnl::memory::format_tag fmt_tag) const;
                dnnl::memory::desc build_memory_descriptor(const Shape& shape,
                                                           const ngraph::element::Type& et,
                                                           dnnl::memory::format_tag fmt_tag) const;
                size_t build_memory(const dnnl::memory::desc& desc);
                void build_memory(const dnnl::memory::desc& desc, size_t index);
                void build_memory(std::vector<dnnl::memory*>& dnnl_memories,
                                  const dnnl::memory::desc& desc,
                                  size_t index);

                template <typename OP>
                dnnl::concat::primitive_desc get_concat_desc(const ngraph::Node* node, size_t nargs)
                {
                    auto concat = static_cast<const OP*>(node);

                    std::vector<dnnl::memory::desc> inputs_desc;
                    for (size_t i = 0; i < nargs; i++)
                    {
                        inputs_desc.push_back(dnnl_utils::get_input_dnnl_md(node, i));
                    }

                    auto result_desc = dnnl_utils::get_output_dnnl_md(node, 0);

                    auto concat_dim = concat->get_concatenation_axis();

                    dnnl::primitive_attr attr;
                    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

                    // concat primitive descriptor
                    return dnnl::concat::primitive_desc(result_desc,
                                                        static_cast<int>(concat_dim),
                                                        inputs_desc,
                                                        runtime::cpu::executor::global_cpu_engine,
                                                        attr);
                }

                template <typename OP>
                dnnl::lstm_forward::desc
                    get_rnn_forward_desc(const ngraph::Node* node,
                                         const std::vector<TensorWrapper>& args,
                                         const std::vector<TensorWrapper>& out)
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

                    auto get_dnnl_rnn_direction = [&]() {
                        switch (direction)
                        {
                        case 1: return dnnl::rnn_direction::unidirectional_left2right;
                        case 2: return dnnl::rnn_direction::bidirectional_concat;
                        default: throw ngraph_error("unsupported dnnl rnn direction");
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
                    Shape src_iter_tz{num_fused_layers, direction, batch, feature_size};
                    Shape src_iter_c_tz{num_fused_layers, direction, batch, feature_size};
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
                    Shape dst_iter_tz{num_fused_layers, direction, batch, feature_size};
                    Shape dst_iter_c_tz{num_fused_layers, direction, batch, feature_size};

                    // We create the memory descriptors used by the user
                    auto src_layer_desc = build_memory_descriptor(
                        src_layer_tz, args[0].get_element_type(), dnnl::memory::FORMAT::tnc);
                    auto src_iter_desc = build_memory_descriptor(
                        src_iter_tz, args[1].get_element_type(), dnnl::memory::FORMAT::ldnc);
                    auto src_iter_c_desc = build_memory_descriptor(
                        src_iter_c_tz, args[2].get_element_type(), dnnl::memory::FORMAT::ldnc);
                    auto weights_layer_desc = build_memory_descriptor(
                        wei_layer_tz, args[3].get_element_type(), dnnl::memory::FORMAT::ldigo);
                    auto weights_iter_desc = build_memory_descriptor(
                        wei_iter_tz, args[4].get_element_type(), dnnl::memory::FORMAT::ldigo);
                    auto bias_desc = build_memory_descriptor(
                        bias_tz, args[5].get_element_type(), dnnl::memory::FORMAT::ldgo);
                    auto dst_layer_desc = build_memory_descriptor(
                        dst_layer_tz, out[0].get_element_type(), dnnl::memory::FORMAT::tnc);
                    auto dst_iter_desc = build_memory_descriptor(
                        dst_iter_tz, out[1].get_element_type(), dnnl::memory::FORMAT::ldnc);
                    auto dst_iter_c_desc = build_memory_descriptor(
                        dst_iter_c_tz, out[2].get_element_type(), dnnl::memory::FORMAT::ldnc);

                    return dnnl::lstm_forward::desc(dnnl::prop_kind::forward_training,
                                                    get_dnnl_rnn_direction(),
                                                    src_layer_desc,
                                                    src_iter_desc,
                                                    src_iter_c_desc,
                                                    weights_layer_desc,
                                                    weights_iter_desc,
                                                    bias_desc,
                                                    dst_layer_desc,
                                                    dst_iter_desc,
                                                    dst_iter_c_desc);
                }

                template <typename OP>
                dnnl::vanilla_rnn_forward::desc
                    get_vanilla_rnn_forward_desc(const ngraph::Node* node,
                                                 const std::vector<TensorWrapper>& args,
                                                 const std::vector<TensorWrapper>& out)
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

                    auto get_dnnl_rnn_direction = [&]() {
                        switch (direction)
                        {
                        case 1: return dnnl::rnn_direction::unidirectional_left2right;
                        case 2: return dnnl::rnn_direction::bidirectional_concat;
                        default: throw ngraph_error("unsupported dnnl rnn direction");
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
                    Shape src_iter_tz{num_fused_layers, direction, batch, feature_size};
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
                    Shape dst_iter_tz{num_fused_layers, direction, batch, feature_size};

                    // We create the memory descriptors used by the user
                    auto src_layer_desc = build_memory_descriptor(
                        src_layer_tz, args[0].get_element_type(), dnnl::memory::FORMAT::tnc);
                    auto src_iter_desc = build_memory_descriptor(
                        src_iter_tz, args[1].get_element_type(), dnnl::memory::FORMAT::ldnc);
                    auto weights_layer_desc = build_memory_descriptor(
                        wei_layer_tz, args[2].get_element_type(), dnnl::memory::FORMAT::ldigo);
                    auto weights_iter_desc = build_memory_descriptor(
                        wei_iter_tz, args[3].get_element_type(), dnnl::memory::FORMAT::ldigo);
                    auto bias_desc = build_memory_descriptor(
                        bias_tz, args[4].get_element_type(), dnnl::memory::FORMAT::ldgo);
                    auto dst_layer_desc = build_memory_descriptor(
                        dst_layer_tz, out[0].get_element_type(), dnnl::memory::FORMAT::tnc);
                    auto dst_iter_desc = build_memory_descriptor(
                        dst_iter_tz, out[1].get_element_type(), dnnl::memory::FORMAT::ldnc);

                    return dnnl::vanilla_rnn_forward::desc(dnnl::prop_kind::forward_training,
                                                           dnnl::algorithm::eltwise_tanh,
                                                           get_dnnl_rnn_direction(),
                                                           src_layer_desc,
                                                           src_iter_desc,
                                                           weights_layer_desc,
                                                           weights_iter_desc,
                                                           bias_desc,
                                                           dst_layer_desc,
                                                           dst_iter_desc);
                }
                void build_rnn_forward(std::vector<dnnl::memory*>& dnnl_memories,
                                       std::vector<dnnl::primitive*>& dnnl_primitives,
                                       std::vector<dnnl::memory::desc*>& dnnl_scratchpad_mds,
                                       std::vector<char*>& dnnl_workspaces,
                                       const dnnl::lstm_forward::desc& desc,
                                       std::vector<size_t>& deps,
                                       size_t rnn_idx);

                void
                    build_vanilla_rnn_forward(std::vector<dnnl::memory*>& dnnl_memories,
                                              std::vector<dnnl::primitive*>& dnnl_primitives,
                                              std::vector<dnnl::memory::desc*>& dnnl_scratchpad_mds,
                                              std::vector<char*>& dnnl_workspaces,
                                              const dnnl::vanilla_rnn_forward::desc& desc,
                                              std::vector<size_t>& deps,
                                              size_t rnn_idx);

                template <bool with_bias>
                void
                    build_convolution_forward(std::vector<dnnl::memory*>& dnnl_memories,
                                              std::vector<dnnl::primitive*>& dnnl_primitives,
                                              std::vector<dnnl::memory::desc*>& dnnl_scratchpad_mds,
                                              const dnnl::convolution_forward::desc& desc,
                                              const dnnl::primitive_attr& attr,
                                              const dnnl::engine& engine,
                                              const std::vector<size_t>& deps,
                                              size_t conv_idx)
                {
                    size_t input_idx, weights_idx, results_idx, bias_idx;
                    input_idx = deps[0];
                    weights_idx = deps[1];
                    dnnl_memories[input_idx] =
                        new dnnl::memory(desc.data.src_desc, engine, nullptr);
                    dnnl_memories[weights_idx] =
                        new dnnl::memory(desc.data.weights_desc, engine, nullptr);
                    if (with_bias)
                    {
                        bias_idx = deps[2];
                        results_idx = deps[3];
                        dnnl_memories[bias_idx] =
                            new dnnl::memory(desc.data.bias_desc, engine, nullptr);
                    }
                    else
                    {
                        results_idx = deps[2];
                    }
                    dnnl_memories[results_idx] =
                        new dnnl::memory(desc.data.dst_desc, engine, nullptr);

                    auto conv_pd = dnnl::convolution_forward::primitive_desc(desc, attr, engine);
                    dnnl_scratchpad_mds[conv_idx] =
                        new dnnl::memory::desc(conv_pd.scratchpad_desc());

                    dnnl::primitive* prim = new dnnl::convolution_forward(conv_pd);
                    dnnl_primitives[conv_idx] = prim;
                }

                template <bool with_bias>
                void build_inner_product_forward(
                    std::vector<dnnl::memory*>& dnnl_memories,
                    std::vector<dnnl::primitive*>& dnnl_primitives,
                    std::vector<dnnl::memory::desc*>& dnnl_scratchpad_mds,
                    const dnnl::inner_product_forward::desc& desc,
                    const dnnl::primitive_attr& attr,
                    const dnnl::engine& engine,
                    const std::vector<size_t>& deps,
                    size_t ip_idx)
                {
                    size_t input_idx, weights_idx, results_idx, bias_idx;
                    input_idx = deps[0];
                    weights_idx = deps[1];
                    dnnl_memories[input_idx] =
                        new dnnl::memory(desc.data.src_desc, engine, nullptr);
                    dnnl_memories[weights_idx] =
                        new dnnl::memory(desc.data.weights_desc, engine, nullptr);
                    if (with_bias)
                    {
                        bias_idx = deps[2];
                        results_idx = deps[3];
                        dnnl_memories[bias_idx] =
                            new dnnl::memory(desc.data.bias_desc, engine, nullptr);
                    }
                    else
                    {
                        results_idx = deps[2];
                    }
                    dnnl_memories[results_idx] =
                        new dnnl::memory(desc.data.dst_desc, engine, nullptr);

                    auto ip_pd = dnnl::inner_product_forward::primitive_desc(desc, attr, engine);
                    dnnl_scratchpad_mds[ip_idx] = new dnnl::memory::desc(ip_pd.scratchpad_desc());

                    dnnl::primitive* prim = new dnnl::inner_product_forward(ip_pd);
                    dnnl_primitives[ip_idx] = prim;
                }

                size_t query_scratchpad_sum(const dnnl::sum::primitive_desc);
                size_t query_scratchpad_concat(const dnnl::concat::primitive_desc);
                size_t query_scratchpad_pooling_forward(const dnnl::pooling_forward::desc& desc);
                size_t query_scratchpad_avg_pooling_backward(
                    const dnnl::pooling_forward::desc& fwd_desc,
                    const dnnl::pooling_backward::desc& bwd_desc);
                size_t query_scratchpad_max_pooling_backward(
                    const dnnl::pooling_forward::desc& fwd_desc,
                    const dnnl::pooling_backward::desc& bwd_desc);
                size_t query_scratchpad_max_pooling_with_indices_backward(
                    const dnnl::pooling_forward::desc& fwd_desc,
                    const dnnl::pooling_backward::desc& bwd_desc);
                size_t query_scratchpad_batchnorm_forward(
                    const dnnl::batch_normalization_forward::desc& desc,
                    const dnnl::post_ops& pops);
                size_t query_scratchpad_batchnorm_backward(
                    const dnnl::batch_normalization_backward::desc& desc,
                    const dnnl::memory::desc& input_desc,
                    float epsilon);
                size_t query_scratchpad_convolution_forward(
                    const dnnl::convolution_forward::desc& desc, dnnl::primitive_attr& attr);
                size_t query_scratchpad_convolution_backward_data(
                    const dnnl::convolution_forward::desc& fwd_desc,
                    const dnnl::convolution_backward_data::desc& bwd_desc);
                size_t query_scratchpad_convolution_backward_weights(
                    const dnnl::convolution_forward::desc& fwd_desc,
                    const dnnl::convolution_backward_weights::desc& bwd_desc);
                size_t query_scratchpad_deconvolution_forward(
                    const dnnl::deconvolution_forward::desc& desc);
                size_t query_scratchpad_eltwise_forward(const dnnl::eltwise_forward::desc& desc);
                size_t
                    query_scratchpad_eltwise_backward(const dnnl::eltwise_forward::desc& fwd_desc,
                                                      const dnnl::eltwise_backward::desc& bwd_desc);
                size_t query_scratchpad_ip_forward(const dnnl::inner_product_forward::desc& desc,
                                                   dnnl::primitive_attr& attr);
                size_t query_scratchpad_reorder(const dnnl::memory::desc& input_desc,
                                                const dnnl::memory::desc& result_desc);
                size_t query_scratchpad_lrn_forward(const dnnl::lrn_forward::desc& desc);
                size_t query_scratchpad_rnn_forward(const dnnl::lstm_forward::desc& desc);
                size_t query_scratchpad_vanilla_rnn_forward(
                    const dnnl::vanilla_rnn_forward::desc& desc);
                size_t query_scratchpad_slice(dnnl::memory::desc& input_desc,
                                              const dnnl::memory::desc& output_desc,
                                              const ngraph::Coordinate& lower_bounds,
                                              const ngraph::Shape& result_shape);
                size_t query_scratchpad_softmax_forward(const dnnl::softmax_forward::desc& desc);

            private:
                std::vector<dnnl::memory*> m_dnnl_memories;
                std::vector<dnnl::primitive*> m_dnnl_primitives;
                std::vector<dnnl::stream> m_dnnl_streams;
                std::unordered_map<size_t, std::vector<size_t>> m_primitive_deps;
                std::vector<std::unique_ptr<DNNLWorkspace>> m_workspaces;
                std::vector<char*> m_workspace_bufs;
                std::vector<dnnl::memory::desc*> m_dnnl_scratchpad_mds;
                size_t m_workspaces_size = 0;
                size_t m_dnnl_descriptors_size = 0;
                size_t m_max_scratchpad_size = 0;
            };
        }
    }
}
