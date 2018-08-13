/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
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

#include <algorithm>
#include <memory>
#include <string>
#include <typeindex>
#include <typeinfo>

#include <mkldnn.hpp>

#include "gpu_layout.hpp"
#include "ngraph/op/replace_slice.hpp"
#include "ngraph/runtime/gpu/gpu_op_annotations.hpp"

using namespace std;
using namespace ngraph;

namespace ngraph
{
    namespace runtime
    {
        namespace gpu
        {
            namespace pass
            {
                template <typename T, bool use_bias, bool default_weights_format>
                void ConvolutionLayout(std::shared_ptr<ngraph::Node> node,
                                       vector<memory::desc>& i_mds,
                                       vector<memory::desc>& o_mds)
                {
                    auto convolution = static_cast<const T*>(node.get());

                    auto arg0_shape = node->get_input_shape(0);
                    auto arg1_shape = node->get_input_shape(1);

                    if (default_weights_format)
                    {
                        arg1_shape = std::dynamic_pointer_cast<ngraph::op::GroupConvolution>(node)
                                         ->get_weights_dimensions();
                    }
                    auto result_shape = node->get_output_shape(0);
                    auto filter_strides = convolution->get_window_movement_strides();
                    auto padding_below = convolution->get_padding_below();
                    auto padding_above = convolution->get_padding_above();

                    Strides window_dilation_strides_adjusted;

                    for (size_t s : convolution->get_window_dilation_strides())
                    {
                        window_dilation_strides_adjusted.push_back(s - 1);
                    }

                    memory::data_type et =
                        mkldnn_utils::get_mkldnn_data_type(node->get_input_element_type(0));

                    engine gpu_engine(engine::gpu, 0);
                    memory::dims mkldnn_arg0_shape(arg0_shape.begin(), arg0_shape.end());
                    memory::dims mkldnn_arg1_shape(arg1_shape.begin(), arg1_shape.end());
                    memory::dims mkldnn_result_shape(result_shape.begin(), result_shape.end());
                    memory::dims mkldnn_filter_strides(filter_strides.begin(),
                                                       filter_strides.end());
                    memory::dims mkldnn_dilated_strides(window_dilation_strides_adjusted.begin(),
                                                        window_dilation_strides_adjusted.end());
                    memory::dims mkldnn_padding_below(padding_below.begin(), padding_below.end());
                    memory::dims mkldnn_padding_above(padding_above.begin(), padding_above.end());
                    const memory::desc input_data_desc(mkldnn_arg0_shape, et, memory::format::any);
                    const memory::desc weights_desc(mkldnn_arg1_shape, et, memory::format::any);
                    const memory::desc result_desc(mkldnn_result_shape, et, memory::format::any);
                    std::unique_ptr<convolution_forward::desc> fwd_desc{nullptr};
                    if (use_bias)
                    {
                        auto arg2_shape = node->get_input_shape(2);
                        ngraph::op::util::validate_convbias_shapes(
                            arg0_shape, arg1_shape, arg2_shape);
                        memory::dims mkldnn_arg2_shape(arg2_shape.begin(), arg2_shape.end());
                        const memory::desc bias_desc(mkldnn_arg2_shape, et, memory::format::any);
                        try
                        {
                            fwd_desc.reset(
                                new convolution_forward::desc(prop_kind::forward,
                                                              algorithm::convolution_direct,
                                                              input_data_desc,
                                                              weights_desc,
                                                              bias_desc, // with bias
                                                              result_desc,
                                                              mkldnn_filter_strides,
                                                              mkldnn_dilated_strides,
                                                              mkldnn_padding_below,
                                                              mkldnn_padding_above,
                                                              padding_kind::zero));
                        }
                        catch (const mkldnn::error& e)
                        {
                            throw ngraph_error(
                                "setting layouts on Convolution failed with MKLDNN error: " +
                                e.message);
                        }
                    }
                    else
                    {
                        try
                        {
                            fwd_desc.reset(
                                new convolution_forward::desc(prop_kind::forward,
                                                              algorithm::convolution_direct,
                                                              input_data_desc,
                                                              weights_desc,
                                                              result_desc,
                                                              mkldnn_filter_strides,
                                                              mkldnn_dilated_strides,
                                                              mkldnn_padding_below,
                                                              mkldnn_padding_above,
                                                              padding_kind::zero));
                        }
                        catch (const mkldnn::error& e)
                        {
                            throw ngraph_error(
                                "setting layouts on Convolution failed with MKLDNN error: " +
                                e.message);
                        }
                    }
                    convolution_forward::primitive_desc prim_desc(*fwd_desc, gpu_engine);
                    i_mds.push_back(prim_desc.src_primitive_desc().desc());

                    if (default_weights_format)
                    {
                        //note, we need the original shape (4D) while arg_shape1 is redefined
                        i_mds.push_back(mkldnn_utils::create_default_mkldnn_md(
                            node.get(), 1, false, memory::format::oihw));
                    }
                    else
                    {
                        i_mds.push_back(prim_desc.weights_primitive_desc().desc());
                    }

                    if (use_bias)
                    {
                        i_mds.push_back(prim_desc.bias_primitive_desc().desc());
                    }
                    o_mds.push_back(prim_desc.dst_primitive_desc().desc());
                }

                template <>
                void GPULayout::LAYOUT_DECL(ngraph::op::Convolution)
                {
                    if (mkldnn_utils::use_mkldnn_kernel(node.get()))
                    {
                        vector<memory::desc> i_mds;
                        vector<memory::desc> o_mds;
                        ConvolutionLayout<ngraph::op::Convolution, false, false>(
                            node, i_mds, o_mds);

                        node = insert_input_conversions(external_function, node, i_mds);
                        set_output_layouts(node, o_mds);
                    }
                    else
                    {
                        set_native_layouts(external_function, node);
                    }
                }

                template <>
                void GPULayout::LAYOUT_DECL(ngraph::op::GroupConvolution)
                {
                    if (mkldnn_utils::use_mkldnn_kernel(node.get()))
                    {
                        vector<memory::desc> i_mds;
                        vector<memory::desc> o_mds;
                        ConvolutionLayout<ngraph::op::GroupConvolution, false, true>(
                            node, i_mds, o_mds);

                        node = insert_input_conversions(external_function, node, i_mds);
                        set_output_layouts(node, o_mds);
                    }
                    else
                    {
                        set_native_layouts(external_function, node);
                    }
                }

                template <>
                void GPULayout::LAYOUT_DECL(ngraph::op::ConvolutionBias)
                {
                    if (mkldnn_utils::use_mkldnn_kernel(node.get()))
                    {
                        vector<memory::desc> i_mds;
                        vector<memory::desc> o_mds;
                        ConvolutionLayout<ngraph::op::ConvolutionBias, true, false>(
                            node, i_mds, o_mds);
                        node = insert_input_conversions(external_function, node, i_mds);
                        set_output_layouts(node, o_mds);
                    }
                    else
                    {
                        set_native_layouts(external_function, node);
                    }
                }

                template <>
                void GPULayout::LAYOUT_DECL(ngraph::op::ConvolutionRelu)
                {
                    if (mkldnn_utils::use_mkldnn_kernel(node.get()))
                    {
                        vector<memory::desc> i_mds;
                        vector<memory::desc> o_mds;
                        ConvolutionLayout<ngraph::op::ConvolutionRelu, false, false>(
                            node, i_mds, o_mds);
                        node = insert_input_conversions(external_function, node, i_mds);
                        set_output_layouts(node, o_mds);
                    }
                    else
                    {
                        set_native_layouts(external_function, node);
                    }
                }

                template <>
                void GPULayout::LAYOUT_DECL(ngraph::op::ConvolutionBiasAdd)
                {
                    if (mkldnn_utils::use_mkldnn_kernel(node.get()))
                    {
                        vector<memory::desc> i_mds;
                        vector<memory::desc> o_mds;
                        ConvolutionLayout<ngraph::op::ConvolutionBiasAdd, true, false>(
                            node, i_mds, o_mds);
                        // Force second input to sum to use the same layout as convolution output
                        i_mds.push_back(o_mds[0]);
                        node = insert_input_conversions(external_function, node, i_mds);
                        set_output_layouts(node, o_mds);
                    }
                    else
                    {
                        set_native_layouts(external_function, node);
                    }
                }

                template <>
                void GPULayout::LAYOUT_DECL(ngraph::op::ConvolutionBackpropData)
                {
                    if (mkldnn_utils::use_mkldnn_kernel(node.get()))
                    {
                        auto convolution =
                            static_cast<const ngraph::op::ConvolutionBackpropData*>(node.get());

                        auto arg0_shape = node->get_input_shape(0);
                        auto arg1_shape = node->get_input_shape(1);
                        auto result_shape = node->get_output_shape(0);
                        auto filter_strides = convolution->get_window_movement_strides_forward();
                        auto padding_below = convolution->get_padding_below_forward();
                        auto padding_above = convolution->get_padding_above_forward();

                        Strides window_dilation_strides_adjusted;

                        for (size_t s : convolution->get_window_dilation_strides_forward())
                        {
                            window_dilation_strides_adjusted.push_back(s - 1);
                        }

                        memory::data_type et =
                            mkldnn_utils::get_mkldnn_data_type(node->get_input_element_type(0));

                        engine gpu_engine(engine::gpu, 0);
                        memory::dims mkldnn_arg0_shape(arg0_shape.begin(), arg0_shape.end());
                        memory::dims mkldnn_arg1_shape(arg1_shape.begin(), arg1_shape.end());
                        memory::dims mkldnn_result_shape(result_shape.begin(), result_shape.end());
                        memory::dims mkldnn_filter_strides(filter_strides.begin(),
                                                           filter_strides.end());
                        memory::dims mkldnn_dilated_strides(
                            window_dilation_strides_adjusted.begin(),
                            window_dilation_strides_adjusted.end());
                        memory::dims mkldnn_padding_below(padding_below.begin(),
                                                          padding_below.end());
                        memory::dims mkldnn_padding_above(padding_above.begin(),
                                                          padding_above.end());

                        const memory::desc weights_desc(mkldnn_arg0_shape, et, memory::format::any);
                        const memory::desc delta_desc(mkldnn_arg1_shape, et, memory::format::any);
                        const memory::desc result_desc(
                            mkldnn_result_shape, et, memory::format::any);

                        convolution_backward_data::desc bwd_desc(algorithm::convolution_direct,
                                                                 result_desc,
                                                                 weights_desc,
                                                                 delta_desc,
                                                                 mkldnn_filter_strides,
                                                                 mkldnn_dilated_strides,
                                                                 mkldnn_padding_below,
                                                                 mkldnn_padding_above,
                                                                 padding_kind::zero);

                        convolution_forward::desc fwd_desc(prop_kind::forward,
                                                           algorithm::convolution_direct,
                                                           result_desc,
                                                           weights_desc,
                                                           delta_desc,
                                                           mkldnn_filter_strides,
                                                           mkldnn_dilated_strides,
                                                           mkldnn_padding_below,
                                                           mkldnn_padding_above,
                                                           padding_kind::zero);
                        convolution_forward::primitive_desc fwd_prim_desc(fwd_desc, gpu_engine);

                        convolution_backward_data::primitive_desc prim_desc(
                            bwd_desc, gpu_engine, fwd_prim_desc);

                        vector<memory::desc> i_mds;
                        vector<memory::desc> o_mds;
                        i_mds.push_back(prim_desc.weights_primitive_desc().desc());
                        i_mds.push_back(prim_desc.diff_dst_primitive_desc().desc());
                        o_mds.push_back(prim_desc.diff_src_primitive_desc().desc());

                        node = insert_input_conversions(external_function, node, i_mds);
                        set_output_layouts(node, o_mds);
                    }
                    else
                    {
                        set_native_layouts(external_function, node);
                    }
                }

                template <typename T, bool use_bias>
                void ConvolutionBackpropFiltersLayout(std::shared_ptr<ngraph::Node> node,
                                                      vector<memory::desc>& i_mds,
                                                      vector<memory::desc>& o_mds)
                {
                    auto convolution = static_cast<const T*>(node.get());

                    auto data_shape = node->get_input_shape(0);
                    auto delta_shape = node->get_input_shape(1);
                    auto filters_shape = node->get_output_shape(0);
                    auto filter_strides = convolution->get_window_movement_strides_forward();
                    auto padding_below = convolution->get_padding_below_forward();
                    auto padding_above = convolution->get_padding_above_forward();

                    Strides window_dilation_strides_adjusted;

                    for (size_t s : convolution->get_window_dilation_strides_forward())
                    {
                        window_dilation_strides_adjusted.push_back(s - 1);
                    }

                    memory::data_type et =
                        mkldnn_utils::get_mkldnn_data_type(node->get_input_element_type(0));

                    engine gpu_engine(engine::gpu, 0);
                    memory::dims mkldnn_data_shape(data_shape.begin(), data_shape.end());
                    memory::dims mkldnn_delta_shape(delta_shape.begin(), delta_shape.end());
                    memory::dims mkldnn_filters_shape(filters_shape.begin(), filters_shape.end());
                    memory::dims mkldnn_filter_strides(filter_strides.begin(),
                                                       filter_strides.end());
                    memory::dims mkldnn_dilated_strides(window_dilation_strides_adjusted.begin(),
                                                        window_dilation_strides_adjusted.end());
                    memory::dims mkldnn_padding_below(padding_below.begin(), padding_below.end());
                    memory::dims mkldnn_padding_above(padding_above.begin(), padding_above.end());

                    const memory::desc data_desc(mkldnn_data_shape, et, memory::format::any);
                    const memory::desc delta_desc(mkldnn_delta_shape, et, memory::format::any);
                    const memory::desc filters_desc(mkldnn_filters_shape, et, memory::format::any);

                    std::unique_ptr<convolution_backward_weights::desc> bwd_desc{nullptr};
                    std::unique_ptr<convolution_forward::desc> fwd_desc{nullptr};
                    if (use_bias)
                    {
                        auto bias_shape = node->get_output_shape(1);
                        ngraph::op::util::validate_convbias_shapes(
                            data_shape, filters_shape, bias_shape);
                        memory::dims mkldnn_bias_shape(bias_shape.begin(), bias_shape.end());
                        const memory::desc bias_desc(mkldnn_bias_shape, et, memory::format::any);
                        bwd_desc.reset(
                            new convolution_backward_weights::desc(algorithm::convolution_direct,
                                                                   data_desc,
                                                                   filters_desc,
                                                                   bias_desc,
                                                                   delta_desc,
                                                                   mkldnn_filter_strides,
                                                                   mkldnn_dilated_strides,
                                                                   mkldnn_padding_below,
                                                                   mkldnn_padding_above,
                                                                   padding_kind::zero));

                        fwd_desc.reset(new convolution_forward::desc(prop_kind::forward,
                                                                     algorithm::convolution_direct,
                                                                     data_desc,
                                                                     filters_desc,
                                                                     bias_desc,
                                                                     delta_desc,
                                                                     mkldnn_filter_strides,
                                                                     mkldnn_dilated_strides,
                                                                     mkldnn_padding_below,
                                                                     mkldnn_padding_above,
                                                                     padding_kind::zero));
                    }
                    else
                    {
                        bwd_desc.reset(
                            new convolution_backward_weights::desc(algorithm::convolution_direct,
                                                                   data_desc,
                                                                   filters_desc,
                                                                   delta_desc,
                                                                   mkldnn_filter_strides,
                                                                   mkldnn_dilated_strides,
                                                                   mkldnn_padding_below,
                                                                   mkldnn_padding_above,
                                                                   padding_kind::zero));

                        fwd_desc.reset(new convolution_forward::desc(prop_kind::forward,
                                                                     algorithm::convolution_direct,
                                                                     data_desc,
                                                                     filters_desc,
                                                                     delta_desc,
                                                                     mkldnn_filter_strides,
                                                                     mkldnn_dilated_strides,
                                                                     mkldnn_padding_below,
                                                                     mkldnn_padding_above,
                                                                     padding_kind::zero));
                    }
                    convolution_forward::primitive_desc fwd_prim_desc(*fwd_desc, gpu_engine);
                    convolution_backward_weights::primitive_desc prim_desc(
                        *bwd_desc, gpu_engine, fwd_prim_desc);

                    i_mds.push_back(prim_desc.src_primitive_desc().desc());
                    i_mds.push_back(prim_desc.diff_dst_primitive_desc().desc());
                    o_mds.push_back(prim_desc.diff_weights_primitive_desc().desc());
                    if (use_bias)
                    {
                        o_mds.push_back(prim_desc.diff_bias_primitive_desc().desc());
                    }
                }

                template <>
                void GPULayout::LAYOUT_DECL(ngraph::op::ConvolutionBackpropFilters)
                {
                    if (mkldnn_utils::use_mkldnn_kernel(node.get()))
                    {
                        vector<memory::desc> i_mds;
                        vector<memory::desc> o_mds;
                        ConvolutionBackpropFiltersLayout<ngraph::op::ConvolutionBackpropFilters,
                                                         false>(node, i_mds, o_mds);

                        node = insert_input_conversions(external_function, node, i_mds);
                        set_output_layouts(node, o_mds);
                    }
                    else
                    {
                        set_native_layouts(external_function, node);
                    }
                }

                template <>
                void GPULayout::LAYOUT_DECL(ngraph::op::ConvolutionBiasBackpropFiltersBias)
                {
                    if (mkldnn_utils::use_mkldnn_kernel(node.get()))
                    {
                        vector<memory::desc> i_mds;
                        vector<memory::desc> o_mds;
                        ConvolutionBackpropFiltersLayout<
                            ngraph::op::ConvolutionBiasBackpropFiltersBias,
                            true>(node, i_mds, o_mds);

                        node = insert_input_conversions(external_function, node, i_mds);
                        set_output_layouts(node, o_mds);
                    }
                    else
                    {
                        set_native_layouts(external_function, node);
                    }
                }

                template <>
                void GPULayout::LAYOUT_DECL(ngraph::op::AvgPool)
                {
                    if (mkldnn_utils::use_mkldnn_kernel(node.get()))
                    {
                        auto avg_pool = static_cast<const ngraph::op::AvgPool*>(node.get());

                        auto arg0_shape = node->get_input_shape(0);
                        auto result_shape = node->get_output_shape(0);
                        auto filter_shape = avg_pool->get_window_shape();
                        auto filter_strides = avg_pool->get_window_movement_strides();
                        auto padding_below = avg_pool->get_padding_below();
                        auto padding_above = avg_pool->get_padding_above();

                        memory::data_type et =
                            mkldnn_utils::get_mkldnn_data_type(node->get_input_element_type(0));

                        algorithm algorithm_enumerator =
                            avg_pool->get_include_padding_in_avg_computation()
                                ? algorithm::pooling_avg_include_padding
                                : algorithm::pooling_avg_exclude_padding;

                        memory::dims mkldnn_arg0_shape(arg0_shape.begin(), arg0_shape.end());
                        memory::dims mkldnn_result_shape(result_shape.begin(), result_shape.end());
                        memory::dims mkldnn_filter_shape(filter_shape.begin(), filter_shape.end());
                        memory::dims mkldnn_filter_strides(filter_strides.begin(),
                                                           filter_strides.end());
                        memory::dims mkldnn_padding_below(padding_below.begin(),
                                                          padding_below.end());
                        memory::dims mkldnn_padding_above(padding_above.begin(),
                                                          padding_above.end());

                        auto input_desc = mkldnn_utils::get_input_mkldnn_md(node.get(), 0);
                        auto result_desc =
                            memory::desc(mkldnn_result_shape, et, memory::format::any);

                        vector<memory::desc> i_mds;
                        vector<memory::desc> o_mds;
                        try
                        {
                            auto prim_desc =
                                pooling_forward::primitive_desc({prop_kind::forward_inference,
                                                                 algorithm_enumerator,
                                                                 input_desc,
                                                                 result_desc,
                                                                 mkldnn_filter_strides,
                                                                 mkldnn_filter_shape,
                                                                 mkldnn_padding_below,
                                                                 mkldnn_padding_above,
                                                                 padding_kind::zero},
                                                                mkldnn_utils::global_gpu_engine);
                            i_mds.push_back(input_desc);
                            o_mds.push_back(prim_desc.dst_primitive_desc().desc());
                        }
                        catch (const mkldnn::error& e)
                        {
                            throw ngraph_error("MKLDNN Unsupported pooling layout" +
                                               to_string(input_desc.data.format) + e.message);
                        }

                        node = insert_input_conversions(external_function, node, i_mds);
                        set_output_layouts(node, o_mds);
                    }
                    else
                    {
                        set_native_layouts(external_function, node);
                    }
                }

                template <>
                void GPULayout::LAYOUT_DECL(ngraph::op::AvgPoolBackprop)
                {
                    if (mkldnn_utils::use_mkldnn_kernel(node.get()))
                    {
                        auto avg_pool = static_cast<const ngraph::op::AvgPoolBackprop*>(node.get());

                        auto arg0_shape = node->get_input_shape(0);
                        auto result_shape = node->get_output_shape(0);
                        auto filter_shape = avg_pool->get_window_shape();
                        auto filter_strides = avg_pool->get_window_movement_strides();
                        auto padding_below = avg_pool->get_padding_below();
                        auto padding_above = avg_pool->get_padding_above();

                        memory::data_type et =
                            mkldnn_utils::get_mkldnn_data_type(node->get_input_element_type(0));

                        algorithm algorithm_enumerator =
                            avg_pool->get_include_padding_in_avg_computation()
                                ? algorithm::pooling_avg_include_padding
                                : algorithm::pooling_avg_exclude_padding;

                        memory::dims mkldnn_arg0_shape(arg0_shape.begin(), arg0_shape.end());
                        memory::dims mkldnn_result_shape(result_shape.begin(), result_shape.end());
                        memory::dims mkldnn_filter_shape(filter_shape.begin(), filter_shape.end());
                        memory::dims mkldnn_filter_strides(filter_strides.begin(),
                                                           filter_strides.end());
                        memory::dims mkldnn_padding_below(padding_below.begin(),
                                                          padding_below.end());
                        memory::dims mkldnn_padding_above(padding_above.begin(),
                                                          padding_above.end());

                        auto input_desc = mkldnn_utils::get_input_mkldnn_md(node.get(), 0);
                        auto result_desc =
                            memory::desc(mkldnn_result_shape, et, memory::format::any);

                        vector<memory::desc> i_mds;
                        vector<memory::desc> o_mds;
                        try
                        {
                            auto fwd_prim_desc =
                                pooling_forward::primitive_desc({prop_kind::forward_inference,
                                                                 algorithm_enumerator,
                                                                 result_desc,
                                                                 input_desc,
                                                                 mkldnn_filter_strides,
                                                                 mkldnn_filter_shape,
                                                                 mkldnn_padding_below,
                                                                 mkldnn_padding_above,
                                                                 padding_kind::zero},
                                                                mkldnn_utils::global_gpu_engine);
                            auto prim_desc =
                                pooling_backward::primitive_desc({algorithm_enumerator,
                                                                  result_desc,
                                                                  input_desc,
                                                                  mkldnn_filter_strides,
                                                                  mkldnn_filter_shape,
                                                                  mkldnn_padding_below,
                                                                  mkldnn_padding_above,
                                                                  padding_kind::zero},
                                                                 mkldnn_utils::global_gpu_engine,
                                                                 fwd_prim_desc);
                            i_mds.push_back(input_desc);
                            o_mds.push_back(prim_desc.diff_src_primitive_desc().desc());
                        }
                        catch (const mkldnn::error& e)
                        {
                            throw ngraph_error("MKLDNN Unsupported pooling layout" +
                                               to_string(input_desc.data.format) + e.message);
                        }

                        node = insert_input_conversions(external_function, node, i_mds);
                        set_output_layouts(node, o_mds);
                    }
                    else
                    {
                        set_native_layouts(external_function, node);
                    }
                }

                template <typename T, prop_kind pk>
                void MaxPoolLayout(std::shared_ptr<ngraph::Node> node,
                                   vector<memory::desc>& i_mds,
                                   vector<memory::desc>& o_mds)
                {
                    auto max_pool = static_cast<const T*>(node.get());

                    auto arg0_shape = node->get_input_shape(0);
                    auto result_shape = node->get_output_shape(0);
                    auto filter_shape = max_pool->get_window_shape();
                    auto filter_strides = max_pool->get_window_movement_strides();
                    auto padding_below = max_pool->get_padding_below();
                    auto padding_above = max_pool->get_padding_above();

                    memory::data_type et =
                        mkldnn_utils::get_mkldnn_data_type(node->get_input_element_type(0));

                    algorithm algorithm_enumerator = algorithm::pooling_max;

                    memory::dims mkldnn_arg0_shape(arg0_shape.begin(), arg0_shape.end());
                    memory::dims mkldnn_result_shape(result_shape.begin(), result_shape.end());
                    memory::dims mkldnn_filter_shape(filter_shape.begin(), filter_shape.end());
                    memory::dims mkldnn_filter_strides(filter_strides.begin(),
                                                       filter_strides.end());
                    memory::dims mkldnn_padding_below(padding_below.begin(), padding_below.end());
                    memory::dims mkldnn_padding_above(padding_above.begin(), padding_above.end());

                    auto input_desc = mkldnn_utils::get_input_mkldnn_md(node.get(), 0);
                    auto result_desc = memory::desc(mkldnn_result_shape, et, memory::format::any);

                    try
                    {
                        auto prim_desc =
                            pooling_forward::primitive_desc({pk,
                                                             algorithm_enumerator,
                                                             input_desc,
                                                             result_desc,
                                                             mkldnn_filter_strides,
                                                             mkldnn_filter_shape,
                                                             mkldnn_padding_below,
                                                             mkldnn_padding_above,
                                                             padding_kind::zero},
                                                            mkldnn_utils::global_gpu_engine);
                        i_mds.push_back(input_desc);
                        o_mds.push_back(prim_desc.dst_primitive_desc().desc());

                        if (pk == prop_kind::forward_training)
                        {
                            o_mds.push_back(prim_desc.workspace_primitive_desc().desc());
                        }
                    }
                    catch (const mkldnn::error& e)
                    {
                        throw ngraph_error("MKLDNN Unsupported pooling fwd layout" +
                                           to_string(input_desc.data.format) + e.message);
                    }
                }

                template <>
                void GPULayout::LAYOUT_DECL(ngraph::op::MaxPoolWithIndices)
                {
                    if (mkldnn_utils::use_mkldnn_kernel(node.get()))
                    {
                        vector<memory::desc> i_mds;
                        vector<memory::desc> o_mds;
                        MaxPoolLayout<ngraph::op::MaxPoolWithIndices, prop_kind::forward_training>(
                            node, i_mds, o_mds);

                        node = insert_input_conversions(external_function, node, i_mds);
                        set_output_layouts(node, o_mds);
                    }
                    else
                    {
                        set_native_layouts(external_function, node);
                    }
                }

                template <>
                void GPULayout::LAYOUT_DECL(ngraph::op::MaxPool)
                {
                    if (mkldnn_utils::use_mkldnn_kernel(node.get()))
                    {
                        vector<memory::desc> i_mds;
                        vector<memory::desc> o_mds;
                        MaxPoolLayout<ngraph::op::MaxPool, prop_kind::forward_inference>(
                            node, i_mds, o_mds);

                        node = insert_input_conversions(external_function, node, i_mds);
                        set_output_layouts(node, o_mds);
                    }
                    else
                    {
                        set_native_layouts(external_function, node);
                    }
                }

                template <typename T, bool with_indices>
                void MaxPoolBackpropLayout(std::shared_ptr<ngraph::Node> node,
                                           vector<memory::desc>& i_mds,
                                           vector<memory::desc>& o_mds)
                {
                    auto max_pool = static_cast<const T*>(node.get());

                    // arg 0 - work input
                    // arg 1 - delta
                    // arg 2 - work space
                    // Propagate fprop's input layout
                    auto arg0_shape = node->get_input_shape(0);
                    auto arg1_shape = node->get_input_shape(1);
                    auto result_shape = node->get_output_shape(0);
                    auto filter_shape = max_pool->get_window_shape();
                    auto filter_strides = max_pool->get_window_movement_strides();
                    auto padding_below = max_pool->get_padding_below();
                    auto padding_above = max_pool->get_padding_above();

                    memory::data_type et =
                        mkldnn_utils::get_mkldnn_data_type(node->get_input_element_type(1));

                    algorithm algorithm_enumerator = algorithm::pooling_max;

                    memory::dims mkldnn_arg0_shape(arg0_shape.begin(), arg0_shape.end());
                    memory::dims mkldnn_arg1_shape(arg1_shape.begin(), arg1_shape.end());
                    memory::dims mkldnn_result_shape(result_shape.begin(), result_shape.end());
                    memory::dims mkldnn_filter_shape(filter_shape.begin(), filter_shape.end());
                    memory::dims mkldnn_filter_strides(filter_strides.begin(),
                                                       filter_strides.end());
                    memory::dims mkldnn_padding_below(padding_below.begin(), padding_below.end());
                    memory::dims mkldnn_padding_above(padding_above.begin(), padding_above.end());

                    auto fprop_input_md = mkldnn_utils::get_input_mkldnn_md(node.get(), 0);
                    auto fprop_input_layout =
                        static_cast<memory::format>(fprop_input_md.data.format);
                    auto diff_dst_desc = memory::desc(mkldnn_arg1_shape, et, fprop_input_layout);
                    auto diff_src_desc = memory::desc(mkldnn_arg0_shape, et, memory::format::any);

                    try
                    {
                        auto fwd_prim_desc =
                            pooling_forward::primitive_desc({prop_kind::forward_training,
                                                             algorithm_enumerator,
                                                             diff_src_desc,
                                                             diff_dst_desc,
                                                             mkldnn_filter_strides,
                                                             mkldnn_filter_shape,
                                                             mkldnn_padding_below,
                                                             mkldnn_padding_above,
                                                             padding_kind::zero},
                                                            mkldnn_utils::global_gpu_engine);

                        auto prim_desc =
                            pooling_backward::primitive_desc({algorithm_enumerator,
                                                              diff_src_desc,
                                                              diff_dst_desc,
                                                              mkldnn_filter_strides,
                                                              mkldnn_filter_shape,
                                                              mkldnn_padding_below,
                                                              mkldnn_padding_above,
                                                              padding_kind::zero},
                                                             mkldnn_utils::global_gpu_engine,
                                                             fwd_prim_desc);
                        i_mds.push_back(fprop_input_md);
                        i_mds.push_back(diff_dst_desc);

                        if (with_indices)
                        {
                            i_mds.push_back(fwd_prim_desc.workspace_primitive_desc().desc());
                        }

                        o_mds.push_back(prim_desc.diff_src_primitive_desc().desc());
                    }
                    catch (const mkldnn::error& e)
                    {
                        throw ngraph_error("MKLDNN Unsupported pooling layout" +
                                           to_string(fprop_input_md.data.format) + e.message);
                    }
                }

                template <>
                void GPULayout::LAYOUT_DECL(ngraph::op::MaxPoolBackprop)
                {
                    if (mkldnn_utils::use_mkldnn_kernel(node.get()))
                    {
                        vector<memory::desc> i_mds;
                        vector<memory::desc> o_mds;
                        MaxPoolBackpropLayout<ngraph::op::MaxPoolBackprop, false>(
                            node, i_mds, o_mds);

                        node = insert_input_conversions(external_function, node, i_mds);
                        set_output_layouts(node, o_mds);
                    }
                    else
                    {
                        set_native_layouts(external_function, node);
                    }
                }

                template <>
                void GPULayout::LAYOUT_DECL(ngraph::op::MaxPoolWithIndicesBackprop)
                {
                    if (mkldnn_utils::use_mkldnn_kernel(node.get()))
                    {
                        vector<memory::desc> i_mds;
                        vector<memory::desc> o_mds;
                        MaxPoolBackpropLayout<ngraph::op::MaxPoolWithIndicesBackprop, true>(
                            node, i_mds, o_mds);

                        node = insert_input_conversions(external_function, node, i_mds);
                        set_output_layouts(node, o_mds);
                    }
                    else
                    {
                        set_native_layouts(external_function, node);
                    }
                }

                template <>
                void GPULayout::LAYOUT_DECL(ngraph::op::Result)
                {
                    auto result = static_cast<const ngraph::op::Result*>(node.get());
                    if (result->needs_default_layout() ||
                        mkldnn_utils::get_input_mkldnn_md(node.get(), 0).data.format ==
                            mkldnn_format_undef)
                    {
                        set_native_layouts(external_function, node, false);
                    }
                    else
                    {
                        auto input_md = mkldnn_utils::get_input_mkldnn_md(node.get(), 0);
                        vector<memory::desc> o_mds;
                        o_mds.push_back(input_md);
                        set_output_layouts(node, o_mds);
                    }
                }

                template <>
                void GPULayout::LAYOUT_DECL(ngraph::op::Reshape)
                {
                    auto reshape = static_cast<ngraph::op::Reshape*>(node.get());
                    if (reshape->get_is_transpose())
                    {
                        auto axis_order = reshape->get_input_order();
                        auto tvl = node->get_inputs()[0]
                                       .get_output()
                                       .get_tensor_view()
                                       ->get_tensor_view_layout();
                        auto gpu_tvl = dynamic_cast<runtime::gpu::LayoutDescriptor*>(tvl.get());
                        if (gpu_tvl && gpu_tvl->is_mkldnn_layout())
                        {
                            // Rotate MKLDNN memory descriptor
                            auto input_md = mkldnn_utils::get_input_mkldnn_md(node.get(), 0);
                            auto output_md = mkldnn_utils::rotate_blocked_md(input_md, axis_order);
                            set_output_layouts(node, {output_md});
                            auto op_annotations = reshape->get_op_annotations();
                            if (op_annotations)
                            {
                                // pass-through
                                op_annotations->add_in_place_oi_pair({0, 0, false});
                            }
                            else
                            {
                                op_annotations =
                                    std::make_shared<ngraph::runtime::gpu::GPUOpAnnotations>();
                                // pass-through
                                op_annotations->add_in_place_oi_pair({0, 0, false});
                                reshape->set_op_annotations(op_annotations);
                            }
                        }
                        else
                        {
                            auto input_strides = gpu_tvl->get_strides();
                            Strides output_strides(input_strides.size());
                            for (size_t i = 0; i < input_strides.size(); i++)
                            {
                                output_strides[i] = input_strides[axis_order[i]];
                            }
                            set_native_layouts(external_function, node);
                            auto output_tvl = dynamic_pointer_cast<runtime::gpu::LayoutDescriptor>(
                                node->get_output_tensor_view()->get_tensor_view_layout());
                            // TODO (jbobba): For now non-MKLDNN layouts are always in row-major format
                            // Enable this once we support non row-major strided formats
                            // output_tvl->set_strides(output_strides);
                        }
                    }
                    else
                    {
                        // Shape change only, tensor in native layout can be
                        // forwarded to output
                        auto op_annotations = reshape->get_op_annotations();
                        if (op_annotations)
                        {
                            // pass-through
                            op_annotations->add_in_place_oi_pair({0, 0, false});
                        }
                        else
                        {
                            op_annotations =
                                std::make_shared<ngraph::runtime::gpu::GPUOpAnnotations>();
                            // pass-through
                            op_annotations->add_in_place_oi_pair({0, 0, false});
                            reshape->set_op_annotations(op_annotations);
                        }
                        set_native_layouts(external_function, node);
                    }
                }

                template <>
                void GPULayout::LAYOUT_DECL(ngraph::op::GetOutputElement)
                {
                    auto goe = static_cast<const ngraph::op::GetOutputElement*>(node.get());

                    if (mkldnn_utils::get_input_mkldnn_md(node.get(), 0).data.format ==
                        mkldnn_format_undef)
                    {
                        set_native_layouts(external_function, node);
                    }
                    else
                    {
                        auto input_md = mkldnn_utils::get_input_mkldnn_md(node.get(), goe->get_n());
                        vector<memory::desc> o_mds;
                        o_mds.push_back(input_md);
                        set_output_layouts(node, o_mds);
                    }
                }

                template <>
                void GPULayout::LAYOUT_DECL(ngraph::op::Relu)
                {
                    if (mkldnn_utils::use_mkldnn_kernel(node.get()))
                    {
                        auto input_md = mkldnn_utils::get_input_mkldnn_md(node.get(), 0);
                        vector<memory::desc> o_mds;
                        o_mds.push_back(input_md);
                        set_output_layouts(node, o_mds);
                    }
                    else
                    {
                        set_native_layouts(external_function, node);
                    }
                }

                template <>
                void GPULayout::LAYOUT_DECL(ngraph::op::LRN)
                {
                    if (runtime::gpu::mkldnn_utils::use_mkldnn_kernel(node.get()))
                    {
                        auto input_md = mkldnn_utils::get_input_mkldnn_md(node.get(), 0);
                        vector<memory::desc> o_mds;
                        o_mds.push_back(input_md);
                        set_output_layouts(node, o_mds);
                    }
                    else
                    {
                        set_native_layouts(external_function, node);
                    }
                }

                template <>
                void GPULayout::LAYOUT_DECL(ngraph::op::Sigmoid)
                {
                    if (mkldnn_utils::use_mkldnn_kernel(node.get()))
                    {
                        auto input_md = mkldnn_utils::get_input_mkldnn_md(node.get(), 0);
                        vector<memory::desc> o_mds;
                        o_mds.push_back(input_md);
                        set_output_layouts(node, o_mds);
                    }
                    else
                    {
                        set_native_layouts(external_function, node);
                    }
                }

                template <>
                void GPULayout::LAYOUT_DECL(ngraph::op::SigmoidBackprop)
                {
                    if (mkldnn_utils::use_mkldnn_kernel(node.get()))
                    {
                        auto input_md = mkldnn_utils::get_input_mkldnn_md(node.get(), 0);
                        vector<memory::desc> i_mds;
                        vector<memory::desc> o_mds;
                        //ensure delta and input have same layout
                        i_mds.push_back(input_md);
                        i_mds.push_back(input_md);
                        o_mds.push_back(input_md);
                        node = insert_input_conversions(external_function, node, i_mds);
                        set_output_layouts(node, o_mds);
                    }
                    else
                    {
                        set_native_layouts(external_function, node);
                    }
                }

                template <>
                void GPULayout::LAYOUT_DECL(ngraph::op::ReluBackprop)
                {
                    if (mkldnn_utils::use_mkldnn_kernel(node.get()))
                    {
                        auto kernel_md = mkldnn_utils::get_input_mkldnn_md(node.get(), 0);
                        auto kernel_layout = static_cast<memory::format>(kernel_md.data.format);
                        if (!mkldnn_utils::is_mkldnn_blocked_data_format(kernel_layout))
                        {
                            // Propagate delta layout
                            kernel_md = mkldnn_utils::get_input_mkldnn_md(node.get(), 1);
                        }

                        vector<memory::desc> i_mds;
                        vector<memory::desc> o_mds;
                        i_mds.push_back(kernel_md);
                        i_mds.push_back(kernel_md);
                        o_mds.push_back(kernel_md);
                        node = insert_input_conversions(external_function, node, i_mds);
                        set_output_layouts(node, o_mds);
                    }
                    else
                    {
                        set_native_layouts(external_function, node);
                    }
                }

                template <typename T>
                void BatchNormLayout(std::shared_ptr<ngraph::Node> node,
                                     vector<memory::desc>& i_mds,
                                     vector<memory::desc>& o_mds)
                {
                    auto bn = static_cast<const ngraph::op::BatchNorm*>(node.get());

                    auto input_md = mkldnn_utils::get_input_mkldnn_md(node.get(), 2);
                    auto arg0_md = mkldnn_utils::create_default_mkldnn_md(
                        node.get(), 0, false, memory::format::x);
                    auto arg1_md = mkldnn_utils::create_default_mkldnn_md(
                        node.get(), 1, false, memory::format::x);

                    if (bn->get_training_flag() && node->get_input_size() == 3)
                    {
                        auto out1_md = mkldnn_utils::create_default_mkldnn_md(
                            node.get(), 1, true, memory::format::x);
                        auto out2_md = mkldnn_utils::create_default_mkldnn_md(
                            node.get(), 2, true, memory::format::x);

                        i_mds.push_back(arg0_md);
                        i_mds.push_back(arg1_md);
                        i_mds.push_back(input_md);
                        o_mds.push_back(input_md);
                        o_mds.push_back(out1_md);
                        o_mds.push_back(out2_md);
                    }
                    else
                    {
                        auto arg3_md = mkldnn_utils::create_default_mkldnn_md(
                            node.get(), 3, false, memory::format::x);
                        auto arg4_md = mkldnn_utils::create_default_mkldnn_md(
                            node.get(), 4, false, memory::format::x);

                        i_mds.push_back(arg0_md);
                        i_mds.push_back(arg1_md);
                        i_mds.push_back(input_md);
                        i_mds.push_back(arg3_md);
                        i_mds.push_back(arg4_md);
                        o_mds.push_back(input_md);
                    }
                }

                template <>
                void GPULayout::LAYOUT_DECL(ngraph::op::BatchNorm)
                {
                    if (mkldnn_utils::use_mkldnn_kernel(node.get()))
                    {
                        vector<memory::desc> i_mds;
                        vector<memory::desc> o_mds;
                        BatchNormLayout<ngraph::op::BatchNorm>(node, i_mds, o_mds);
                        node = insert_input_conversions(external_function, node, i_mds);
                        set_output_layouts(node, o_mds);
                    }
                    else
                    {
                        set_native_layouts(external_function, node);
                    }
                }

                template <>
                void GPULayout::LAYOUT_DECL(ngraph::op::BatchNormRelu)
                {
                    if (mkldnn_utils::use_mkldnn_kernel(node.get()))
                    {
                        vector<memory::desc> i_mds;
                        vector<memory::desc> o_mds;
                        BatchNormLayout<ngraph::op::BatchNormRelu>(node, i_mds, o_mds);
                        node = insert_input_conversions(external_function, node, i_mds);
                        set_output_layouts(node, o_mds);
                    }
                    else
                    {
                        throw ngraph_error("BatchnormRelu only supported in MKLDNN for now");
                    }
                }

                template <>
                void GPULayout::LAYOUT_DECL(ngraph::op::BatchNormBackprop)
                {
                    if (mkldnn_utils::use_mkldnn_kernel(node.get()))
                    {
                        auto kernel_md = mkldnn_utils::get_input_mkldnn_md(node.get(), 2);
                        auto kernel_layout = static_cast<memory::format>(kernel_md.data.format);
                        auto arg0_md = mkldnn_utils::create_default_mkldnn_md(
                            node.get(), 0, false, memory::format::x);
                        auto arg1_md = mkldnn_utils::create_default_mkldnn_md(
                            node.get(), 1, false, memory::format::x);
                        auto arg3_md = mkldnn_utils::create_default_mkldnn_md(
                            node.get(), 3, false, memory::format::x);
                        auto arg4_md = mkldnn_utils::create_default_mkldnn_md(
                            node.get(), 4, false, memory::format::x);
                        auto out1_md = mkldnn_utils::create_default_mkldnn_md(
                            node.get(), 1, true, memory::format::x);
                        auto out2_md = mkldnn_utils::create_default_mkldnn_md(
                            node.get(), 2, true, memory::format::x);

                        if (!mkldnn_utils::is_mkldnn_blocked_data_format(kernel_layout))
                        {
                            // Propagate delta layout
                            kernel_md = mkldnn_utils::get_input_mkldnn_md(node.get(), 5);
                        }

                        vector<memory::desc> i_mds;
                        vector<memory::desc> o_mds;

                        i_mds.push_back(arg0_md);
                        i_mds.push_back(arg1_md);
                        i_mds.push_back(kernel_md);
                        i_mds.push_back(arg3_md);
                        i_mds.push_back(arg4_md);
                        i_mds.push_back(kernel_md);

                        o_mds.push_back(kernel_md);
                        o_mds.push_back(out1_md);
                        o_mds.push_back(out2_md);

                        node = insert_input_conversions(external_function, node, i_mds);
                        set_output_layouts(node, o_mds);
                    }
                    else
                    {
                        throw ngraph_error("Batchnorm Backprop only supported in MKLDNN for now");
                    }
                }

                template <>
                void GPULayout::LAYOUT_DECL(ngraph::op::Add)
                {
                    if (mkldnn_utils::use_mkldnn_kernel(node.get()))
                    {
                        auto input0_md = mkldnn_utils::get_input_mkldnn_md(node.get(), 0);

                        vector<memory::desc> i_mds;
                        vector<memory::desc> o_mds;
                        i_mds.push_back(input0_md);
                        i_mds.push_back(input0_md);
                        o_mds.push_back(input0_md);
                        node = insert_input_conversions(external_function, node, i_mds);
                        set_output_layouts(node, o_mds);
                    }
                    else
                    {
                        set_native_layouts(external_function, node);
                    }
                }

                template <>
                void GPULayout::LAYOUT_DECL(ngraph::op::Concat)
                {
                    if (mkldnn_utils::use_mkldnn_kernel(node.get()))
                    {
                        auto concat = static_cast<const ngraph::op::Concat*>(node.get());

                        size_t concat_dim = concat->get_concatenation_axis();
                        auto result_desc = mkldnn_utils::create_default_mkldnn_md(
                            node.get(), 0, true, memory::format::any);

                        std::vector<mkldnn::memory::primitive_desc> inputs_pd;

                        for (size_t i = 0; i < node->get_input_size(); i++)
                        {
                            auto input_md = mkldnn_utils::get_input_mkldnn_md(node.get(), i);
                            inputs_pd.push_back(mkldnn::memory::primitive_desc(
                                input_md, mkldnn_utils::global_gpu_engine));
                        }
                        try
                        {
                            auto prim_desc = concat::primitive_desc(
                                result_desc, static_cast<int>(concat_dim), inputs_pd);

                            vector<memory::desc> i_mds;
                            vector<memory::desc> o_mds;
                            for (size_t i = 0; i < node->get_input_size(); i++)
                            {
                                i_mds.push_back(inputs_pd[i].desc());
                            }
                            o_mds.push_back(prim_desc.dst_primitive_desc().desc());
                            node = insert_input_conversions(external_function, node, i_mds);
                            set_output_layouts(node, o_mds);
                        }
                        catch (const mkldnn::error& e)
                        {
                            throw ngraph_error(e.message);
                        }
                    }
                    else
                    {
                        set_native_layouts(external_function, node);
                    }
                }

                template <>
                void GPULayout::LAYOUT_DECL(ngraph::op::Lstm)
                {
                    if (mkldnn_utils::use_mkldnn_kernel(node.get()))
                    {
                        // TODO: for now, framework formats for src_layer, src_iter, weights_layer and weights_iter
                        // matches to the expected mkldnn format. we need to handle a case to insert convert Op's
                        // if the format doesn't matches.
                        set_native_layouts(external_function, node, false);
                    }
                    else
                    {
                        throw ngraph_error("LSTM fused op is only supported in MKLDNN for now.");
                    }
                }

                template <>
                void GPULayout::LAYOUT_DECL(ngraph::op::Rnn)
                {
                    if (mkldnn_utils::use_mkldnn_kernel(node.get()))
                    {
                        // TODO: for now, framework formats for src_layer, src_iter, weights_layer and weights_iter
                        // matches to the expected mkldnn format. we need to handle a case to insert convert Op's
                        // if the format doesn't matches.
                        set_native_layouts(external_function, node, false);
                    }
                    else
                    {
                        throw ngraph_error("RNN fused op is only supported in MKLDNN for now.");
                    }
                }

                template <>
                void GPULayout::LAYOUT_DECL(ngraph::op::Softmax)
                {
                    if (mkldnn_utils::use_mkldnn_kernel(node.get()))
                    {
                        auto input_md = mkldnn_utils::get_input_mkldnn_md(node.get(), 0);
                        vector<memory::desc> o_mds;
                        o_mds.push_back(input_md);
                        set_output_layouts(node, o_mds);
                    }
                    else
                    {
                        set_native_layouts(external_function, node);
                    }
                }

                template <>
                void GPULayout::LAYOUT_DECL(ngraph::op::BoundedRelu)
                {
                    if (mkldnn_utils::use_mkldnn_kernel(node.get()))
                    {
                        auto input_md = mkldnn_utils::get_input_mkldnn_md(node.get(), 0);
                        vector<memory::desc> o_mds;
                        o_mds.push_back(input_md);
                        set_output_layouts(node, o_mds);
                    }
                    else
                    {
                        set_native_layouts(external_function, node);
                    }
                }
            }
        }
    }
}

#define TI(x) type_index(typeid(x))

static const runtime::gpu::pass::LayoutOpMap s_dispatcher{
    {TI(ngraph::op::Add), &runtime::gpu::pass::GPULayout::layout<ngraph::op::Add>},
    {TI(ngraph::op::Concat), &runtime::gpu::pass::GPULayout::layout<ngraph::op::Concat>},
    {TI(ngraph::op::AvgPool), &runtime::gpu::pass::GPULayout::layout<ngraph::op::AvgPool>},
    {TI(ngraph::op::AvgPoolBackprop),
     &runtime::gpu::pass::GPULayout::layout<ngraph::op::AvgPoolBackprop>},
    {TI(ngraph::op::Convolution), &runtime::gpu::pass::GPULayout::layout<ngraph::op::Convolution>},
    {TI(ngraph::op::GroupConvolution),
     &runtime::gpu::pass::GPULayout::layout<ngraph::op::GroupConvolution>},
    {TI(ngraph::op::ConvolutionBackpropData),
     &runtime::gpu::pass::GPULayout::layout<ngraph::op::ConvolutionBackpropData>},
    {TI(ngraph::op::ConvolutionBackpropFilters),
     &runtime::gpu::pass::GPULayout::layout<ngraph::op::ConvolutionBackpropFilters>},
    {TI(ngraph::op::MaxPool), &runtime::gpu::pass::GPULayout::layout<ngraph::op::MaxPool>},
    {TI(ngraph::op::MaxPoolWithIndices),
     &runtime::gpu::pass::GPULayout::layout<ngraph::op::MaxPoolWithIndices>},
    {TI(ngraph::op::MaxPoolBackprop),
     &runtime::gpu::pass::GPULayout::layout<ngraph::op::MaxPoolBackprop>},
    {TI(ngraph::op::MaxPoolWithIndicesBackprop),
     &runtime::gpu::pass::GPULayout::layout<ngraph::op::MaxPoolWithIndicesBackprop>},
    {TI(ngraph::op::ConvolutionBias),
     &runtime::gpu::pass::GPULayout::layout<ngraph::op::ConvolutionBias>},
    {TI(ngraph::op::ConvolutionRelu),
     &runtime::gpu::pass::GPULayout::layout<ngraph::op::ConvolutionRelu>},
    {TI(ngraph::op::ConvolutionBiasAdd),
     &runtime::gpu::pass::GPULayout::layout<ngraph::op::ConvolutionBiasAdd>},
    {TI(ngraph::op::ConvolutionBiasBackpropFiltersBias),
     &runtime::gpu::pass::GPULayout::layout<ngraph::op::ConvolutionBiasBackpropFiltersBias>},
    {TI(ngraph::op::BatchNorm), &runtime::gpu::pass::GPULayout::layout<ngraph::op::BatchNorm>},
    {TI(ngraph::op::BatchNormRelu),
     &runtime::gpu::pass::GPULayout::layout<ngraph::op::BatchNormRelu>},
    {TI(ngraph::op::BatchNormBackprop),
     &runtime::gpu::pass::GPULayout::layout<ngraph::op::BatchNormBackprop>},
    {TI(ngraph::op::GetOutputElement),
     &runtime::gpu::pass::GPULayout::layout<ngraph::op::GetOutputElement>},
    {TI(ngraph::op::LRN), &runtime::gpu::pass::GPULayout::layout<ngraph::op::LRN>},
    {TI(ngraph::op::Relu), &runtime::gpu::pass::GPULayout::layout<ngraph::op::Relu>},
    {TI(ngraph::op::Reshape), &runtime::gpu::pass::GPULayout::layout<ngraph::op::Reshape>},
    {TI(ngraph::op::Result), &runtime::gpu::pass::GPULayout::layout<ngraph::op::Result>},
    {TI(ngraph::op::ReluBackprop),
     &runtime::gpu::pass::GPULayout::layout<ngraph::op::ReluBackprop>},
    {TI(ngraph::op::Sigmoid), &runtime::gpu::pass::GPULayout::layout<ngraph::op::Sigmoid>},
    {TI(ngraph::op::SigmoidBackprop),
     &runtime::gpu::pass::GPULayout::layout<ngraph::op::SigmoidBackprop>},
    {TI(ngraph::op::Lstm), &runtime::gpu::pass::GPULayout::layout<ngraph::op::Lstm>},
    {TI(ngraph::op::Rnn), &runtime::gpu::pass::GPULayout::layout<ngraph::op::Rnn>},
    {TI(ngraph::op::Softmax), &runtime::gpu::pass::GPULayout::layout<ngraph::op::Softmax>},
    {TI(ngraph::op::BoundedRelu), &runtime::gpu::pass::GPULayout::layout<ngraph::op::BoundedRelu>},
};

bool runtime::gpu::pass::GPULayout::run_on_call_graph(const std::list<std::shared_ptr<Node>>& nodes)
{
    for (const auto& node : nodes)
    {
        auto& n = *node;
        auto handler = s_dispatcher.find(TI(n));
        if (handler != s_dispatcher.end())
        {
            handler->second(m_external_function, node);
        }
        else
        {
            set_native_layouts(m_external_function, node);
        }
    }

    return false;
}
