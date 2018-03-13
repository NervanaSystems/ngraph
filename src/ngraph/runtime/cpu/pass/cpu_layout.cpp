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

#include "cpu_layout.hpp"
#include "ngraph/descriptor/output.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/ops/add.hpp"
#include "ngraph/ops/avg_pool.hpp"
#include "ngraph/ops/batch_norm.hpp"
#include "ngraph/ops/convolution.hpp"
#include "ngraph/ops/get_output_element.hpp"
#include "ngraph/ops/max_pool.hpp"
#include "ngraph/ops/op.hpp"
#include "ngraph/ops/relu.hpp"
#include "ngraph/ops/result.hpp"
#include "ngraph/runtime/cpu/cpu_layout_descriptor.hpp"
#include "ngraph/runtime/cpu/cpu_op_annotations.hpp"
#include "ngraph/runtime/cpu/mkldnn_utils.hpp"
#include "ngraph/runtime/cpu/ops/conv_bias.hpp"
#include "ngraph/runtime/cpu/ops/convert_layout.hpp"
#include "ngraph/runtime/cpu/ops/sigmoid.hpp"

using namespace std;
using namespace mkldnn;
using namespace ngraph;

shared_ptr<Node> runtime::cpu::pass::CPULayout::insert_input_conversions(
    runtime::cpu::CPU_ExternalFunction* external_function,
    shared_ptr<Node>& node,
    const vector<memory::format>& required_formats)
{
    vector<shared_ptr<Node>> new_args;
    bool replace_node = false;
    uint index = 0;
    for (const descriptor::Input& input : node->get_inputs())
    {
        const auto& output = input.get_output();
        auto tv = output.get_tensor_view();
        auto tvt = tv->get_tensor_view_type();
        auto rank = tvt->get_shape().size();
        auto tvl = tv->get_tensor_view_layout();
        auto mkldnn_tvl = dynamic_cast<runtime::cpu::LayoutDescriptor*>(tvl.get());
        if (!mkldnn_tvl ||
            !runtime::cpu::mkldnn_utils::compare_mkldnn_formats(mkldnn_tvl->get_mkldnn_format(),
                                                                required_formats[index]))
        {
            auto native_axis_order =
                ngraph::runtime::cpu::LayoutDescriptor::create_native_axis_order(rank);
            auto layout =
                std::make_shared<ngraph::runtime::cpu::LayoutDescriptor>(*tv, native_axis_order);
            layout->set_mkldnn_format(required_formats[index]);
            auto new_node = std::shared_ptr<Node>(
                new runtime::cpu::op::ConvertLayout(output.get_node(), output.get_index(), layout));
            new_args.push_back(new_node);
            replace_node = true;
            NGRAPH_DEBUG << "Inserted conversion node " << new_node->get_name() << " between "
                         << output.get_node()->get_name()
                         << "(layout: " << mkldnn_tvl->get_mkldnn_format() << ") and "
                         << node->get_name() << "(layout: " << required_formats[index] << ")";
        }
        else
        {
            new_args.push_back(output.get_node());
        }
        index++;
    }

    shared_ptr<Node> new_node;
    if (replace_node)
    {
        new_node = node->copy_with_new_args(new_args);
        if (node->is_output())
        {
            external_function->get_function()->replace_node(node, new_node);
        }
        else
        {
            ngraph::replace_node(node, new_node);
        }
        NGRAPH_DEBUG << "Replaced " << node->get_name() << " with " << new_node->get_name();
        auto old_op_annotations = static_pointer_cast<ngraph::op::Op>(node)->get_op_annotations();
        static_pointer_cast<ngraph::op::Op>(new_node)->set_op_annotations(old_op_annotations);
        node = new_node;
    }
    return node;
}

void runtime::cpu::pass::CPULayout::set_output_layouts(shared_ptr<Node>& node,
                                                       const vector<memory::format>& output_formats)
{
    for (size_t i = 0; i < node->get_output_size(); ++i)
    {
        auto tv = node->get_output_tensor_view(i);
        auto tvt = tv->get_tensor_view_type();
        auto rank = tvt->get_shape().size();

        auto tvl = tv->get_tensor_view_layout();
        if (tvl)
        {
            throw ngraph_error("Node output layout already set");
        }

        auto native_axis_order =
            ngraph::runtime::cpu::LayoutDescriptor::create_native_axis_order(rank);

        auto layout =
            std::make_shared<ngraph::runtime::cpu::LayoutDescriptor>(*tv, native_axis_order);

        layout->set_mkldnn_format(output_formats[i]);
        tv->set_tensor_view_layout(layout);
        NGRAPH_DEBUG << "Setting Node: " << node->get_name()
                     << " output layout: " << output_formats[i] << endl;
    }
}

void runtime::cpu::pass::CPULayout::set_default_layouts(
    runtime::cpu::CPU_ExternalFunction* external_function, std::shared_ptr<Node> node)
{
    std::vector<shared_ptr<Node>> new_args;
    bool replace_node = false;
    uint index = 0;
    for (const descriptor::Input& input : node->get_inputs())
    {
        const auto& output = input.get_output();
        auto tv = output.get_tensor_view();
        auto tvt = tv->get_tensor_view_type();
        auto rank = tvt->get_shape().size();
        auto tvl = tv->get_tensor_view_layout();
        auto cpu_tvl = dynamic_cast<runtime::cpu::LayoutDescriptor*>(tvl.get());
        if (cpu_tvl && cpu_tvl->get_mkldnn_format() != memory::format::format_undef &&
            !runtime::cpu::mkldnn_utils::compare_mkldnn_formats(
                cpu_tvl->get_mkldnn_format(),
                runtime::cpu::mkldnn_utils::CreateNativeDataFormat(*cpu_tvl)))
        {
            auto native_axis_order =
                ngraph::runtime::cpu::LayoutDescriptor::create_native_axis_order(rank);
            auto layout =
                std::make_shared<ngraph::runtime::cpu::LayoutDescriptor>(*tv, native_axis_order);
            layout->set_mkldnn_format(runtime::cpu::mkldnn_utils::CreateNativeDataFormat(*cpu_tvl));
            auto new_node = std::shared_ptr<Node>(
                new runtime::cpu::op::ConvertLayout(output.get_node(), output.get_index(), layout));
            new_args.push_back(new_node);
            replace_node = true;
            NGRAPH_DEBUG << "Inserted conversion node " << new_node->get_name() << " between "
                         << output.get_node()->get_name()
                         << "(layout: " << cpu_tvl->get_mkldnn_format() << ") and "
                         << node->get_name() << "(layout: default)";
        }
        else
        {
            new_args.push_back(output.get_node());
        }
        index++;
    }

    shared_ptr<Node> new_node;
    if (replace_node)
    {
        new_node = node->copy_with_new_args(new_args);
        if (node->is_output())
        {
            external_function->get_function()->replace_node(node, new_node);
        }
        else
        {
            ngraph::replace_node(node, new_node);
        }
        NGRAPH_DEBUG << "Replaced " << node->get_name() << " with " << new_node->get_name();
        auto old_op_annotations = static_pointer_cast<ngraph::op::Op>(node)->get_op_annotations();
        static_pointer_cast<ngraph::op::Op>(new_node)->set_op_annotations(old_op_annotations);
        node = new_node;
    }

    for (size_t i = 0; i < node->get_output_size(); ++i)
    {
        auto tv = node->get_output_tensor_view(i);
        if (tv->get_tensor_view_layout())
        {
            continue;
        }

        auto tvt = tv->get_tensor_view_type();
        auto rank = tvt->get_shape().size();

        auto native_axis_order =
            ngraph::runtime::cpu::LayoutDescriptor::create_native_axis_order(rank);

        auto layout =
            std::make_shared<ngraph::runtime::cpu::LayoutDescriptor>(*tv, native_axis_order);

        // Set the MKLDNN format to native row-major variants
        layout->set_mkldnn_format(mkldnn_utils::CreateNativeDataFormat(*layout));
        tv->set_tensor_view_layout(layout);
    }
}

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            namespace pass
            {
                template <typename T, bool use_bias>
                void ConvolutionLayout(std::shared_ptr<ngraph::Node> node,
                                       vector<memory::format>& prim_input_formats,
                                       vector<memory::format>& prim_output_formats)
                {
                    auto convolution = static_cast<const T*>(node.get());

                    auto arg0_shape = node->get_input_shape(0);
                    auto arg1_shape = node->get_input_shape(1);
                    auto result_shape = node->get_output_shape(0);
                    auto filter_strides = convolution->get_window_movement_strides();
                    auto padding_below = convolution->get_padding_below();
                    auto padding_above = convolution->get_padding_above();

                    Strides window_dilation_strides_adjusted;

                    for (size_t s : convolution->get_window_dilation_strides())
                    {
                        window_dilation_strides_adjusted.push_back(s - 1);
                    }

                    memory::data_type et = runtime::cpu::mkldnn_utils::get_mkldnn_data_type(
                        node->get_input_element_type(0));

                    engine cpu_engine(engine::cpu, 0);
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
                        memory::dims mkldnn_arg2_shape(arg2_shape.begin(), arg2_shape.end());
                        const memory::desc bias_desc(mkldnn_arg2_shape, et, memory::format::any);

                        fwd_desc.reset(new convolution_forward::desc(prop_kind::forward,
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
                    else
                    {
                        fwd_desc.reset(new convolution_forward::desc(prop_kind::forward,
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
                    convolution_forward::primitive_desc prim_desc(*fwd_desc, cpu_engine);
                    prim_input_formats.push_back(static_cast<memory::format>(
                        prim_desc.src_primitive_desc().desc().data.format));
                    prim_input_formats.push_back(static_cast<memory::format>(
                        prim_desc.weights_primitive_desc().desc().data.format));
                    if (use_bias)
                    {
                        prim_input_formats.push_back(static_cast<memory::format>(
                            prim_desc.bias_primitive_desc().desc().data.format));
                    }
                    prim_output_formats.push_back(static_cast<memory::format>(
                        prim_desc.dst_primitive_desc().desc().data.format));
                }

                template <>
                void CPULayout::LAYOUT_DECL(ngraph::op::Convolution)
                {
                    if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node.get()))
                    {
                        vector<memory::format> prim_input_formats;
                        vector<memory::format> prim_output_formats;
                        ConvolutionLayout<ngraph::op::Convolution, false>(
                            node, prim_input_formats, prim_output_formats);

                        node =
                            insert_input_conversions(external_function, node, prim_input_formats);
                        set_output_layouts(node, prim_output_formats);
                    }
                    else
                    {
                        set_default_layouts(external_function, node);
                    }
                }

                template <>
                void CPULayout::LAYOUT_DECL(ngraph::op::ConvolutionBias)
                {
                    if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node.get()))
                    {
                        vector<memory::format> prim_input_formats;
                        vector<memory::format> prim_output_formats;
                        ConvolutionLayout<ngraph::op::ConvolutionBias, true>(
                            node, prim_input_formats, prim_output_formats);
                        node =
                            insert_input_conversions(external_function, node, prim_input_formats);
                        set_output_layouts(node, prim_output_formats);
                    }
                    else
                    {
                        set_default_layouts(external_function, node);
                    }
                }

                template <>
                void CPULayout::LAYOUT_DECL(ngraph::op::ConvolutionBackpropData)
                {
                    if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node.get()))
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

                        memory::data_type et = runtime::cpu::mkldnn_utils::get_mkldnn_data_type(
                            node->get_input_element_type(0));

                        engine cpu_engine(engine::cpu, 0);
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
                        convolution_forward::primitive_desc fwd_prim_desc(fwd_desc, cpu_engine);

                        convolution_backward_data::primitive_desc prim_desc(
                            bwd_desc, cpu_engine, fwd_prim_desc);

                        vector<memory::format> prim_input_formats;
                        vector<memory::format> prim_output_formats;
                        prim_input_formats.push_back(static_cast<memory::format>(
                            prim_desc.weights_primitive_desc().desc().data.format));
                        prim_input_formats.push_back(static_cast<memory::format>(
                            prim_desc.diff_dst_primitive_desc().desc().data.format));
                        prim_output_formats.push_back(static_cast<memory::format>(
                            prim_desc.diff_src_primitive_desc().desc().data.format));

                        node =
                            insert_input_conversions(external_function, node, prim_input_formats);
                        set_output_layouts(node, prim_output_formats);
                    }
                    else
                    {
                        set_default_layouts(external_function, node);
                    }
                }

                template <typename T, bool use_bias>
                void ConvolutionBackpropFiltersLayout(std::shared_ptr<ngraph::Node> node,
                                                      vector<memory::format>& prim_input_formats,
                                                      vector<memory::format>& prim_output_formats)
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

                    memory::data_type et = runtime::cpu::mkldnn_utils::get_mkldnn_data_type(
                        node->get_input_element_type(0));

                    engine cpu_engine(engine::cpu, 0);
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
                    convolution_forward::primitive_desc fwd_prim_desc(*fwd_desc, cpu_engine);
                    convolution_backward_weights::primitive_desc prim_desc(
                        *bwd_desc, cpu_engine, fwd_prim_desc);

                    prim_input_formats.push_back(static_cast<memory::format>(
                        prim_desc.src_primitive_desc().desc().data.format));
                    prim_input_formats.push_back(static_cast<memory::format>(
                        prim_desc.diff_dst_primitive_desc().desc().data.format));
                    prim_output_formats.push_back(static_cast<memory::format>(
                        prim_desc.diff_weights_primitive_desc().desc().data.format));
                    if (use_bias)
                    {
                        prim_output_formats.push_back(static_cast<memory::format>(
                            prim_desc.diff_bias_primitive_desc().desc().data.format));
                    }
                }

                template <>
                void CPULayout::LAYOUT_DECL(ngraph::op::ConvolutionBackpropFilters)
                {
                    if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node.get()))
                    {
                        vector<memory::format> prim_input_formats;
                        vector<memory::format> prim_output_formats;
                        ConvolutionBackpropFiltersLayout<ngraph::op::ConvolutionBackpropFilters,
                                                         false>(
                            node, prim_input_formats, prim_output_formats);

                        node =
                            insert_input_conversions(external_function, node, prim_input_formats);
                        set_output_layouts(node, prim_output_formats);
                    }
                    else
                    {
                        set_default_layouts(external_function, node);
                    }
                }

                template <>
                void CPULayout::LAYOUT_DECL(ngraph::op::ConvolutionBiasBackpropFiltersBias)
                {
                    if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node.get()))
                    {
                        vector<memory::format> prim_input_formats;
                        vector<memory::format> prim_output_formats;
                        ConvolutionBackpropFiltersLayout<
                            ngraph::op::ConvolutionBiasBackpropFiltersBias,
                            true>(node, prim_input_formats, prim_output_formats);

                        node =
                            insert_input_conversions(external_function, node, prim_input_formats);
                        set_output_layouts(node, prim_output_formats);
                    }
                    else
                    {
                        set_default_layouts(external_function, node);
                    }
                }

                template <>
                void CPULayout::LAYOUT_DECL(ngraph::op::AvgPool)
                {
                    if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node.get()))
                    {
                        auto avg_pool = static_cast<const ngraph::op::AvgPool*>(node.get());

                        auto arg0_shape = node->get_input_shape(0);
                        auto result_shape = node->get_output_shape(0);
                        auto filter_shape = avg_pool->get_window_shape();
                        auto filter_strides = avg_pool->get_window_movement_strides();
                        auto padding_below = avg_pool->get_padding_below();
                        auto padding_above = avg_pool->get_padding_above();

                        memory::data_type et = runtime::cpu::mkldnn_utils::get_mkldnn_data_type(
                            node->get_input_element_type(0));

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

                        auto input_layout =
                            runtime::cpu::mkldnn_utils::get_input_mkldnn_format(node.get(), 0);
                        auto input_desc = memory::desc(mkldnn_arg0_shape, et, input_layout);
                        auto result_desc =
                            memory::desc(mkldnn_result_shape, et, memory::format::any);

                        vector<memory::format> prim_input_formats;
                        vector<memory::format> prim_output_formats;
                        try
                        {
                            auto prim_desc = pooling_forward::primitive_desc(
                                {prop_kind::forward_inference,
                                 algorithm_enumerator,
                                 input_desc,
                                 result_desc,
                                 mkldnn_filter_strides,
                                 mkldnn_filter_shape,
                                 mkldnn_padding_below,
                                 mkldnn_padding_above,
                                 padding_kind::zero},
                                runtime::cpu::mkldnn_utils::global_cpu_engine);
                            prim_input_formats.push_back(input_layout);
                            prim_output_formats.push_back(static_cast<memory::format>(
                                prim_desc.dst_primitive_desc().desc().data.format));
                        }
                        catch (const mkldnn::error& e)
                        {
                            throw ngraph_error("MKLDNN Unsupported pooling layout" +
                                               to_string(input_layout) + e.message);
                        }

                        node =
                            insert_input_conversions(external_function, node, prim_input_formats);
                        set_output_layouts(node, prim_output_formats);
                    }
                    else
                    {
                        set_default_layouts(external_function, node);
                    }
                }

                template <>
                void CPULayout::LAYOUT_DECL(ngraph::op::AvgPoolBackprop)
                {
                    if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node.get()))
                    {
                        auto avg_pool = static_cast<const ngraph::op::AvgPoolBackprop*>(node.get());

                        auto arg0_shape = node->get_input_shape(0);
                        auto result_shape = node->get_output_shape(0);
                        auto filter_shape = avg_pool->get_window_shape();
                        auto filter_strides = avg_pool->get_window_movement_strides();
                        auto padding_below = avg_pool->get_padding_below();
                        auto padding_above = avg_pool->get_padding_above();

                        memory::data_type et = runtime::cpu::mkldnn_utils::get_mkldnn_data_type(
                            node->get_input_element_type(0));

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

                        auto input_layout =
                            runtime::cpu::mkldnn_utils::get_input_mkldnn_format(node.get(), 0);
                        auto input_desc = memory::desc(mkldnn_arg0_shape, et, input_layout);
                        auto result_desc =
                            memory::desc(mkldnn_result_shape, et, memory::format::any);

                        vector<memory::format> prim_input_formats;
                        vector<memory::format> prim_output_formats;
                        try
                        {
                            auto fwd_prim_desc = pooling_forward::primitive_desc(
                                {prop_kind::forward_inference,
                                 algorithm_enumerator,
                                 result_desc,
                                 input_desc,
                                 mkldnn_filter_strides,
                                 mkldnn_filter_shape,
                                 mkldnn_padding_below,
                                 mkldnn_padding_above,
                                 padding_kind::zero},
                                runtime::cpu::mkldnn_utils::global_cpu_engine);
                            auto prim_desc = pooling_backward::primitive_desc(
                                {algorithm_enumerator,
                                 result_desc,
                                 input_desc,
                                 mkldnn_filter_strides,
                                 mkldnn_filter_shape,
                                 mkldnn_padding_below,
                                 mkldnn_padding_above,
                                 padding_kind::zero},
                                runtime::cpu::mkldnn_utils::global_cpu_engine,
                                fwd_prim_desc);
                            prim_input_formats.push_back(input_layout);
                            prim_output_formats.push_back(static_cast<memory::format>(
                                prim_desc.diff_src_primitive_desc().desc().data.format));
                        }
                        catch (const mkldnn::error& e)
                        {
                            throw ngraph_error("MKLDNN Unsupported pooling layout" +
                                               to_string(input_layout) + e.message);
                        }

                        node =
                            insert_input_conversions(external_function, node, prim_input_formats);
                        set_output_layouts(node, prim_output_formats);
                    }
                    else
                    {
                        set_default_layouts(external_function, node);
                    }
                }

                template <>
                void CPULayout::LAYOUT_DECL(ngraph::op::MaxPool)
                {
                    if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node.get()))
                    {
                        auto max_pool = static_cast<const ngraph::op::MaxPool*>(node.get());

                        auto arg0_shape = node->get_input_shape(0);
                        auto result_shape = node->get_output_shape(0);
                        auto filter_shape = max_pool->get_window_shape();
                        auto filter_strides = max_pool->get_window_movement_strides();
                        auto padding_below = max_pool->get_padding_below();
                        auto padding_above = max_pool->get_padding_above();

                        memory::data_type et = runtime::cpu::mkldnn_utils::get_mkldnn_data_type(
                            node->get_input_element_type(0));

                        algorithm algorithm_enumerator = algorithm::pooling_max;

                        memory::dims mkldnn_arg0_shape(arg0_shape.begin(), arg0_shape.end());
                        memory::dims mkldnn_result_shape(result_shape.begin(), result_shape.end());
                        memory::dims mkldnn_filter_shape(filter_shape.begin(), filter_shape.end());
                        memory::dims mkldnn_filter_strides(filter_strides.begin(),
                                                           filter_strides.end());
                        memory::dims mkldnn_padding_below(padding_below.begin(),
                                                          padding_below.end());
                        memory::dims mkldnn_padding_above(padding_above.begin(),
                                                          padding_above.end());

                        auto input_layout =
                            runtime::cpu::mkldnn_utils::get_input_mkldnn_format(node.get(), 0);
                        auto input_desc = memory::desc(mkldnn_arg0_shape, et, input_layout);
                        auto result_desc =
                            memory::desc(mkldnn_result_shape, et, memory::format::any);

                        vector<memory::format> prim_input_formats;
                        vector<memory::format> prim_output_formats;
                        try
                        {
                            auto prim_desc = pooling_forward::primitive_desc(
                                {prop_kind::forward_inference,
                                 algorithm_enumerator,
                                 input_desc,
                                 result_desc,
                                 mkldnn_filter_strides,
                                 mkldnn_filter_shape,
                                 mkldnn_padding_below,
                                 mkldnn_padding_above,
                                 padding_kind::zero},
                                runtime::cpu::mkldnn_utils::global_cpu_engine);
                            prim_input_formats.push_back(input_layout);
                            prim_output_formats.push_back(static_cast<memory::format>(
                                prim_desc.dst_primitive_desc().desc().data.format));
                            // TODO (jbobba): Add workspace layouts here
                        }
                        catch (const mkldnn::error& e)
                        {
                            throw ngraph_error("MKLDNN Unsupported pooling fwd layout" +
                                               to_string(input_layout) + e.message);
                        }

                        node =
                            insert_input_conversions(external_function, node, prim_input_formats);
                        set_output_layouts(node, prim_output_formats);
                    }
                    else
                    {
                        set_default_layouts(external_function, node);
                    }
                }

                template <>
                void CPULayout::LAYOUT_DECL(ngraph::op::MaxPoolBackprop)
                {
                    if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node.get()))
                    {
                        auto max_pool = static_cast<const ngraph::op::MaxPoolBackprop*>(node.get());

                        // arg 0 - fprop input
                        // arg 1 - delta
                        // Propagate fprop's input layout
                        auto arg0_shape = node->get_input_shape(0);
                        auto arg1_shape = node->get_input_shape(1);
                        auto result_shape = node->get_output_shape(0);
                        auto filter_shape = max_pool->get_window_shape();
                        auto filter_strides = max_pool->get_window_movement_strides();
                        auto padding_below = max_pool->get_padding_below();
                        auto padding_above = max_pool->get_padding_above();

                        memory::data_type et = runtime::cpu::mkldnn_utils::get_mkldnn_data_type(
                            node->get_input_element_type(1));

                        algorithm algorithm_enumerator = algorithm::pooling_max;

                        memory::dims mkldnn_arg0_shape(arg0_shape.begin(), arg0_shape.end());
                        memory::dims mkldnn_arg1_shape(arg1_shape.begin(), arg1_shape.end());
                        memory::dims mkldnn_result_shape(result_shape.begin(), result_shape.end());
                        memory::dims mkldnn_filter_shape(filter_shape.begin(), filter_shape.end());
                        memory::dims mkldnn_filter_strides(filter_strides.begin(),
                                                           filter_strides.end());
                        memory::dims mkldnn_padding_below(padding_below.begin(),
                                                          padding_below.end());
                        memory::dims mkldnn_padding_above(padding_above.begin(),
                                                          padding_above.end());

                        auto fprop_input_layout =
                            runtime::cpu::mkldnn_utils::get_input_mkldnn_format(node.get(), 0);

                        auto diff_dst_desc =
                            memory::desc(mkldnn_arg1_shape, et, fprop_input_layout);
                        auto diff_src_desc =
                            memory::desc(mkldnn_arg0_shape, et, memory::format::any);

                        vector<memory::format> prim_input_formats;
                        vector<memory::format> prim_output_formats;
                        try
                        {
                            auto fwd_prim_desc = pooling_forward::primitive_desc(
                                {prop_kind::forward_training,
                                 algorithm_enumerator,
                                 diff_src_desc,
                                 diff_dst_desc,
                                 mkldnn_filter_strides,
                                 mkldnn_filter_shape,
                                 mkldnn_padding_below,
                                 mkldnn_padding_above,
                                 padding_kind::zero},
                                runtime::cpu::mkldnn_utils::global_cpu_engine);

                            auto prim_desc = pooling_backward::primitive_desc(
                                {algorithm_enumerator,
                                 diff_src_desc,
                                 diff_dst_desc,
                                 mkldnn_filter_strides,
                                 mkldnn_filter_shape,
                                 mkldnn_padding_below,
                                 mkldnn_padding_above,
                                 padding_kind::zero},
                                runtime::cpu::mkldnn_utils::global_cpu_engine,
                                fwd_prim_desc);
                            prim_input_formats.push_back(fprop_input_layout);
                            prim_input_formats.push_back(fprop_input_layout);
                            prim_output_formats.push_back(static_cast<memory::format>(
                                prim_desc.diff_src_primitive_desc().desc().data.format));
                        }
                        catch (const mkldnn::error& e)
                        {
                            throw ngraph_error("MKLDNN Unsupported pooling layout" +
                                               to_string(fprop_input_layout) + e.message);
                        }

                        node =
                            insert_input_conversions(external_function, node, prim_input_formats);
                        set_output_layouts(node, prim_output_formats);
                    }
                    else
                    {
                        set_default_layouts(external_function, node);
                    }
                }

                template <>
                void CPULayout::LAYOUT_DECL(ngraph::op::Result)
                {
                    auto input_layout =
                        runtime::cpu::mkldnn_utils::get_input_mkldnn_format(node.get(), 0);
                    vector<memory::format> prim_output_formats;
                    prim_output_formats.push_back(input_layout);
                    set_output_layouts(node, prim_output_formats);
                }

                template <>
                void CPULayout::LAYOUT_DECL(ngraph::op::GetOutputElement)
                {
                    auto goe = static_cast<const ngraph::op::GetOutputElement*>(node.get());
                    auto input_layout = runtime::cpu::mkldnn_utils::get_input_mkldnn_format(
                        node.get(), goe->get_n());
                    vector<memory::format> prim_output_formats;
                    prim_output_formats.push_back(input_layout);
                    set_output_layouts(node, prim_output_formats);
                }

                template <>
                void CPULayout::LAYOUT_DECL(ngraph::op::Relu)
                {
                    if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node.get()))
                    {
                        auto input_layout =
                            runtime::cpu::mkldnn_utils::get_input_mkldnn_format(node.get(), 0);
                        vector<memory::format> prim_output_formats;
                        prim_output_formats.push_back(input_layout);
                        set_output_layouts(node, prim_output_formats);
                    }
                    else
                    {
                        set_default_layouts(external_function, node);
                    }
                }

                template <>
                void CPULayout::LAYOUT_DECL(ngraph::op::Sigmoid)
                {
                    if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node.get()))
                    {
                        auto input_layout =
                            runtime::cpu::mkldnn_utils::get_input_mkldnn_format(node.get(), 0);
                        vector<memory::format> prim_output_formats;
                        prim_output_formats.push_back(input_layout);
                        set_output_layouts(node, prim_output_formats);
                    }
                    else
                    {
                        set_default_layouts(external_function, node);
                    }
                }

                template <>
                void CPULayout::LAYOUT_DECL(ngraph::op::SigmoidBackprop)
                {
                    if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node.get()))
                    {
                        auto input_layout =
                            runtime::cpu::mkldnn_utils::get_input_mkldnn_format(node.get(), 0);
                        vector<memory::format> prim_input_formats;
                        vector<memory::format> prim_output_formats;
                        //ensure delta and input have same layout
                        prim_input_formats.push_back(input_layout);
                        prim_input_formats.push_back(input_layout);
                        prim_output_formats.push_back(input_layout);
                        node =
                            insert_input_conversions(external_function, node, prim_input_formats);
                        set_output_layouts(node, prim_output_formats);
                    }
                    else
                    {
                        set_default_layouts(external_function, node);
                    }
                }

                template <>
                void CPULayout::LAYOUT_DECL(ngraph::op::ReluBackprop)
                {
                    if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node.get()))
                    {
                        auto input_layout =
                            runtime::cpu::mkldnn_utils::get_input_mkldnn_format(node.get(), 0);

                        vector<memory::format> prim_input_formats;
                        vector<memory::format> prim_output_formats;
                        prim_input_formats.push_back(input_layout);
                        prim_input_formats.push_back(input_layout);
                        prim_output_formats.push_back(input_layout);
                        node =
                            insert_input_conversions(external_function, node, prim_input_formats);
                        set_output_layouts(node, prim_output_formats);
                    }
                    else
                    {
                        set_default_layouts(external_function, node);
                    }
                }

                template <>
                void CPULayout::LAYOUT_DECL(ngraph::op::BatchNorm)
                {
                    if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node.get()))
                    {
                        auto input_layout =
                            runtime::cpu::mkldnn_utils::get_input_mkldnn_format(node.get(), 2);

                        vector<memory::format> prim_input_formats;
                        vector<memory::format> prim_output_formats;
                        prim_input_formats.push_back(memory::format::x);
                        prim_input_formats.push_back(memory::format::x);
                        prim_input_formats.push_back(input_layout);
                        prim_output_formats.push_back(input_layout);
                        prim_output_formats.push_back(memory::format::x);
                        prim_output_formats.push_back(memory::format::x);
                        node =
                            insert_input_conversions(external_function, node, prim_input_formats);
                        set_output_layouts(node, prim_output_formats);
                    }
                    else
                    {
                        throw ngraph_error("Batchnorm only supported in MKLDNN for now");
                    }
                }

                template <>
                void CPULayout::LAYOUT_DECL(ngraph::op::BatchNormBackprop)
                {
                    if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node.get()))
                    {
                        auto delta_layout =
                            runtime::cpu::mkldnn_utils::get_input_mkldnn_format(node.get(), 5);

                        vector<memory::format> prim_input_formats;
                        vector<memory::format> prim_output_formats;

                        prim_input_formats.push_back(memory::format::x);  // gamma
                        prim_input_formats.push_back(memory::format::x);  // beta
                        prim_input_formats.push_back(delta_layout);       // input
                        prim_input_formats.push_back(memory::format::x);  // mean
                        prim_input_formats.push_back(memory::format::x);  // variance
                        prim_input_formats.push_back(delta_layout);       // delta
                        prim_output_formats.push_back(delta_layout);      // dinput
                        prim_output_formats.push_back(memory::format::x); // dgamma
                        prim_output_formats.push_back(memory::format::x); // dbeta
                        node =
                            insert_input_conversions(external_function, node, prim_input_formats);
                        set_output_layouts(node, prim_output_formats);
                    }
                    else
                    {
                        throw ngraph_error("Batchnorm Backprop only supported in MKLDNN for now");
                    }
                }

                template <>
                void CPULayout::LAYOUT_DECL(ngraph::op::Add)
                {
                    if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node.get()))
                    {
                        auto input0_layout =
                            runtime::cpu::mkldnn_utils::get_input_mkldnn_format(node.get(), 0);

                        vector<memory::format> prim_input_formats;
                        vector<memory::format> prim_output_formats;
                        prim_input_formats.push_back(input0_layout);
                        prim_input_formats.push_back(input0_layout);
                        prim_output_formats.push_back(input0_layout);
                        node =
                            insert_input_conversions(external_function, node, prim_input_formats);
                        set_output_layouts(node, prim_output_formats);
                    }
                    else
                    {
                        set_default_layouts(external_function, node);
                    }
                }
            }
        }
    }
}

#define TI(x) type_index(typeid(x))

static const runtime::cpu::pass::LayoutOpMap s_dispatcher{
    {TI(ngraph::op::Add), &runtime::cpu::pass::CPULayout::layout<ngraph::op::Add>},
    {TI(ngraph::op::AvgPool), &runtime::cpu::pass::CPULayout::layout<ngraph::op::AvgPool>},
    {TI(ngraph::op::AvgPoolBackprop),
     &runtime::cpu::pass::CPULayout::layout<ngraph::op::AvgPoolBackprop>},
    {TI(ngraph::op::Convolution), &runtime::cpu::pass::CPULayout::layout<ngraph::op::Convolution>},
    {TI(ngraph::op::ConvolutionBackpropData),
     &runtime::cpu::pass::CPULayout::layout<ngraph::op::ConvolutionBackpropData>},
    {TI(ngraph::op::ConvolutionBackpropFilters),
     &runtime::cpu::pass::CPULayout::layout<ngraph::op::ConvolutionBackpropFilters>},
    {TI(ngraph::op::MaxPool), &runtime::cpu::pass::CPULayout::layout<ngraph::op::MaxPool>},
    {TI(ngraph::op::MaxPoolBackprop),
     &runtime::cpu::pass::CPULayout::layout<ngraph::op::MaxPoolBackprop>},
    {TI(ngraph::op::ConvolutionBias),
     &runtime::cpu::pass::CPULayout::layout<ngraph::op::ConvolutionBias>},
    {TI(ngraph::op::ConvolutionBiasBackpropFiltersBias),
     &runtime::cpu::pass::CPULayout::layout<ngraph::op::ConvolutionBiasBackpropFiltersBias>},
    {TI(ngraph::op::BatchNorm), &runtime::cpu::pass::CPULayout::layout<ngraph::op::BatchNorm>},
    {TI(ngraph::op::BatchNormBackprop),
     &runtime::cpu::pass::CPULayout::layout<ngraph::op::BatchNormBackprop>},
    {TI(ngraph::op::GetOutputElement),
     &runtime::cpu::pass::CPULayout::layout<ngraph::op::GetOutputElement>},
    {TI(ngraph::op::Relu), &runtime::cpu::pass::CPULayout::layout<ngraph::op::Relu>},
    {TI(ngraph::op::Result), &runtime::cpu::pass::CPULayout::layout<ngraph::op::Result>},
    {TI(ngraph::op::ReluBackprop),
     &runtime::cpu::pass::CPULayout::layout<ngraph::op::ReluBackprop>},
    {TI(ngraph::op::Sigmoid), &runtime::cpu::pass::CPULayout::layout<ngraph::op::Sigmoid>},
    {TI(ngraph::op::SigmoidBackprop),
     &runtime::cpu::pass::CPULayout::layout<ngraph::op::SigmoidBackprop>},
};

bool runtime::cpu::pass::CPULayout::run_on_call_graph(const std::list<std::shared_ptr<Node>>& nodes)
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
            set_default_layouts(m_external_function, node);
        }
    }

    return false;
}
