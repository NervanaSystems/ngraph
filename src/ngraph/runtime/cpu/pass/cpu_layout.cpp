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

#include <algorithm>
#include <memory>
#include <string>
#include <typeindex>
#include <typeinfo>

#include <mkldnn.hpp>

#include "cpu_layout.hpp"
#include "ngraph/axis_vector.hpp"
#include "ngraph/descriptor/output.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/avg_pool.hpp"
#include "ngraph/op/batch_norm.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/convolution.hpp"
#include "ngraph/op/experimental/quantized_avg_pool.hpp"
#include "ngraph/op/experimental/quantized_conv.hpp"
#include "ngraph/op/experimental/quantized_conv_bias.hpp"
#include "ngraph/op/experimental/quantized_conv_relu.hpp"
#include "ngraph/op/experimental/quantized_max_pool.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/op/lrn.hpp"
#include "ngraph/op/max_pool.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/op/relu.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/result.hpp"
#include "ngraph/op/sigmoid.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/op/softmax.hpp"
#include "ngraph/runtime/cpu/cpu_executor.hpp"
#include "ngraph/runtime/cpu/cpu_layout_descriptor.hpp"
#include "ngraph/runtime/cpu/cpu_op_annotations.hpp"
#include "ngraph/runtime/cpu/mkldnn_utils.hpp"
#include "ngraph/runtime/cpu/op/batch_norm_relu.hpp"
#include "ngraph/runtime/cpu/op/bounded_relu.hpp"
#include "ngraph/runtime/cpu/op/conv_add.hpp"
#include "ngraph/runtime/cpu/op/conv_bias.hpp"
#include "ngraph/runtime/cpu/op/conv_relu.hpp"
#include "ngraph/runtime/cpu/op/convert_layout.hpp"
#include "ngraph/runtime/cpu/op/group_conv.hpp"
#include "ngraph/runtime/cpu/op/group_conv_bias.hpp"
#include "ngraph/runtime/cpu/op/leaky_relu.hpp"
#include "ngraph/runtime/cpu/op/lstm.hpp"
#include "ngraph/runtime/cpu/op/max_pool_with_indices.hpp"
#include "ngraph/runtime/cpu/op/rnn.hpp"

using namespace std;
using namespace mkldnn;
using namespace ngraph;
using namespace ngraph::runtime::cpu;

// Check if the input layout matches the layout requested in `required_mds`
// If not, insert a layout conversion node between the input tensor and
// the `node`. For now, only MKLDNN nodes/kernels can request specific layouts
static shared_ptr<Node>
    insert_input_conversions(runtime::cpu::CPU_ExternalFunction* external_function,
                             shared_ptr<Node>& node,
                             const vector<memory::desc>& required_mds)
{
    vector<shared_ptr<Node>> new_args;
    bool replace_node = false;
    uint index = 0;

    if (required_mds.size() != node->get_input_size())
    {
        throw ngraph_error("In insert_input_conversions: expects number of required layouts (" +
                           to_string(required_mds.size()) + ")to match number of node inputs (" +
                           to_string(node->get_input_size()) + ")");
    }

    for (const descriptor::Input& input : node->get_inputs())
    {
        const auto& output = input.get_output();
        auto tv = output.get_tensor_ptr();
        auto tvl = dynamic_pointer_cast<runtime::cpu::LayoutDescriptor>(tv->get_tensor_layout());
        if (!tvl)
        {
            throw ngraph_error(
                "In insert_input_conversions: Expecting Layout descriptor to be already set on " +
                output.get_node()->get_name());
        }

        if (input.get_shape() == Shape{})
        {
            tvl->set_mkldnn_md(required_mds[index]);
        }
        if (!tvl->is_mkldnn_layout())
        {
            throw ngraph_error(
                "In insert_input_conversions: MKLDNN layout requested on an non-MKLDNN compatible "
                "layout " +
                output.get_node()->get_name());
        }

        if (!mkldnn_utils::compare_mkldnn_mds(tvl->get_mkldnn_md(), required_mds[index]))
        {
            auto layout = std::make_shared<ngraph::runtime::cpu::LayoutDescriptor>(*tv);
            layout->set_mkldnn_md(required_mds[index]);
            auto new_node = std::shared_ptr<Node>(
                new runtime::cpu::op::ConvertLayout(output.get_node(), output.get_index(), layout));
            new_args.push_back(new_node);
            replace_node = true;
            NGRAPH_DEBUG << "Inserted conversion node " << new_node->get_name() << " between "
                         << output.get_node()->get_name()
                         << "(layout: " << tvl->get_mkldnn_md().data.format << ") and "
                         << node->get_name() << "(layout: " << required_mds[index].data.format
                         << ")";
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

static void set_output_layouts(shared_ptr<Node>& node, const vector<memory::desc>& output_mds)
{
    for (size_t i = 0; i < node->get_output_size(); ++i)
    {
        auto tv = node->get_output_tensor_ptr(i);
        auto tvl = tv->get_tensor_layout();
        if (tvl)
        {
            throw ngraph_error("Node (" + node->get_name() +
                               ") output layout already set. This node is most likely present in "
                               "multiple graphs which could lead to unpredictable results.");
        }
        auto layout = std::make_shared<ngraph::runtime::cpu::LayoutDescriptor>(*tv);
        layout->set_mkldnn_md(output_mds[i]);
        tv->set_tensor_layout(layout);
        NGRAPH_DEBUG << "Setting Node: " << node->get_name()
                     << " output layout: " << output_mds[i].data.format << endl;
    }
}

static void set_native_layouts(runtime::cpu::CPU_ExternalFunction* external_function,
                               std::shared_ptr<Node> node,
                               bool use_replace = true)
{
    std::vector<shared_ptr<Node>> new_args;
    bool replace_node = false;
    uint index = 0;
    for (descriptor::Input& input : node->get_inputs())
    {
        const auto& output = input.get_output();
        auto tv = output.get_tensor_ptr();
        auto et = tv->get_element_type();
        auto shape = tv->get_shape();
        auto tvl = tv->get_tensor_layout();
        auto cpu_tvl = dynamic_cast<runtime::cpu::LayoutDescriptor*>(tvl.get());

        if (cpu_tvl && cpu_tvl->is_mkldnn_layout())
        {
            auto native_md =
                mkldnn_utils::create_blocked_mkldnn_md(shape, cpu_tvl->get_strides(), et);
            if (!mkldnn_utils::compare_mkldnn_mds(cpu_tvl->get_mkldnn_md(), native_md))
            {
                auto layout = std::make_shared<ngraph::runtime::cpu::LayoutDescriptor>(*tv);
                layout->set_mkldnn_md(native_md);
                auto new_node = std::shared_ptr<Node>(new runtime::cpu::op::ConvertLayout(
                    output.get_node(), output.get_index(), layout));
                new_args.push_back(new_node);
                if (use_replace)
                {
                    replace_node = true;
                }
                else
                {
                    input.replace_output(new_node->get_outputs().at(0));
                }
                NGRAPH_DEBUG << "Inserted conversion node " << new_node->get_name() << " between "
                             << output.get_node()->get_name()
                             << "(layout: " << cpu_tvl->get_mkldnn_md().data.format << ") and "
                             << node->get_name() << "(layout: default)";
            }
            else
            {
                new_args.push_back(output.get_node());
            }
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
        auto tv = node->get_output_tensor_ptr(i);
        if (tv->get_tensor_layout())
        {
            // TODO(jbobba): Should this be an error instead?
            // Some unit tests are sharing nodes across graphs
            continue;
        }

        auto shape = tv->get_shape();
        auto et = tv->get_element_type();
        auto layout = std::make_shared<ngraph::runtime::cpu::LayoutDescriptor>(*tv);
        if (mkldnn_utils::can_create_mkldnn_md(shape, layout->get_strides(), et))
        {
            auto native_md =
                mkldnn_utils::create_blocked_mkldnn_md(shape, layout->get_strides(), et);
            layout->set_mkldnn_md(native_md);
        }
        tv->set_tensor_layout(layout);
    }
}

static void set_layouts_unaryeltwise(ngraph::runtime::cpu::CPU_ExternalFunction* external_function,
                                     std::shared_ptr<ngraph::Node> node)
{
    auto input_md = mkldnn_utils::get_input_mkldnn_md(node.get(), 0);
    // Non MKLDNN kernels can handle MKLDNN layouts as long as there are not padded
    bool md_check = input_md.data.format != mkldnn_format_undef &&
                    !mkldnn_utils::is_mkldnn_padded_layout(
                        input_md, ngraph::get_default_order(node->get_input_shape(0)));
    if (mkldnn_utils::use_mkldnn_kernel(node.get()) || md_check)
    {
        vector<memory::desc> o_mds;
        o_mds.push_back(input_md);
        set_output_layouts(node, o_mds);
    }
    else
    {
        set_native_layouts(external_function, node);
    }
}

void set_layouts_binaryeltwise(ngraph::runtime::cpu::CPU_ExternalFunction* external_function,
                               std::shared_ptr<ngraph::Node> node)
{
    std::vector<mkldnn::memory::desc> arg_mds{mkldnn_utils::get_input_mkldnn_md(node.get(), 0),
                                              mkldnn_utils::get_input_mkldnn_md(node.get(), 1)};
    bool md_check = arg_mds[0].data.format != mkldnn_format_undef &&
                    arg_mds[1].data.format != mkldnn_format_undef &&
                    !mkldnn_utils::is_mkldnn_padded_layout(
                        arg_mds[0], ngraph::get_default_order(node->get_input_shape(0))) &&
                    !mkldnn_utils::is_mkldnn_padded_layout(
                        arg_mds[1], ngraph::get_default_order(node->get_input_shape(1)));
    if (mkldnn_utils::use_mkldnn_kernel(node.get()) || md_check)
    {
        vector<memory::desc> i_mds;
        vector<memory::desc> o_mds;
        int select = 0;
        if (std::getenv("NGRAPH_PASS_CPU_LAYOUT_ELTWISE") != nullptr)
        {
            const int user_select = std::atoi(std::getenv("NGRAPH_PASS_CPU_LAYOUT_ELTWISE"));
            select = (user_select == 0 || user_select == 1) ? user_select : select;
        }
        i_mds.push_back(arg_mds[select]);
        i_mds.push_back(arg_mds[select]);
        o_mds.push_back(arg_mds[select]);
        node = insert_input_conversions(external_function, node, i_mds);
        set_output_layouts(node, o_mds);
    }
    else
    {
        set_native_layouts(external_function, node);
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
                                       vector<memory::desc>& i_mds,
                                       vector<memory::desc>& o_mds)
                {
                    auto convolution = static_cast<const T*>(node.get());

                    auto arg0_shape = node->get_input_shape(0);
                    auto arg1_shape = node->get_input_shape(1);

                    // Convert filters to MKLDNN shape
                    // o,i,h,w -> g,o,i,h,w (e.g., {6, 2, 1, 1}, groups = 2 -> {2, 3, 1, 1, 1})
                    if (auto gconv = std::dynamic_pointer_cast<ngraph::op::GroupConvolution>(node))
                    {
                        arg1_shape = gconv->get_weights_dimensions();
                    }
                    if (auto gconv =
                            std::dynamic_pointer_cast<ngraph::op::GroupConvolutionBias>(node))
                    {
                        arg1_shape = gconv->get_weights_dimensions();
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

                    memory::data_type et_weights = runtime::cpu::mkldnn_utils::get_mkldnn_data_type(
                        node->get_input_element_type(1));
                    memory::data_type et_result = runtime::cpu::mkldnn_utils::get_mkldnn_data_type(
                        node->get_output_element_type(0));

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
                    const memory::desc weights_desc(
                        mkldnn_arg1_shape, et_weights, memory::format::any);
                    const memory::desc result_desc(
                        mkldnn_result_shape, et_result, memory::format::any);
                    std::unique_ptr<convolution_forward::desc> fwd_desc{nullptr};
                    if (use_bias)
                    {
                        memory::data_type et_bias =
                            mkldnn_utils::get_mkldnn_data_type(node->get_input_element_type(2));
                        auto arg2_shape = node->get_input_shape(2);
                        memory::dims mkldnn_arg2_shape(arg2_shape.begin(), arg2_shape.end());
                        const memory::desc bias_desc(
                            mkldnn_arg2_shape, et_bias, memory::format::any);
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
                    convolution_forward::primitive_desc prim_desc(*fwd_desc, cpu_engine);
                    i_mds.push_back(prim_desc.src_primitive_desc().desc());
                    i_mds.push_back(prim_desc.weights_primitive_desc().desc());

                    if (use_bias)
                    {
                        i_mds.push_back(prim_desc.bias_primitive_desc().desc());
                    }
                    o_mds.push_back(prim_desc.dst_primitive_desc().desc());
                }

                template <>
                void CPULayout::LAYOUT_DECL(ngraph::op::QuantizedConvolution)
                {
                    if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node.get()))
                    {
                        vector<memory::desc> i_mds;
                        vector<memory::desc> o_mds;
                        ConvolutionLayout<ngraph::op::QuantizedConvolution, false>(
                            node, i_mds, o_mds);

                        auto scale_input_md = mkldnn_utils::create_default_mkldnn_md(
                            node.get(), 2, false, memory::format::x);

                        i_mds.push_back(scale_input_md);
                        node = insert_input_conversions(external_function, node, i_mds);
                        set_output_layouts(node, o_mds);
                    }
                    else
                    {
                        set_native_layouts(external_function, node);
                    }
                }

                template <>
                void CPULayout::LAYOUT_DECL(ngraph::op::Convolution)
                {
                    if (mkldnn_utils::use_mkldnn_kernel(node.get()))
                    {
                        vector<memory::desc> i_mds;
                        vector<memory::desc> o_mds;
                        ConvolutionLayout<ngraph::op::Convolution, false>(node, i_mds, o_mds);

                        node = insert_input_conversions(external_function, node, i_mds);
                        set_output_layouts(node, o_mds);
                    }
                    else
                    {
                        set_native_layouts(external_function, node);
                    }
                }

                template <>
                void CPULayout::LAYOUT_DECL(ngraph::op::GroupConvolution)
                {
                    if (mkldnn_utils::use_mkldnn_kernel(node.get()))
                    {
                        vector<memory::desc> i_mds;
                        vector<memory::desc> o_mds;
                        ConvolutionLayout<ngraph::op::GroupConvolution, false>(node, i_mds, o_mds);

                        node = insert_input_conversions(external_function, node, i_mds);
                        set_output_layouts(node, o_mds);
                    }
                    else
                    {
                        set_native_layouts(external_function, node);
                    }
                }

                template <>
                void CPULayout::LAYOUT_DECL(ngraph::op::GroupConvolutionBias)
                {
                    if (mkldnn_utils::use_mkldnn_kernel(node.get()))
                    {
                        vector<memory::desc> i_mds;
                        vector<memory::desc> o_mds;
                        ConvolutionLayout<ngraph::op::GroupConvolutionBias, true>(
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
                void CPULayout::LAYOUT_DECL(ngraph::op::ConvolutionBias)
                {
                    if (mkldnn_utils::use_mkldnn_kernel(node.get()))
                    {
                        vector<memory::desc> i_mds;
                        vector<memory::desc> o_mds;
                        ConvolutionLayout<ngraph::op::ConvolutionBias, true>(node, i_mds, o_mds);
                        node = insert_input_conversions(external_function, node, i_mds);
                        set_output_layouts(node, o_mds);
                    }
                    else
                    {
                        set_native_layouts(external_function, node);
                    }
                }

                template <>
                void CPULayout::LAYOUT_DECL(ngraph::op::QuantizedConvolutionBias)
                {
                    if (mkldnn_utils::use_mkldnn_kernel(node.get()))
                    {
                        vector<memory::desc> i_mds;
                        vector<memory::desc> o_mds;
                        ConvolutionLayout<ngraph::op::QuantizedConvolutionBias, true>(
                            node, i_mds, o_mds);

                        auto scale_input_md = mkldnn_utils::create_default_mkldnn_md(
                            node.get(), 3, false, memory::format::x);

                        i_mds.push_back(scale_input_md);

                        node = insert_input_conversions(external_function, node, i_mds);
                        set_output_layouts(node, o_mds);
                    }
                    else
                    {
                        set_native_layouts(external_function, node);
                    }
                }

                template <>
                void CPULayout::LAYOUT_DECL(ngraph::op::QuantizedConvolutionBiasAdd)
                {
                    if (mkldnn_utils::use_mkldnn_kernel(node.get()))
                    {
                        vector<memory::desc> i_mds;
                        vector<memory::desc> o_mds;
                        ConvolutionLayout<ngraph::op::QuantizedConvolutionBiasAdd, true>(
                            node, i_mds, o_mds);

                        auto scale_input_md = mkldnn_utils::create_default_mkldnn_md(
                            node.get(), 4, false, memory::format::x);
                        auto sum_scale_input_md = mkldnn_utils::create_default_mkldnn_md(
                            node.get(), 5, false, memory::format::x);

                        i_mds.push_back(o_mds[0]);
                        i_mds.push_back(scale_input_md);
                        i_mds.push_back(sum_scale_input_md);

                        node = insert_input_conversions(external_function, node, i_mds);
                        set_output_layouts(node, o_mds);
                    }
                    else
                    {
                        set_native_layouts(external_function, node);
                    }
                }

                template <>
                void CPULayout::LAYOUT_DECL(ngraph::op::QuantizedConvolutionBiasSignedAdd)
                {
                    if (mkldnn_utils::use_mkldnn_kernel(node.get()))
                    {
                        vector<memory::desc> i_mds;
                        vector<memory::desc> o_mds;
                        ConvolutionLayout<ngraph::op::QuantizedConvolutionBiasSignedAdd, true>(
                            node, i_mds, o_mds);

                        auto scale_input_md = mkldnn_utils::create_default_mkldnn_md(
                            node.get(), 4, false, memory::format::x);
                        auto sum_scale_input_md = mkldnn_utils::create_default_mkldnn_md(
                            node.get(), 5, false, memory::format::x);

                        i_mds.push_back(o_mds[0]);
                        i_mds.push_back(scale_input_md);
                        i_mds.push_back(sum_scale_input_md);

                        node = insert_input_conversions(external_function, node, i_mds);
                        set_output_layouts(node, o_mds);
                    }
                    else
                    {
                        set_native_layouts(external_function, node);
                    }
                }

                template <>
                void CPULayout::LAYOUT_DECL(ngraph::op::ConvolutionRelu)
                {
                    if (mkldnn_utils::use_mkldnn_kernel(node.get()))
                    {
                        vector<memory::desc> i_mds;
                        vector<memory::desc> o_mds;
                        ConvolutionLayout<ngraph::op::ConvolutionRelu, false>(node, i_mds, o_mds);
                        node = insert_input_conversions(external_function, node, i_mds);
                        set_output_layouts(node, o_mds);
                    }
                    else
                    {
                        set_native_layouts(external_function, node);
                    }
                }

                template <>
                void CPULayout::LAYOUT_DECL(ngraph::op::QuantizedConvolutionRelu)
                {
                    if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node.get()))
                    {
                        vector<memory::desc> i_mds;
                        vector<memory::desc> o_mds;
                        ConvolutionLayout<ngraph::op::QuantizedConvolutionRelu, false>(
                            node, i_mds, o_mds);

                        auto scale_input_md = mkldnn_utils::create_default_mkldnn_md(
                            node.get(), 2, false, memory::format::x);

                        i_mds.push_back(scale_input_md);

                        node = insert_input_conversions(external_function, node, i_mds);
                        set_output_layouts(node, o_mds);
                    }
                    else
                    {
                        set_native_layouts(external_function, node);
                    }
                }

                template <>
                void CPULayout::LAYOUT_DECL(ngraph::op::ConvolutionBiasAdd)
                {
                    if (mkldnn_utils::use_mkldnn_kernel(node.get()))
                    {
                        vector<memory::desc> i_mds;
                        vector<memory::desc> o_mds;
                        ConvolutionLayout<ngraph::op::ConvolutionBiasAdd, true>(node, i_mds, o_mds);
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
                void CPULayout::LAYOUT_DECL(ngraph::op::ConvolutionAdd)
                {
                    if (mkldnn_utils::use_mkldnn_kernel(node.get()))
                    {
                        vector<memory::desc> i_mds;
                        vector<memory::desc> o_mds;
                        ConvolutionLayout<ngraph::op::ConvolutionAdd, false>(node, i_mds, o_mds);
                        // Force second input to sum to use the same layout as convolution output
                        i_mds.push_back(o_mds[0]);
                        node = insert_input_conversions(external_function, node, i_mds);
                        set_output_layouts(node, o_mds);
                    }
                    else
                    {
                        throw ngraph_error("ConvolutionAdd only supported in MKLDNN for now");
                    }
                }

                template <>
                void CPULayout::LAYOUT_DECL(ngraph::op::ConvolutionBackpropData)
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
                    convolution_forward::primitive_desc fwd_prim_desc(*fwd_desc, cpu_engine);
                    convolution_backward_weights::primitive_desc prim_desc(
                        *bwd_desc, cpu_engine, fwd_prim_desc);

                    i_mds.push_back(prim_desc.src_primitive_desc().desc());
                    i_mds.push_back(prim_desc.diff_dst_primitive_desc().desc());
                    o_mds.push_back(prim_desc.diff_weights_primitive_desc().desc());
                    if (use_bias)
                    {
                        o_mds.push_back(prim_desc.diff_bias_primitive_desc().desc());
                    }
                }

                template <>
                void CPULayout::LAYOUT_DECL(ngraph::op::ConvolutionBackpropFilters)
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
                void CPULayout::LAYOUT_DECL(ngraph::op::ConvolutionBiasBackpropFiltersBias)
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

                template <typename T>
                void AvgPoolLayout(std::shared_ptr<ngraph::Node> node,
                                   vector<memory::desc>& i_mds,
                                   vector<memory::desc>& o_mds)
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
                    memory::dims mkldnn_padding_below(padding_below.begin(), padding_below.end());
                    memory::dims mkldnn_padding_above(padding_above.begin(), padding_above.end());

                    auto input_desc = mkldnn_utils::get_input_mkldnn_md(node.get(), 0);
                    auto result_desc = memory::desc(mkldnn_result_shape, et, memory::format::any);

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
                                                            executor::global_cpu_engine);
                        i_mds.push_back(input_desc);
                        o_mds.push_back(prim_desc.dst_primitive_desc().desc());
                    }
                    catch (const mkldnn::error& e)
                    {
                        throw ngraph_error("MKLDNN Unsupported pooling layout" +
                                           to_string(input_desc.data.format) + e.message);
                    }
                }

                template <>
                void CPULayout::LAYOUT_DECL(ngraph::op::AvgPool)
                {
                    if (mkldnn_utils::use_mkldnn_kernel(node.get()))
                    {
                        vector<memory::desc> i_mds;
                        vector<memory::desc> o_mds;

                        AvgPoolLayout<ngraph::op::AvgPool>(node, i_mds, o_mds);

                        node = insert_input_conversions(external_function, node, i_mds);
                        set_output_layouts(node, o_mds);
                    }
                    else
                    {
                        set_native_layouts(external_function, node);
                    }
                }

                template <>
                void CPULayout::LAYOUT_DECL(ngraph::op::AvgPoolBackprop)
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
                                                                executor::global_cpu_engine);
                            auto prim_desc =
                                pooling_backward::primitive_desc({algorithm_enumerator,
                                                                  result_desc,
                                                                  input_desc,
                                                                  mkldnn_filter_strides,
                                                                  mkldnn_filter_shape,
                                                                  mkldnn_padding_below,
                                                                  mkldnn_padding_above,
                                                                  padding_kind::zero},
                                                                 executor::global_cpu_engine,
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
                                                            executor::global_cpu_engine);
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
                void CPULayout::LAYOUT_DECL(ngraph::op::QuantizedMaxPool)
                {
                    if (mkldnn_utils::use_mkldnn_kernel(node.get()))
                    {
                        vector<memory::desc> i_mds;
                        vector<memory::desc> o_mds;

                        MaxPoolLayout<ngraph::op::QuantizedMaxPool, prop_kind::forward_inference>(
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
                void CPULayout::LAYOUT_DECL(ngraph::op::QuantizedAvgPool)
                {
                    if (mkldnn_utils::use_mkldnn_kernel(node.get()))
                    {
                        vector<memory::desc> i_mds;
                        vector<memory::desc> o_mds;

                        AvgPoolLayout<ngraph::op::QuantizedAvgPool>(node, i_mds, o_mds);

                        node = insert_input_conversions(external_function, node, i_mds);
                        set_output_layouts(node, o_mds);
                    }
                    else
                    {
                        set_native_layouts(external_function, node);
                    }
                }

                template <>
                void CPULayout::LAYOUT_DECL(ngraph::op::MaxPoolWithIndices)
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
                void CPULayout::LAYOUT_DECL(ngraph::op::MaxPool)
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
                                                            executor::global_cpu_engine);

                        auto prim_desc =
                            pooling_backward::primitive_desc({algorithm_enumerator,
                                                              diff_src_desc,
                                                              diff_dst_desc,
                                                              mkldnn_filter_strides,
                                                              mkldnn_filter_shape,
                                                              mkldnn_padding_below,
                                                              mkldnn_padding_above,
                                                              padding_kind::zero},
                                                             executor::global_cpu_engine,
                                                             fwd_prim_desc);
                        i_mds.push_back(fprop_input_md);
                        i_mds.push_back(diff_dst_desc);

                        if (with_indices)
                        {
                            i_mds.push_back(fwd_prim_desc.workspace_primitive_desc().desc());
                        }
                        else if (node->get_input_size() == 3)
                        {
                            i_mds.push_back(diff_dst_desc);
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
                void CPULayout::LAYOUT_DECL(ngraph::op::MaxPoolBackprop)
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
                void CPULayout::LAYOUT_DECL(ngraph::op::MaxPoolWithIndicesBackprop)
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
                void CPULayout::LAYOUT_DECL(ngraph::op::Result)
                {
                    auto result = static_cast<const ngraph::op::Result*>(node.get());
                    auto cpu_tvl = dynamic_pointer_cast<runtime::cpu::LayoutDescriptor>(
                        node->get_inputs()[0].get_output().get_tensor_ptr()->get_tensor_layout());

                    if (result->needs_default_layout() || !cpu_tvl->is_mkldnn_layout() ||
                        cpu_tvl->get_size() * cpu_tvl->get_element_type().size() !=
                            cpu_tvl->get_allocated_size())
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

                static bool can_be_rotated(const ngraph::op::Reshape* reshape,
                                           const mkldnn::memory::desc& md)
                {
                    auto axis_order = reshape->get_input_order();
                    auto input_shape = reshape->get_input_shape(0);
                    auto output_shape = reshape->get_output_shape();
                    if (input_shape.size() != output_shape.size())
                        return false;

                    if ((shape_size(input_shape)) == 1)
                        return false;

                    for (size_t i = 0; i < output_shape.size(); i++)
                    {
                        if (input_shape[axis_order[i]] != output_shape[i])
                            return false;
                    }
                    return true;
                }

                static bool can_be_squeezed(const ngraph::op::Reshape* reshape,
                                            const mkldnn::memory::desc& md,
                                            AxisVector& squeezed_axis)
                {
                    auto input_shape = reshape->get_input_shape(0);
                    auto output_shape = reshape->get_output_shape();

                    if (input_shape.size() <= output_shape.size())
                        return false;

                    if ((shape_size(input_shape)) == 1)
                        return false;

                    for (size_t i = 0, j = 0; i < input_shape.size(); i++)
                    {
                        if (j >= output_shape.size() || input_shape[i] != output_shape[j])
                        {
                            // Squeezed axis
                            if (input_shape[i] != 1)
                                return false;
                            squeezed_axis.push_back(i);
                        }
                        else
                        {
                            j++;
                        }
                    }

                    if (mkldnn_utils::is_mkldnn_padded_layout(md, squeezed_axis))
                    {
                        return false;
                    }

                    return true;
                }

                static bool can_be_expanded(const ngraph::op::Reshape* reshape,
                                            const mkldnn::memory::desc& md,
                                            AxisVector& expanded_axis)
                {
                    auto input_shape = reshape->get_input_shape(0);
                    auto output_shape = reshape->get_output_shape();

                    if (input_shape.size() >= output_shape.size())
                        return false;

                    if ((shape_size(input_shape)) == 1)
                        return false;

                    for (size_t i = 0, j = 0; j < output_shape.size(); j++)
                    {
                        if (i >= input_shape.size() || input_shape[i] != output_shape[j])
                        {
                            // Expanded axis
                            if (output_shape[j] != 1)
                                return false;
                            expanded_axis.push_back(j);
                        }
                        else
                        {
                            i++;
                        }
                    }
                    return true;
                }

                template <>
                void CPULayout::LAYOUT_DECL(ngraph::op::Reshape)
                {
                    auto reshape = static_cast<ngraph::op::Reshape*>(node.get());
                    bool skip_reshape = false;
                    bool skip_input_reorder = false;

                    auto tvl =
                        node->get_inputs()[0].get_output().get_tensor_ptr()->get_tensor_layout();
                    auto cpu_tvl = dynamic_cast<runtime::cpu::LayoutDescriptor*>(tvl.get());
                    if (cpu_tvl && cpu_tvl->is_mkldnn_layout())
                    {
                        auto input_md = mkldnn_utils::get_input_mkldnn_md(node.get(), 0);
                        auto input_shape = reshape->get_input_shape(0);
                        auto output_shape = reshape->get_output_shape();
                        AxisVector squeezed_axis;
                        AxisVector expanded_axis;

                        // Case 1: Transpose only. Rotate layouts
                        // Case 2: Squeeze dims. Removes size-1 dimensions. Squeeze mkldnn layout
                        // Case 3: Exapnd dims. Add size-1 dimensions. Expand mkldnn layout
                        // Default: Convert to row-major if needed
                        if (can_be_rotated(reshape, input_md))
                        {
                            auto output_md = mkldnn_utils::rotate_blocked_md(
                                input_md, reshape->get_input_order());
                            set_output_layouts(node, {output_md});
                            skip_reshape = true;
                            skip_input_reorder = true;
                        }
                        else if (can_be_squeezed(reshape, input_md, squeezed_axis))
                        {
                            auto output_md =
                                mkldnn_utils::squeeze_blocked_md(input_md, squeezed_axis);
                            set_output_layouts(node, {output_md});
                            skip_reshape = true;
                            skip_input_reorder = true;
                        }
                        else if (can_be_expanded(reshape, input_md, expanded_axis))
                        {
                            auto output_md =
                                mkldnn_utils::expand_blocked_md(input_md, expanded_axis);
                            set_output_layouts(node, {output_md});
                            skip_reshape = true;
                            skip_input_reorder = true;
                        }
                        else
                        {
                            if (!reshape->get_is_transpose())
                                skip_reshape = true;
                        }
                    }
                    else
                    {
                        // Input is in row-major layout
                        if (reshape->get_is_transpose())
                        {
                            auto input_strides = cpu_tvl->get_strides();
                            auto axis_order = reshape->get_input_order();
                            Strides output_strides(input_strides.size());
                            for (size_t i = 0; i < input_strides.size(); i++)
                            {
                                output_strides[i] = input_strides[axis_order[i]];
                            }

                            auto output_tvl = dynamic_pointer_cast<runtime::cpu::LayoutDescriptor>(
                                node->get_output_tensor_ptr()->get_tensor_layout());
                            // TODO (jbobba): For now non-MKLDNN layouts are always in row-major format
                            // Enable this once we support non row-major strided formats
                            // output_tvl->set_strides(output_strides);
                        }
                        else
                        {
                            skip_reshape = true;
                        }
                    }

                    if (skip_reshape)
                    {
                        auto op_annotations = reshape->get_op_annotations();
                        if (!op_annotations)
                        {
                            op_annotations =
                                std::make_shared<ngraph::runtime::cpu::CPUOpAnnotations>();
                            reshape->set_op_annotations(op_annotations);
                        }
                        // pass-through
                        op_annotations->add_in_place_oi_pair({0, 0, false});
                    }

                    if (!skip_input_reorder)
                    {
                        set_native_layouts(external_function, node);
                    }
                }

                template <>
                void CPULayout::LAYOUT_DECL(ngraph::op::GetOutputElement)
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
                void CPULayout::LAYOUT_DECL(ngraph::op::LRN)
                {
                    if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node.get()))
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
                void CPULayout::LAYOUT_DECL(ngraph::op::SigmoidBackprop)
                {
                    if (mkldnn_utils::use_mkldnn_kernel(node.get()))
                    {
                        auto input_md = mkldnn_utils::get_input_mkldnn_md(node.get(), 0);
                        vector<memory::desc> i_mds;
                        vector<memory::desc> o_mds;
                        // ensure delta and input have same layout
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
                void CPULayout::LAYOUT_DECL(ngraph::op::ReluBackprop)
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
                    auto input_md = mkldnn_utils::get_input_mkldnn_md(node.get(), 2);
                    auto arg0_md = mkldnn_utils::create_default_mkldnn_md(
                        node.get(), 0, false, memory::format::x);
                    auto arg1_md = mkldnn_utils::create_default_mkldnn_md(
                        node.get(), 1, false, memory::format::x);

                    if (node->get_input_size() == 3)
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
                void CPULayout::LAYOUT_DECL(ngraph::op::BatchNormTraining)
                {
                    if (mkldnn_utils::use_mkldnn_kernel(node.get()))
                    {
                        vector<memory::desc> i_mds;
                        vector<memory::desc> o_mds;
                        BatchNormLayout<ngraph::op::BatchNormTraining>(node, i_mds, o_mds);
                        node = insert_input_conversions(external_function, node, i_mds);
                        set_output_layouts(node, o_mds);
                    }
                    else
                    {
                        set_native_layouts(external_function, node);
                    }
                }

                template <>
                void CPULayout::LAYOUT_DECL(ngraph::op::BatchNormInference)
                {
                    if (mkldnn_utils::use_mkldnn_kernel(node.get()))
                    {
                        vector<memory::desc> i_mds;
                        vector<memory::desc> o_mds;
                        BatchNormLayout<ngraph::op::BatchNormInference>(node, i_mds, o_mds);
                        node = insert_input_conversions(external_function, node, i_mds);
                        set_output_layouts(node, o_mds);
                    }
                    else
                    {
                        set_native_layouts(external_function, node);
                    }
                }

                template <>
                void CPULayout::LAYOUT_DECL(ngraph::op::BatchNormTrainingRelu)
                {
                    if (mkldnn_utils::use_mkldnn_kernel(node.get()))
                    {
                        vector<memory::desc> i_mds;
                        vector<memory::desc> o_mds;
                        BatchNormLayout<ngraph::op::BatchNormTrainingRelu>(node, i_mds, o_mds);
                        node = insert_input_conversions(external_function, node, i_mds);
                        set_output_layouts(node, o_mds);
                    }
                    else
                    {
                        throw ngraph_error("BatchnormRelu only supported in MKLDNN for now");
                    }
                }

                template <>
                void CPULayout::LAYOUT_DECL(ngraph::op::BatchNormInferenceRelu)
                {
                    if (mkldnn_utils::use_mkldnn_kernel(node.get()))
                    {
                        vector<memory::desc> i_mds;
                        vector<memory::desc> o_mds;
                        BatchNormLayout<ngraph::op::BatchNormInferenceRelu>(node, i_mds, o_mds);
                        node = insert_input_conversions(external_function, node, i_mds);
                        set_output_layouts(node, o_mds);
                    }
                    else
                    {
                        throw ngraph_error("BatchnormRelu only supported in MKLDNN for now");
                    }
                }

                template <>
                void CPULayout::LAYOUT_DECL(ngraph::op::BatchNormTrainingBackprop)
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
                void CPULayout::LAYOUT_DECL(ngraph::op::Slice)
                {
                    if (mkldnn_utils::use_mkldnn_kernel(node.get()))
                    {
                        const ngraph::op::Slice* slice =
                            static_cast<const ngraph::op::Slice*>(node.get());
                        auto lower_bounds = slice->get_lower_bounds();
                        auto result_shape = slice->get_output_shape(0);

                        auto input_md = mkldnn_utils::get_input_mkldnn_md(node.get(), 0);
                        NGRAPH_DEBUG << "input memory format: " << input_md.data.format << "\n";
                        auto result_format =
                            static_cast<mkldnn::memory::format>(input_md.data.format);

                        // check lower bounds and output shape
                        for (auto i = 0; i < input_md.data.ndims; i++)
                        {
                            auto block_size = input_md.data.layout_desc.blocking.block_dims[i];
                            if (block_size != 0 && (lower_bounds[i] % block_size != 0 ||
                                                    result_shape[i] % block_size != 0))
                            {
                                NGRAPH_DEBUG << "slice: number of channels in lower bounds or "
                                                "output shape is not multiple of block size, "
                                                "set native layout\n";
                                set_native_layouts(external_function, node);
                                return;
                            }
                        }

                        if (result_format == mkldnn::memory::blocked)
                        {
                            set_native_layouts(external_function, node);
                        }
                        else
                        {
                            vector<memory::desc> o_mds;
                            auto result_desc = mkldnn_utils::create_default_mkldnn_md(
                                node.get(), 0, true, result_format);
                            o_mds.push_back(result_desc);
                            set_output_layouts(node, o_mds);
                        }
                    }
                    else
                    {
                        set_native_layouts(external_function, node);
                    }
                }

                template <>
                void CPULayout::LAYOUT_DECL(ngraph::op::Concat)
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
                                input_md, executor::global_cpu_engine));
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
                void CPULayout::LAYOUT_DECL(ngraph::op::Lstm)
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
                void CPULayout::LAYOUT_DECL(ngraph::op::Rnn)
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
                void CPULayout::LAYOUT_DECL(ngraph::op::Softmax)
                {
                    // Softmax cannot use the default unary layout method since the kernels
                    // need to know the reduction dimension
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

static const runtime::cpu::pass::LayoutOpMap s_dispatcher{
    {TI(ngraph::op::Concat), &runtime::cpu::pass::CPULayout::layout<ngraph::op::Concat>},
    {TI(ngraph::op::AvgPool), &runtime::cpu::pass::CPULayout::layout<ngraph::op::AvgPool>},
    {TI(ngraph::op::AvgPoolBackprop),
     &runtime::cpu::pass::CPULayout::layout<ngraph::op::AvgPoolBackprop>},
    {TI(ngraph::op::QuantizedConvolution),
     &runtime::cpu::pass::CPULayout::layout<ngraph::op::QuantizedConvolution>},
    {TI(ngraph::op::Convolution), &runtime::cpu::pass::CPULayout::layout<ngraph::op::Convolution>},
    {TI(ngraph::op::GroupConvolution),
     &runtime::cpu::pass::CPULayout::layout<ngraph::op::GroupConvolution>},
    {TI(ngraph::op::ConvolutionBackpropData),
     &runtime::cpu::pass::CPULayout::layout<ngraph::op::ConvolutionBackpropData>},
    {TI(ngraph::op::ConvolutionBackpropFilters),
     &runtime::cpu::pass::CPULayout::layout<ngraph::op::ConvolutionBackpropFilters>},
    {TI(ngraph::op::MaxPool), &runtime::cpu::pass::CPULayout::layout<ngraph::op::MaxPool>},
    {TI(ngraph::op::QuantizedMaxPool),
     &runtime::cpu::pass::CPULayout::layout<ngraph::op::QuantizedMaxPool>},
    {TI(ngraph::op::QuantizedAvgPool),
     &runtime::cpu::pass::CPULayout::layout<ngraph::op::QuantizedAvgPool>},
    {TI(ngraph::op::MaxPoolWithIndices),
     &runtime::cpu::pass::CPULayout::layout<ngraph::op::MaxPoolWithIndices>},
    {TI(ngraph::op::MaxPoolBackprop),
     &runtime::cpu::pass::CPULayout::layout<ngraph::op::MaxPoolBackprop>},
    {TI(ngraph::op::MaxPoolWithIndicesBackprop),
     &runtime::cpu::pass::CPULayout::layout<ngraph::op::MaxPoolWithIndicesBackprop>},
    {TI(ngraph::op::ConvolutionBias),
     &runtime::cpu::pass::CPULayout::layout<ngraph::op::ConvolutionBias>},
    {TI(ngraph::op::ConvolutionRelu),
     &runtime::cpu::pass::CPULayout::layout<ngraph::op::ConvolutionRelu>},
    {TI(ngraph::op::ConvolutionBiasAdd),
     &runtime::cpu::pass::CPULayout::layout<ngraph::op::ConvolutionBiasAdd>},
    {TI(ngraph::op::ConvolutionBiasBackpropFiltersBias),
     &runtime::cpu::pass::CPULayout::layout<ngraph::op::ConvolutionBiasBackpropFiltersBias>},
    {TI(ngraph::op::BatchNormTraining),
     &runtime::cpu::pass::CPULayout::layout<ngraph::op::BatchNormTraining>},
    {TI(ngraph::op::BatchNormInference),
     &runtime::cpu::pass::CPULayout::layout<ngraph::op::BatchNormInference>},
    {TI(ngraph::op::BatchNormInferenceRelu),
     &runtime::cpu::pass::CPULayout::layout<ngraph::op::BatchNormInferenceRelu>},
    {TI(ngraph::op::BatchNormTrainingRelu),
     &runtime::cpu::pass::CPULayout::layout<ngraph::op::BatchNormTrainingRelu>},
    {TI(ngraph::op::BatchNormTrainingBackprop),
     &runtime::cpu::pass::CPULayout::layout<ngraph::op::BatchNormTrainingBackprop>},
    {TI(ngraph::op::GetOutputElement),
     &runtime::cpu::pass::CPULayout::layout<ngraph::op::GetOutputElement>},
    {TI(ngraph::op::LRN), &runtime::cpu::pass::CPULayout::layout<ngraph::op::LRN>},
    {TI(ngraph::op::Reshape), &runtime::cpu::pass::CPULayout::layout<ngraph::op::Reshape>},
    {TI(ngraph::op::Result), &runtime::cpu::pass::CPULayout::layout<ngraph::op::Result>},
    {TI(ngraph::op::ReluBackprop),
     &runtime::cpu::pass::CPULayout::layout<ngraph::op::ReluBackprop>},
    {TI(ngraph::op::SigmoidBackprop),
     &runtime::cpu::pass::CPULayout::layout<ngraph::op::SigmoidBackprop>},
    {TI(ngraph::op::Lstm), &runtime::cpu::pass::CPULayout::layout<ngraph::op::Lstm>},
    {TI(ngraph::op::Rnn), &runtime::cpu::pass::CPULayout::layout<ngraph::op::Rnn>},
    {TI(ngraph::op::Softmax), &runtime::cpu::pass::CPULayout::layout<ngraph::op::Softmax>},
    {TI(ngraph::op::ConvolutionAdd),
     &runtime::cpu::pass::CPULayout::layout<ngraph::op::ConvolutionAdd>},
    {TI(ngraph::op::Slice), &runtime::cpu::pass::CPULayout::layout<ngraph::op::Slice>},
    {TI(ngraph::op::QuantizedConvolutionRelu),
     &runtime::cpu::pass::CPULayout::layout<ngraph::op::QuantizedConvolutionRelu>},
    {TI(ngraph::op::QuantizedConvolutionBias),
     &runtime::cpu::pass::CPULayout::layout<ngraph::op::QuantizedConvolutionBias>},
    {TI(ngraph::op::QuantizedConvolutionBiasAdd),
     &runtime::cpu::pass::CPULayout::layout<ngraph::op::QuantizedConvolutionBiasAdd>},
    {TI(ngraph::op::QuantizedConvolutionBiasSignedAdd),
     &runtime::cpu::pass::CPULayout::layout<ngraph::op::QuantizedConvolutionBiasSignedAdd>},
    {TI(ngraph::op::GroupConvolutionBias),
     &runtime::cpu::pass::CPULayout::layout<ngraph::op::GroupConvolutionBias>},
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
        else if (dynamic_pointer_cast<ngraph::op::util::UnaryElementwiseArithmetic>(node) !=
                 nullptr)
        {
            set_layouts_unaryeltwise(m_external_function, node);
        }
        else if (dynamic_pointer_cast<ngraph::op::util::BinaryElementwiseArithmetic>(node) !=
                 nullptr)
        {
            set_layouts_binaryeltwise(m_external_function, node);
        }
        else
        {
            set_native_layouts(m_external_function, node);
        }
    }

    return false;
}
