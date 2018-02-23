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
#include "ngraph/ops/convolution.hpp"
#include "ngraph/ops/op.hpp"
#include "ngraph/runtime/cpu/cpu_layout_descriptor.hpp"
#include "ngraph/runtime/cpu/cpu_op_annotations.hpp"
#include "ngraph/runtime/cpu/mkldnn_utils.hpp"
#include "ngraph/runtime/cpu/ops/convert_layout.hpp"

using namespace std;
using namespace mkldnn;
using namespace ngraph;

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
            cpu_tvl->get_mkldnn_format() !=
                runtime::cpu::mkldnn_utils::CreateNativeDataFormat(*cpu_tvl))
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
            new_args.push_back(node->get_input_op(index));
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
                template <>
                void CPULayout::LAYOUT_DECL(ngraph::op::Convolution)
                {
                    auto op_annotations =
                        static_pointer_cast<ngraph::op::Op>(node)->get_op_annotations();
                    if (op_annotations &&
                        static_pointer_cast<ngraph::runtime::cpu::CPUOpAnnotations>(op_annotations)
                            ->is_mkldnn_op())
                    {
                        auto convolution = static_cast<const ngraph::op::Convolution*>(node.get());

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
                        memory::dims mkldnn_dilated_strides(
                            window_dilation_strides_adjusted.begin(),
                            window_dilation_strides_adjusted.end());
                        memory::dims mkldnn_padding_below(padding_below.begin(),
                                                          padding_below.end());
                        memory::dims mkldnn_padding_above(padding_above.begin(),
                                                          padding_above.end());
                        const memory::desc input_data_desc(
                            mkldnn_arg0_shape, et, memory::format::any);
                        const memory::desc weights_desc(mkldnn_arg1_shape, et, memory::format::any);
                        const memory::desc result_desc(
                            mkldnn_result_shape, et, memory::format::any);
                        convolution_forward::desc fwd_desc(prop_kind::forward,
                                                           algorithm::convolution_direct,
                                                           input_data_desc,
                                                           weights_desc,
                                                           result_desc,
                                                           mkldnn_filter_strides,
                                                           mkldnn_dilated_strides,
                                                           mkldnn_padding_below,
                                                           mkldnn_padding_above,
                                                           padding_kind::zero);
                        convolution_forward::primitive_desc prim_desc(fwd_desc, cpu_engine);
                        memory::format prim_input_formats[2];
                        memory::format prim_output_formats[1];
                        prim_input_formats[0] = static_cast<memory::format>(
                            prim_desc.src_primitive_desc().desc().data.format);
                        prim_output_formats[0] = static_cast<memory::format>(
                            prim_desc.dst_primitive_desc().desc().data.format);
                        prim_input_formats[1] = static_cast<memory::format>(
                            prim_desc.weights_primitive_desc().desc().data.format);

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
                            auto mkldnn_tvl =
                                dynamic_cast<runtime::cpu::LayoutDescriptor*>(tvl.get());
                            if (!mkldnn_tvl ||
                                mkldnn_tvl->get_mkldnn_format() != prim_input_formats[index])
                            {
                                auto native_axis_order = ngraph::runtime::cpu::LayoutDescriptor::
                                    create_native_axis_order(rank);
                                auto layout =
                                    std::make_shared<ngraph::runtime::cpu::LayoutDescriptor>(
                                        *tv, native_axis_order);
                                layout->set_mkldnn_format(prim_input_formats[index]);
                                auto new_node =
                                    std::shared_ptr<Node>(new runtime::cpu::op::ConvertLayout(
                                        output.get_node(), output.get_index(), layout));
                                new_args.push_back(new_node);
                                replace_node = true;
                                NGRAPH_DEBUG << "Inserted conversion node " << new_node->get_name()
                                             << " between " << output.get_node()->get_name()
                                             << "(layout: " << mkldnn_tvl->get_mkldnn_format()
                                             << ") and " << node->get_name()
                                             << "(layout: " << prim_input_formats[index] << ")";
                            }
                            else
                            {
                                new_args.push_back(node->get_input_op(index));
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
                            NGRAPH_DEBUG << "Replaced " << node->get_name() << " with "
                                         << new_node->get_name();
                            auto old_op_annotations =
                                static_pointer_cast<ngraph::op::Op>(node)->get_op_annotations();
                            static_pointer_cast<ngraph::op::Op>(new_node)->set_op_annotations(
                                old_op_annotations);
                            node = new_node;
                        }

                        // Set convolution output format
                        for (size_t i = 0; i < node->get_output_size(); ++i)
                        {
                            auto tv = node->get_output_tensor_view(i);
                            auto tvt = tv->get_tensor_view_type();
                            auto rank = tvt->get_shape().size();

                            auto tvl = tv->get_tensor_view_layout();
                            if (tvl)
                            {
                                throw ngraph_error("Convolution output layout already set");
                            }

                            auto native_axis_order =
                                ngraph::runtime::cpu::LayoutDescriptor::create_native_axis_order(
                                    rank);

                            auto layout = std::make_shared<ngraph::runtime::cpu::LayoutDescriptor>(
                                *tv, native_axis_order);

                            layout->set_mkldnn_format(prim_output_formats[i]);
                            tv->set_tensor_view_layout(layout);
                            NGRAPH_DEBUG << "Setting Node: " << node->get_name()
                                         << " output layout: " << prim_output_formats[i] << endl;
                        }
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
    {TI(ngraph::op::Convolution), &runtime::cpu::pass::CPULayout::layout<ngraph::op::Convolution>},
};

bool runtime::cpu::pass::CPULayout::run_on_call_graph(const std::list<std::shared_ptr<Node>>& nodes)
{
    for (const auto& node : nodes)
    {
        auto& n = *node;
        auto handler = s_dispatcher.find(TI(n));
        if (handler != s_dispatcher.end())
        {
            handler->second(m_external_function.get(), node);
        }
        else
        {
            set_default_layouts(m_external_function.get(), node);
        }
    }

    return false;
}
