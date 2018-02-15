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
#include <typeindex>
#include <typeinfo>

#include <mkldnn.hpp>

#include "cpu_layout.hpp"
#include "ngraph/descriptor/output.hpp"
#include "ngraph/ops/convolution.hpp"
#include "ngraph/runtime/cpu/cpu_layout_descriptor.hpp"
#include "ngraph/runtime/cpu/mkldnn_utils.hpp"

using namespace std;
using namespace mkldnn;
using namespace ngraph;

#define TI(x) type_index(typeid(x))

static const runtime::cpu::pass::LayoutOpMap dispatcher{
    {TI(ngraph::op::Convolution), &runtime::cpu::pass::CPULayout::LayoutConvolution},
};

bool runtime::cpu::pass::CPULayout::run_on_call_graph(const std::list<std::shared_ptr<Node>>& nodes)
{
    for (const auto& node : nodes)
    {
        auto& n = *node;
        auto handler = dispatcher.find(TI(n));
        if (handler != dispatcher.end())
        {
            handler->second(m_external_function.get(), node.get());
        }
    }

    for (const auto& node : nodes)
    {
        for (size_t i = 0; i < node->get_output_size(); ++i)
        {
            auto tv = node->get_output_tensor_view(i);
            if (tv->get_tensor_view_layout())
            {
                continue;
            }

            auto tvt = tv->get_tensor_view_type();
            auto& tensor = tv->get_tensor();
            auto rank = tvt->get_shape().size();

            auto native_axis_order =
                ngraph::runtime::cpu::LayoutDescriptor::create_native_axis_order(rank);

            auto layout =
                std::make_shared<ngraph::runtime::cpu::LayoutDescriptor>(*tv, native_axis_order);

            if (tensor.is_output() || tensor.is_input() || tensor.is_constant())
            {
                // Set the MKLDNN format to native row-major variants
                layout->set_mkldnn_format(mkldnn_utils::CreateNativeDataFormat(*layout));
            }
            else
            {
                if (ngraph::runtime::cpu::mkldnn_utils::IsMKLDNNOp(*node))
                {
                    // TODO(jmenon): get_inputs is marked as to-be-deprecated
                    // but get_input_ops isn't a suitable API so this needs to be
                    // reworked
                    for (const descriptor::Input& input : node->get_inputs())
                    {
                        const auto& output = input.get_output();
                        auto output_tv = output.get_tensor_view();
                        auto output_tvl = output_tv->get_tensor_view_layout();

                        // TODO(jmenon): Propagate layout based on inputs
                        // TODO(jmenon): Insert layout conversions when needed
                    }
                }
                else
                {
                    layout->set_mkldnn_format(mkldnn::memory::format::format_undef);
                }
            }
            tv->set_tensor_view_layout(layout);
        }
    }

    return false;
}

void runtime::cpu::pass::CPULayout::LAYOUT_DECL(LayoutConvolution)
{
    if (external_function->get_op_annotations(node)->is_mkldnn_op)
    {
        auto convolution = static_cast<const op::Convolution*>(node);

        auto arg0_shape = node->get_input_shape(0);
        auto arg1_shape = node->get_input_shape(1);
        auto result_shape = node->get_output_shape(0);
        auto arg0_rank = arg0_shape.size();
        auto arg1_rank = arg1_shape.size();
        auto filter_strides = convolution->get_window_movement_strides();
        auto padding_below = convolution->get_padding_below();
        auto padding_above = convolution->get_padding_above();

        Strides window_dilation_strides_adjusted;

        for (size_t s : convolution->get_window_dilation_strides())
        {
            window_dilation_strides_adjusted.push_back(s - 1);
        }

        memory::data_type et = runtime::cpu::mkldnn_utils::get_mkldnn_data_type(
            node->get_input_element_type(0).c_type_string());

        engine cpu_engine(engine::cpu, 0);
        memory::dims mkldnn_arg0_shape(arg0_shape.begin(), arg0_shape.end());
        memory::dims mkldnn_arg1_shape(arg1_shape.begin(), arg1_shape.end());
        memory::dims mkldnn_result_shape(result_shape.begin(), result_shape.end());
        memory::dims mkldnn_filter_strides(filter_strides.begin(), filter_strides.end());
        memory::dims mkldnn_dilated_strides(window_dilation_strides_adjusted.begin(),
                                            window_dilation_strides_adjusted.end());
        memory::dims mkldnn_padding_below(padding_below.begin(), padding_below.end());
        memory::dims mkldnn_padding_above(padding_above.begin(), padding_above.end());
        const memory::desc input_data_desc(mkldnn_arg0_shape, et, memory::format::any);
        const memory::desc weights_desc(mkldnn_arg1_shape, et, memory::format::any);
        const memory::desc result_desc(mkldnn_result_shape, et, memory::format::any);
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
        mkldnn_memory_format_t prim_src_format = prim_desc.src_primitive_desc().desc().data.format;
        mkldnn_memory_format_t prim_dst_format = prim_desc.dst_primitive_desc().desc().data.format;
        mkldnn_memory_format_t prim_weights_format =
            prim_desc.weights_primitive_desc().desc().data.format;
        cout << "Convolution Preferred layout src: " << prim_src_format
             << " weights: " << prim_weights_format << " dst: " << prim_dst_format << endl;
    }
}