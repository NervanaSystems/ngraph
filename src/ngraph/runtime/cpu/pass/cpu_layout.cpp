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

#include <mkldnn.hpp>

#include "cpu_layout.hpp"
#include "ngraph/descriptor/output.hpp"
#include "ngraph/runtime/cpu/cpu_layout_descriptor.hpp"
#include "ngraph/runtime/cpu/mkldnn_utils.hpp"

//using namespace ngraph::runtime::cpu::pass;
using namespace ngraph;

bool runtime::cpu::pass::CPULayout::run_on_call_graph(const std::list<std::shared_ptr<Node>>& nodes)
{
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
