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

#include <algorithm>
#include <iostream>
#include <map>
#include <memory>

#include "ngraph/op/reshape.hpp"
#include "ngraph/runtime/cpu/mkldnn_utils.hpp"
#include "ngraph/runtime/cpu/op/convert_layout.hpp"

#include "cpu_shuffle_folding.hpp"

static const std::map<const ngraph::AxisVector, const mkldnn::memory::format>
    input_order_format_map{{ngraph::AxisVector{3, 2, 0, 1}, mkldnn::memory::format::hwio}};

bool ngraph::runtime::cpu::pass::CPUShuffleFolding::run_on_function(
    std::shared_ptr<ngraph::Function> function)
{
    bool clobbered = false;

    for (const auto& n : function->get_ordered_ops())
    {
        auto convert_layout = std::dynamic_pointer_cast<op::ConvertLayout>(n);

        if (convert_layout)
        {
            auto reshape = std::dynamic_pointer_cast<ngraph::op::Reshape>(n->get_argument(0));
            if (reshape)
            {
                auto output_shape = reshape->get_output_shape();
                auto input_shape = reshape->get_input_shape(0);

                if (output_shape.size() != input_shape.size())
                {
                    continue;
                }

                size_t j = 0;
                bool is_shuffle = true;
                for (auto i : reshape->get_input_order())
                {
                    if (input_shape.at(i) != output_shape.at(j++))
                    {
                        is_shuffle = false;
                        break;
                    }
                }

                if (!is_shuffle)
                {
                    continue;
                }

                auto reshape_input_layout =
                    reshape->get_argument(0)->get_output_tensor_view()->get_tensor_view_layout();
                auto output_layout =
                    convert_layout->get_output_tensor_view()->get_tensor_view_layout();

                if (reshape_input_layout)
                {
                    auto reshape_input_layout_descriptor =
                        std::static_pointer_cast<runtime::cpu::LayoutDescriptor>(
                            reshape_input_layout);
                    auto reshape_input_format =
                        reshape_input_layout_descriptor->get_mkldnn_format();
                    auto output_format =
                        std::static_pointer_cast<runtime::cpu::LayoutDescriptor>(output_layout)
                            ->get_mkldnn_format();

                    if (mkldnn_utils::is_mkldnn_filter_format(output_format) &&
                        output_format == mkldnn::memory::format::OIhw16i16o &&
                        reshape_input_format == mkldnn::memory::format::nchw)
                    {
                        if (input_order_format_map.find(reshape->get_input_order()) !=
                            input_order_format_map.end())
                        {
                            reshape_input_layout_descriptor->set_mkldnn_format(
                                input_order_format_map.at(reshape->get_input_order()));
                            reshape_input_layout_descriptor->set_axis_order(
                                reshape->get_input_order());
                            function->replace_node(reshape, reshape->get_argument(0));
                        }
                    }
                }
            }
        }
    }

    return clobbered;
}
