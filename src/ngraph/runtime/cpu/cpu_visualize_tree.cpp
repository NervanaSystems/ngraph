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

#include <string>

#include "ngraph/log.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/runtime/cpu/cpu_layout_descriptor.hpp"
#include "ngraph/runtime/cpu/cpu_op_annotations.hpp"
#include "ngraph/runtime/cpu/cpu_visualize_tree.hpp"
#include "ngraph/runtime/cpu/mkldnn_utils.hpp"
#include "ngraph/runtime/cpu/op/convert_layout.hpp"

using namespace mkldnn;
using namespace ngraph;
using namespace std;

static void visualize_layout_format(const Node& node, ostream& ss)
{
    try
    {
        auto input_desc = node.input(0).get_tensor().get_tensor_layout();
        auto result_desc = node.get_output_tensor_ptr()->get_tensor_layout();

        auto in_tvl = static_pointer_cast<runtime::cpu::LayoutDescriptor>(input_desc);
        auto out_tvl = static_pointer_cast<runtime::cpu::LayoutDescriptor>(result_desc);

        if (!in_tvl || !out_tvl)
        {
            return;
        }
        if (!in_tvl->is_mkldnn_layout() || !out_tvl->is_mkldnn_layout())
        {
            return;
        }
        if (auto reshape = dynamic_cast<const op::Reshape*>(&node))
        {
            ss << "\ninput_order=" << reshape->get_input_order();
        }
#if MKLDNN_VERSION_MAJOR >= 1
        auto in_md = in_tvl->get_mkldnn_md();
        auto out_md = out_tvl->get_mkldnn_md();
        ss << "\nin strides={";
        for (auto i = 0; i < in_md.data.ndims - 1; i++)
        {
            ss << in_md.data.format_desc.blocking.strides[i] << ",";
        }
        ss << in_md.data.format_desc.blocking.strides[in_md.data.ndims - 1] << "}";
        ss << "\nout strides={";
        for (auto i = 0; i < out_md.data.ndims - 1; i++)
        {
            ss << out_md.data.format_desc.blocking.strides[i] << ",";
        }
        ss << out_md.data.format_desc.blocking.strides[out_md.data.ndims - 1] << "}";
#else
        ss << "\nin=" << runtime::cpu::mkldnn_utils::get_mkldnn_format_string(
                             static_cast<mkldnn::memory::FORMAT_KIND>(
                                 in_tvl->get_mkldnn_md().data.FORMAT_KIND));
        ss << " out=" << runtime::cpu::mkldnn_utils::get_mkldnn_format_string(
                             static_cast<mkldnn::memory::FORMAT_KIND>(
                                 out_tvl->get_mkldnn_md().data.FORMAT_KIND));
#endif
        ss << " ";
    }
    catch (...)
    {
        NGRAPH_DEBUG << "Exception in visualize_reshape \n";
    }
}

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            const visualize_tree_ops_map_t& get_visualize_tree_ops_map()
            {
                const static visualize_tree_ops_map_t vtom{
                    {runtime::cpu::op::ConvertLayout::type_info, visualize_layout_format},
                    {ngraph::op::Reshape::type_info, visualize_layout_format}};
                return vtom;
            }
        }
    }
}
