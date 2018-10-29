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

#include "cpu_visualize_tree.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/runtime/cpu/cpu_layout_descriptor.hpp"
#include "ngraph/runtime/cpu/cpu_op_annotations.hpp"
#include "ngraph/runtime/cpu/mkldnn_utils.hpp"
#include "ngraph/runtime/cpu/op/convert_layout.hpp"

using namespace mkldnn;
using namespace ngraph;
using namespace std;

#define TI(x) std::type_index(typeid(x))

static void visualize_convert_layout(const Node& node, ostream& ss)
{
    auto input_desc = runtime::cpu::mkldnn_utils::get_input_mkldnn_md(&node, 0);
    auto result_desc = runtime::cpu::mkldnn_utils::get_output_mkldnn_md(&node, 0);

    ss << "in=" << runtime::cpu::mkldnn_utils::get_mkldnn_format_string(
                       static_cast<mkldnn::memory::format>(input_desc.data.format));
    ss << " out=" << runtime::cpu::mkldnn_utils::get_mkldnn_format_string(
                         static_cast<mkldnn::memory::format>(result_desc.data.format));
    ss << " ";
}

static void visualize_reshape(const Node& node, ostream& ss)
{
    try
    {
        auto input_desc = node.get_inputs().at(0).get_tensor().get_tensor_layout();
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
        ss << "\nin="
           << runtime::cpu::mkldnn_utils::get_mkldnn_format_string(
                  static_cast<mkldnn::memory::format>(in_tvl->get_mkldnn_md().data.format));
        ss << " out="
           << runtime::cpu::mkldnn_utils::get_mkldnn_format_string(
                  static_cast<mkldnn::memory::format>(out_tvl->get_mkldnn_md().data.format));
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
                    {TI(runtime::cpu::op::ConvertLayout), visualize_convert_layout},
                    {TI(ngraph::op::Reshape), visualize_reshape}};
                return vtom;
            }
        }
    }
}
