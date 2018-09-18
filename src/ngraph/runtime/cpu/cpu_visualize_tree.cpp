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

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            const visualize_tree_ops_map_t& get_visualize_tree_ops_map()
            {
                const static visualize_tree_ops_map_t vtom{
                    {TI(runtime::cpu::op::ConvertLayout), visualize_convert_layout}};
                return vtom;
            }
        }
    }
}
