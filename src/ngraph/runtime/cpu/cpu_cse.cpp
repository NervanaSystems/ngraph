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

#include "cpu_cse.hpp"
#include "ngraph/log.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/runtime/cpu/cpu_layout_descriptor.hpp"
#include "ngraph/runtime/cpu/cpu_op_annotations.hpp"
#include "ngraph/runtime/cpu/mkldnn_utils.hpp"
#include "ngraph/runtime/cpu/op/convert_layout.hpp"

using namespace mkldnn;
using namespace ngraph;
using namespace std;

#define TI(x) std::type_index(typeid(x))

static bool cse_convertlayout(std::shared_ptr<Node> a, std::shared_ptr<Node> b)
{
    NGRAPH_DEBUG << "In cse_convertlayout for " << a->get_name() << " and " << b->get_name();

    auto ar_a = std::static_pointer_cast<runtime::cpu::op::ConvertLayout>(a);
    auto ar_b = std::static_pointer_cast<runtime::cpu::op::ConvertLayout>(b);

    // gets the tensor layout from the given node
    auto get_tensor_layout = [](std::shared_ptr<Node> node) {
        auto tensor = node->get_output_tensor_ptr();
        auto cpu_tl =
            static_cast<ngraph::runtime::cpu::LayoutDescriptor*>(tensor->get_tensor_layout().get());
        return cpu_tl;
    };

    auto a_layout_desc = get_tensor_layout(a);
    auto b_layout_desc = get_tensor_layout(b);
    bool is_args_same = (ar_a->get_argument(0) == ar_b->get_argument(0));
    bool is_output_mem_desc_same = runtime::cpu::mkldnn_utils::compare_mkldnn_mds(
        a_layout_desc->get_mkldnn_md(), b_layout_desc->get_mkldnn_md());

    return is_args_same && is_output_mem_desc_same;
}

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            const std::unordered_map<
                std::type_index,
                std::function<bool(std::shared_ptr<Node>, std::shared_ptr<Node>)>>&
                get_cse_handlers_map()
            {
                const static std::unordered_map<
                    std::type_index,
                    std::function<bool(std::shared_ptr<Node>, std::shared_ptr<Node>)>>
                    cse_map{{TI(runtime::cpu::op::ConvertLayout), cse_convertlayout}};
                return cse_map;
            }
        }
    }
}
