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

#include "cpu_cse.hpp"
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
    return false;
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
