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

#include "ngraph/runtime/gpu/pass/gpu_assignment.hpp"
#include <algorithm>
#include <cassert>
#include <memory>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>

#include "ngraph/descriptor/output.hpp"
#include "ngraph/op/replace_slice.hpp"
#include "ngraph/runtime/gpu/gpu_op_annotations.hpp"

using namespace std;
using namespace ngraph;

namespace ngraph
{
    namespace runtime
    {
        namespace gpu
        {
            namespace pass
            {
                template <>
                void GPUAssignment::ASSIGN_DECL(ngraph::op::ReplaceSlice)
                {
                    auto rep_slice = static_cast<ngraph::op::ReplaceSlice*>(node);

                    auto op_annotations = rep_slice->get_op_annotations();
                    if (op_annotations)
                    {
                        // pass-through
                        op_annotations->add_in_place_oi_pair({0, 0, false});
                    }
                    else
                    {
                        op_annotations = std::make_shared<ngraph::runtime::gpu::GPUOpAnnotations>();
                        // pass-through
                        op_annotations->add_in_place_oi_pair({0, 0, false});
                        rep_slice->set_op_annotations(op_annotations);
                    }
                }
            }
        }
    }
}

#define TI(x) type_index(typeid(x))

static const runtime::gpu::pass::AssignOpMap s_dispatcher{
    {TI(ngraph::op::ReplaceSlice),
     &runtime::gpu::pass::GPUAssignment::assign<ngraph::op::ReplaceSlice>},
};

bool runtime::gpu::pass::GPUAssignment::run_on_call_graph(
    const std::list<std::shared_ptr<Node>>& nodes)
{
    for (const auto& node : nodes)
    {
        auto& n = *node;
        auto handler = s_dispatcher.find(TI(n));
        if (handler != s_dispatcher.end())
        {
            handler->second(m_external_function, node.get());
        }
    }

    return false;
}
