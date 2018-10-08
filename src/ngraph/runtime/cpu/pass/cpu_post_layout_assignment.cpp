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

#include "ngraph/runtime/cpu/pass/cpu_post_layout_assignment.hpp"
#include <typeindex>
#include <typeinfo>

#include "ngraph/descriptor/output.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/runtime/cpu/cpu_op_annotations.hpp"

using namespace std;
using namespace ngraph;

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            namespace pass
            {
                template <>
                void CPUPostLayoutAssignment::ASSIGN_DECL(ngraph::op::Concat)
                {
                    auto concat = static_cast<op::Concat*>(node);
                    auto shape = concat->get_input_shape(0);
                    auto axis = concat->get_concatenation_axis();
                    auto product = 1;
                    for (int i = 0; i < axis; i++)
                    {
                        product *= shape[i];
                    }
                    if (product != 1)
                    {
                        NGRAPH_DEBUG << "cpu_post_layout_assignment: The product of Concat's shape "
                                        "before concat axis is not 1, no in place concat";
                        return;
                    }

                    bool in_place_concat = false;

                    for (descriptor::Input& input : concat->get_inputs())
                    {
                        if (shape_size(input.get_shape()) == 0)
                        {
                            NGRAPH_DEBUG << "cpu_post_layout_assignment: 0 length tensor, no in "
                                            "place concat";
                            return;
                        }
                        const auto& output = input.get_output();
                        auto arg = output.get_node();
                        if (std::dynamic_pointer_cast<op::Constant>(arg) ||
                            std::dynamic_pointer_cast<op::Parameter>(arg))
                        {
                            NGRAPH_DEBUG << "cpu_post_layout_assignment: " << arg->get_name()
                                         << ": constant or parameter, no in place concat";
                            return;
                        }
                        else if (output.get_inputs().size() != 1)
                        {
                            // check if we can do in place concat
                            auto concat_count = 0;
                            for (auto output_input : output.get_inputs())
                            {
                                auto user = output_input->get_node();
                                if (std::dynamic_pointer_cast<op::Concat>(user))
                                {
                                    concat_count++;
                                    if (concat_count == 2)
                                    {
                                        NGRAPH_DEBUG << "cpu_post_layout_assignment: multiple "
                                                        "concat users, no in place concat";
                                        return;
                                    }
                                }
                            }

                            std::unordered_set<Node*> visited;
                            std::deque<Node*> stack;
                            stack.push_front(arg.get());

                            while (stack.size() > 0)
                            {
                                ngraph::Node* curr = stack.front();
                                visited.insert(curr);
                                if (curr->is_output())
                                {
                                    NGRAPH_DEBUG << "cpu_post_layout_assignment: not post "
                                                    "dominated, no in place concat";
                                    return;
                                }
                                else
                                {
                                    if (auto op = dynamic_cast<op::Op*>(curr))
                                    {
                                        if (auto op_annotations = op->get_op_annotations())
                                        {
                                            for (auto oi_pair :
                                                 op_annotations->get_in_place_oi_pairs())
                                            {
                                                if (oi_pair.destructive)
                                                {
                                                    NGRAPH_DEBUG << "cpu_post_layout_assignment: "
                                                                    "destructive in place oi, no "
                                                                    "in place concat";
                                                    return;
                                                }
                                            }
                                        }
                                    }
                                }
                                stack.pop_front();
                                if (curr != concat)
                                {
                                    for (auto next : curr->get_users())
                                    {
                                        if (visited.count(next.get()) == 0)
                                        {
                                            stack.push_front(next.get());
                                        }
                                    }
                                }
                            }
                            in_place_concat = true;
                        }
                        else
                        {
                            in_place_concat = true;
                        }
                    }

                    if (in_place_concat)
                    {
                        auto op_annotations = concat->get_op_annotations();
                        if (op_annotations)
                        {
                            op_annotations->add_in_place_oi_pair({0, 0, false});
                        }
                        else
                        {
                            op_annotations =
                                std::make_shared<ngraph::runtime::cpu::CPUOpAnnotations>();
                            op_annotations->add_in_place_oi_pair({0, 0, false});
                            concat->set_op_annotations(op_annotations);
                        }
                    }
                }
            }
        }
    }
}

#define TI(x) type_index(typeid(x))

static const runtime::cpu::pass::PostLayoutAssignOpMap s_dispatcher{
    {TI(ngraph::op::Concat),
     &runtime::cpu::pass::CPUPostLayoutAssignment::assign<ngraph::op::Concat>},
};

bool runtime::cpu::pass::CPUPostLayoutAssignment::run_on_call_graph(
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
