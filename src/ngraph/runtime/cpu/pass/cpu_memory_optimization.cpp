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

#include "ngraph/runtime/cpu/pass/cpu_memory_optimization.hpp"

#include "ngraph/descriptor/output.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/runtime/cpu/cpu_op_annotations.hpp"

using namespace ngraph;

bool runtime::cpu::pass::CPUMemoryOptimization::run_on_function(std::shared_ptr<Function> function)
{
    for (auto n : function->get_ordered_ops())
    {
        if (auto concat = std::dynamic_pointer_cast<op::Concat>(n))
        {
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
                continue;
            }

            bool in_place_concat = true;

            for (descriptor::Input& input : concat->get_inputs())
            {
                if (shape_size(input.get_shape()) == 0)
                {
                    NGRAPH_DEBUG << "cpu_post_layout_assignment: 0 length tensor, no in "
                                    "place concat";
                    in_place_concat = false;
                    break;
                }

                const auto& output = input.get_output();
                auto arg = output.get_node();
                if (std::dynamic_pointer_cast<op::Constant>(arg) ||
                    std::dynamic_pointer_cast<op::Parameter>(arg))
                {
                    NGRAPH_DEBUG << "cpu_post_layout_assignment: " << arg->get_name()
                                 << ": constant or parameter, no in place concat";
                    in_place_concat = false;
                    break;
                }

                if (arg->get_output_size() != 1)
                {
                    NGRAPH_DEBUG << "cpu_post_layout_assignment: " << arg->get_name()
                                 << ": multiple outputs, no in place concat";
                    in_place_concat = false;
                    break;
                }

                if (!std::dynamic_pointer_cast<op::Concat>(arg))
                {
                    if (auto op = std::dynamic_pointer_cast<op::Op>(arg))
                    {
                        auto annotation = op->get_op_annotations();
                        if (annotation && annotation->get_in_place_oi_pairs().size() > 0)

                        {
                            NGRAPH_DEBUG << "cpu_post_layout_assignment: " << arg->get_name()
                                         << ": in place non concat op, no in place concat";
                            in_place_concat = false;
                            break;
                        }
                    }
                }

                if (output.get_inputs().size() != 1)
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
                                in_place_concat = false;
                                break;
                            }
                        }
                    }
                    if (!in_place_concat)
                    {
                        break;
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
                            in_place_concat = false;
                            break;
                        }
                        else
                        {
                            if (auto op = dynamic_cast<op::Op*>(curr))
                            {
                                if (auto op_annotations = op->get_op_annotations())
                                {
                                    for (auto oi_pair : op_annotations->get_in_place_oi_pairs())
                                    {
                                        if (oi_pair.destructive)
                                        {
                                            NGRAPH_DEBUG << "cpu_post_layout_assignment: "
                                                            "destructive in place oi, no "
                                                            "in place concat";
                                            in_place_concat = false;
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                        stack.pop_front();
                        if (curr != concat.get())
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
                    if (!in_place_concat)
                    {
                        break;
                    }
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
                    op_annotations = std::make_shared<ngraph::runtime::cpu::CPUOpAnnotations>();
                    op_annotations->add_in_place_oi_pair({0, 0, false});
                    concat->set_op_annotations(op_annotations);
                }
            }
        }
    }
    return false;
}
