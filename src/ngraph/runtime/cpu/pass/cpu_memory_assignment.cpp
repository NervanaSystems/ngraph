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

#include <exception>
#include <sstream>

#include "ngraph/log.hpp"
#include "ngraph/log.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/memory_layout.hpp"
#include "ngraph/runtime/cpu/cpu_op_annotations.hpp"
#include "ngraph/runtime/cpu/pass/cpu_memory_assignment.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

runtime::cpu::pass::CPUMemoryAssignment::CPUMemoryAssignment(size_t alignment,
                                                             bool disable_memory_sharing)
    : m_alignment(alignment)
    , m_disable_memory_sharing(disable_memory_sharing)
{
    if (m_alignment == 0)
    {
        throw invalid_argument("Memory alignment must be > 0");
    }
}

bool runtime::cpu::pass::CPUMemoryAssignment::run_on_function(shared_ptr<ngraph::Function> function)
{
    list<shared_ptr<Node>> ops = function->get_ordered_ops();

    // build tensor alias maps for in place ops
    // forward in place ops such as in place slice or reshape
    unordered_map<descriptor::Tensor*, descriptor::Tensor*> tensor_alias_map;
    // backward in place ops: concat
    unordered_map<descriptor::Tensor*, descriptor::Tensor*> tensor_alias_backward_map;

    auto propagate_in_place_concat = [&](shared_ptr<ngraph::op::Concat> concat,
                                         descriptor::Tensor* output_tensor) {
        std::deque<std::shared_ptr<ngraph::op::Concat>> stack;
        stack.push_front(concat);

        while (stack.size() > 0)
        {
            auto it = stack.front();
            stack.pop_front();
            if (auto op_annotations = it->get_op_annotations())
            {
                auto in_place_oi_pairs = op_annotations->get_in_place_oi_pairs();
                if (in_place_oi_pairs.size() > 0)
                {
                    for (auto arg : it->get_arguments())
                    {
                        auto input_tensor = &arg->get_output_tensor();
                        if (tensor_alias_map.find(input_tensor) == tensor_alias_map.end())
                        {
                            tensor_alias_backward_map[input_tensor] = output_tensor;
                            if (arg->description() == "Concat")
                            {
                                auto arg_concat = std::static_pointer_cast<ngraph::op::Concat>(arg);
                                stack.push_front(arg_concat);
                            }
                        }
                    }
                }
            }
        }
    };

    for (auto it = ops.begin(); it != ops.end(); it++)
    {
        const shared_ptr<Node>& node = *it;
        if (node->is_parameter() || node->is_constant())
        {
            continue;
        }
        if (node->is_op())
        {
            auto op = std::static_pointer_cast<op::Op>(node);
            if (auto op_annotations = op->get_op_annotations())
            {
                auto in_place_oi_pairs = op_annotations->get_in_place_oi_pairs();
                if (in_place_oi_pairs.size() > 0)
                {
                    if (node->description() == "Concat")
                    {
                        auto concat = std::static_pointer_cast<ngraph::op::Concat>(node);
                        auto cpu_op_annotations =
                            std::static_pointer_cast<runtime::cpu::CPUOpAnnotations>(
                                op_annotations);
                        // last concat, propagate
                        if (!cpu_op_annotations->can_free_memory())
                        {
                            //propagate tensor alias
                            auto output_tensor = &concat->get_output_tensor();
                            for (auto& arg : concat->get_arguments())
                            {
                                auto input_tensor = &arg->get_output_tensor();
                                if (tensor_alias_map.find(input_tensor) == tensor_alias_map.end())
                                {
                                    tensor_alias_backward_map[input_tensor] = output_tensor;
                                    if (arg->description() == "Concat")
                                    {
                                        auto arg_concat =
                                            static_pointer_cast<ngraph::op::Concat>(arg);
                                        propagate_in_place_concat(arg_concat, output_tensor);
                                    }
                                }
                            }
                        }
                    }
                    else
                    {
                        for (auto oi_pair : op_annotations->get_in_place_oi_pairs())
                        {
                            // check if any user has destructive io
                            bool no_tensor_alias = false;
                            for (auto& user : node->get_outputs().at(oi_pair.output).get_inputs())
                            {
                                auto user_node = user->get_node();
                                auto input_index = user->get_index();
                                if (user_node->is_op())
                                {
                                    auto user_op = std::static_pointer_cast<op::Op>(user_node);
                                    if (auto user_op_annotations = op->get_op_annotations())
                                    {
                                        auto user_in_place_oi_pairs =
                                            user_op_annotations->get_in_place_oi_pairs();
                                        for (auto& user_oi_pair : user_in_place_oi_pairs)
                                        {
                                            if (user_oi_pair.input =
                                                    input_index && user_oi_pair.destructive)
                                            {
                                                no_tensor_alias = true;
                                                break;
                                            }
                                        }
                                    }
                                }
                                if (no_tensor_alias)
                                {
                                    break;
                                }
                            }
                            if (no_tensor_alias)
                            {
                                continue;
                            }
                            auto output_tensor =
                                &node->get_outputs().at(oi_pair.output).get_tensor();
                            auto input_tensor = &node->get_inputs().at(oi_pair.input).get_tensor();
                            if (tensor_alias_map.find(input_tensor) == tensor_alias_map.end())
                            {
                                tensor_alias_map[output_tensor] = input_tensor;
                            }
                            else
                            {
                                tensor_alias_map[output_tensor] = tensor_alias_map[input_tensor];
                            }
                        }
                    }
                }
            }
        }
    }

    // liveness analysis using tensor alias maps
    unordered_set<descriptor::Tensor*> persistent_tensors;
    unordered_set<descriptor::Tensor*> output_tensors;
    for (const shared_ptr<op::Parameter>& node : function->get_parameters())
    {
        for (size_t i = 0; i < node->get_output_size(); ++i)
        {
            descriptor::Tensor& tensor = node->get_output_tensor(i);
            persistent_tensors.insert(&tensor);
        }
    }
    for (const shared_ptr<op::Result>& node : function->get_results())
    {
        for (size_t i = 0; i < node->get_output_size(); ++i)
        {
            descriptor::Tensor& tensor = node->get_output_tensor(i);
            persistent_tensors.insert(&tensor);
            output_tensors.insert(&tensor);
        }
    }
    for (const shared_ptr<Node>& node : ops)
    {
        if (auto constant_node = dynamic_pointer_cast<op::Constant>(node))
        {
            for (size_t i = 0; i < constant_node->get_output_size(); ++i)
            {
                descriptor::Tensor& tensor = constant_node->get_output_tensor(i);
                persistent_tensors.insert(&tensor);
            }
        }
    }

    unordered_set<descriptor::Tensor*> currently_live;
    for (auto it = ops.rbegin(); it != ops.rend(); it++)
    {
        const shared_ptr<Node>& node = *it;
        node->liveness_new_list.clear();
        node->liveness_free_list.clear();
        unordered_set<descriptor::Tensor*> input_tensor_decls;
        for (descriptor::Input& input_decl : node->get_inputs())
        {
            descriptor::Tensor& tensor = input_decl.get_tensor();
            if (persistent_tensors.find(&tensor) == persistent_tensors.end())
            {
                input_tensor_decls.insert(&tensor);
            }
        }

        unordered_set<descriptor::Tensor*> output_tensor_decls;
        for (size_t i = 0; i < node->get_output_size(); ++i)
        {
            descriptor::Tensor& tensor = node->get_output_tensor(i);
            if (persistent_tensors.find(&tensor) == persistent_tensors.end())
            {
                output_tensor_decls.insert(&tensor);
            }
        }

        unordered_set<descriptor::Tensor*> free_tensor_decls;
        unordered_set<descriptor::Tensor*> new_tensor_decls;
        unordered_set<descriptor::Tensor*> all_tensor_decls = input_tensor_decls;
        all_tensor_decls.insert(output_tensor_decls.begin(), output_tensor_decls.end());

        for (descriptor::Tensor* tensor_decl : all_tensor_decls)
        {
            if (tensor_alias_map.find(tensor_decl) != tensor_alias_map.end())
            {
                if (currently_live.find(tensor_alias_map[tensor_decl]) == currently_live.end())
                {
                    // this is the last node that value is seen in
                    // delete it at the end of the op
                    currently_live.insert(tensor_alias_map[tensor_decl]);
                    if (output_tensors.find(tensor_alias_map[tensor_decl]) ==
                            output_tensors.end() &&
                        persistent_tensors.find(tensor_alias_map[tensor_decl]) ==
                            persistent_tensors.end())
                    {
                        // Don't free output tensors
                        free_tensor_decls.insert(tensor_alias_map[tensor_decl]);
                    }
                }
            }
            else if (tensor_alias_backward_map.find(tensor_decl) == tensor_alias_backward_map.end())
            {
                if (currently_live.find(tensor_decl) == currently_live.end())
                {
                    // this is the last node that value is seen in
                    // delete it at the end of the op
                    currently_live.insert(tensor_decl);
                    if (output_tensors.find(tensor_decl) == output_tensors.end())
                    {
                        // Don't free output tensors
                        free_tensor_decls.insert(tensor_decl);
                    }
                }
            }
        }

        for (descriptor::Tensor* output_decl : output_tensor_decls)
        {
            if (tensor_alias_backward_map.find(output_decl) != tensor_alias_backward_map.end())
            {
                auto currently_live_it =
                    currently_live.find(tensor_alias_backward_map[output_decl]);
                if (currently_live_it != currently_live.end())
                {
                    new_tensor_decls.insert(tensor_alias_backward_map[output_decl]);
                    currently_live.erase(currently_live_it);
                }
            }
            else
            {
                auto currently_live_it = currently_live.find(output_decl);
                if (currently_live_it != currently_live.end())
                {
                    new_tensor_decls.insert(output_decl);
                    currently_live.erase(currently_live_it);
                }
            }
        }
        node->liveness_free_list = free_tensor_decls;
        node->liveness_new_list = new_tensor_decls;
    }

    // memory assignment using liveness analysis result
    // memory manager for non-cacheable ops, memory allocation will be freed when not longer in use
    ngraph::pass::MemoryManager mm(m_alignment, m_disable_memory_sharing);
    // memory manager for cacheable ops, memory allocation will never be freed
    ngraph::pass::MemoryManager mm_caching(m_alignment, true);

    if (!m_disable_memory_sharing)
    {
        // build caching map from cacheability
        for (shared_ptr<Node> node : function->get_ordered_ops())
        {
            if (node->is_op())
            {
                auto op = std::static_pointer_cast<op::Op>(node);
                if (auto op_annotations = op->get_op_annotations())
                {
                    auto cacheable = op_annotations->is_cacheable();

                    if (cacheable)
                    {
                        for (size_t i = 0; i < node->get_output_size(); ++i)
                        {
                            shared_ptr<descriptor::Tensor> tv = node->get_output_tensor_ptr(i);
                            m_tensor_caching.insert(tv.get());
                        }
                    }
                }
            }
        }

        // add other tensors due to in place propagation
        for (shared_ptr<Node> node : function->get_ordered_ops())
        {
            if (node->is_op())
            {
                auto op = std::static_pointer_cast<op::Op>(node);
                if (auto op_annotations = op->get_op_annotations())
                {
                    //if it is last concat node of in-place-concat-chain, put it in caching map
                    auto in_place_oi_pairs = op_annotations->get_in_place_oi_pairs();
                    if (in_place_oi_pairs.size() > 0)
                    {
                        if (node->description() == "Concat")
                        {
                            auto concat = std::static_pointer_cast<ngraph::op::Concat>(node);
                            auto cpu_op_annotations =
                                std::static_pointer_cast<runtime::cpu::CPUOpAnnotations>(
                                    op_annotations);
                            if (!cpu_op_annotations->can_free_memory())
                            {
                                shared_ptr<descriptor::Tensor> tv = node->get_output_tensor_ptr(0);
                                m_tensor_caching.insert(tv.get());
                            }
                        }
                    }
                }
            }
        }
    }

    for (shared_ptr<Node> node : function->get_ordered_ops())
    {
        for (descriptor::Tensor* tensor : node->liveness_new_list)
        {
            size_t offset = 0;
            if (m_tensor_caching.count(tensor) != 0)
            {
                offset = mm_caching.allocate(tensor->size());
            }
            else
            {
                offset = mm.allocate(tensor->size());
            }
            tensor->set_pool_offset(offset);
        }

        if (!m_disable_memory_sharing)
        {
            for (descriptor::Tensor* tensor : node->liveness_free_list)
            {
                if (m_tensor_caching.empty() ||
                    (!m_tensor_caching.empty() && m_tensor_caching.count(tensor) == 0))
                {
                    mm.free(tensor->get_pool_offset());
                }
            }
        }
    }

    //update the offset for tensors in tensor_caching
    auto start = mm.max_allocated();
    for (auto item : m_tensor_caching)
    {
        auto new_offset = item->get_pool_offset() + start;
        item->set_pool_offset(new_offset);
    }

    //set pool offset for original tensors in the tensor alias maps.
    for (auto it = tensor_alias_map.begin(); it != tensor_alias_map.end(); it++)
    {
        it->first->set_pool_offset(it->second->get_pool_offset());
    }
    for (auto it = tensor_alias_backward_map.begin(); it != tensor_alias_backward_map.end(); it++)
    {
        it->first->set_pool_offset(it->second->get_pool_offset());
    }

    NGRAPH_DEBUG << "cpu_memory_assignemnt: max allocated for mm is " << mm.max_allocated();
    NGRAPH_DEBUG << "cpu_memory_assignment: max allocated for mm_caching is "
                 << mm_caching.max_allocated();
    NGRAPH_DEBUG << "cpu_memory_assignment: max allocated in total is "
                 << mm.max_allocated() + mm_caching.max_allocated();

    function->set_temporary_pool_size(mm.max_allocated() + mm_caching.max_allocated());

    return false;
}
