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

runtime::cpu::pass::CPUMemoryAssignment::CPUMemoryAssignment(
    unordered_map<size_t, std::pair<CPUTensorRole, unordered_set<descriptor::Tensor*>>>&
        key_to_tensors_set_map,
    unordered_map<descriptor::Tensor*, size_t>& tensor_to_key_map,
    size_t alignment,
    bool disable_memory_sharing)
    : m_alignment(alignment)
    , m_disable_memory_sharing(disable_memory_sharing)
    , m_key_to_tensors_set_map(key_to_tensors_set_map)
    , m_tensor_to_key_map(tensor_to_key_map)
{
    if (m_alignment == 0)
    {
        throw invalid_argument("Memory alignment must be > 0");
    }
}

void runtime::cpu::pass::CPUMemoryAssignment::process_in_place_concat(
    std::list<std::shared_ptr<Node>> nodes)
{
    for (shared_ptr<Node> node : nodes)
    {
        if (node->description() == "Concat")
        {
            auto concat = std::static_pointer_cast<ngraph::op::Concat>(node);
            if (auto op_annotations = concat->get_op_annotations())
            {
                auto in_place_oi_pairs = op_annotations->get_in_place_oi_pairs();
                if (in_place_oi_pairs.size() > 0)
                {
                    // check if it is the last in place concat
                    bool found_last_concat = true;
                    for (auto user : concat->get_users())
                    {
                        if (user->description() == "Concat")
                        {
                            auto user_concat = std::static_pointer_cast<ngraph::op::Concat>(user);
                            if (auto user_op_annotations = user_concat->get_op_annotations())
                            {
                                auto user_in_place_oi_pairs =
                                    user_op_annotations->get_in_place_oi_pairs();
                                if (user_in_place_oi_pairs.size() > 0)
                                {
                                    found_last_concat = false;
                                    break;
                                }
                            }
                        }
                    }
                    // start from the last concat
                    if (found_last_concat)
                    {
                        auto output_tensor = &concat->get_output_tensor();
                        NGRAPH_ASSERT(m_tensor_to_key_map.find(output_tensor) !=
                                      m_tensor_to_key_map.end());
                        auto output_key = m_tensor_to_key_map[output_tensor];

                        auto offset = output_tensor->get_pool_offset();
                        size_t arg_index = 0;
                        for (auto arg : concat->get_arguments())
                        {
                            auto input_tensor = &arg->get_output_tensor();
                            NGRAPH_ASSERT(m_tensor_to_key_map.find(input_tensor) !=
                                          m_tensor_to_key_map.end());
                            auto input_key = m_tensor_to_key_map[input_tensor];
                            // same set, in place concat allowed
                            if (input_key == output_key)
                            {
                                auto old_offset = input_tensor->get_pool_offset();
                                input_tensor->set_pool_offset(offset);
                                NGRAPH_DEBUG
                                    << "cpu_memory_assignment: change offset, old offset is "
                                    << old_offset << ", new offset is " << offset << std::endl;
                                offset += input_tensor->size();

                                // check if need to propagate backward
                                if (arg->is_op())
                                {
                                    auto arg_op = std::static_pointer_cast<ngraph::op::Op>(arg);
                                    if (auto arg_op_annotations = arg_op->get_op_annotations())
                                    {
                                        auto arg_in_place_oi_pairs =
                                            arg_op_annotations->get_in_place_oi_pairs();
                                        if (arg_in_place_oi_pairs.size() > 0)
                                        {
                                            auto input = &arg_op->get_inputs().at(0);
                                            auto output_index = input->get_output().get_index();
                                            NGRAPH_DEBUG << "cpu_memory_assignment: call "
                                                            "propagate_in_place_concat for "
                                                         << arg->get_name();
                                            propagate_in_place_concat(arg_op, output_index);
                                        }
                                    }
                                }
                            }
                            arg_index++;
                        }
                    }
                }
            }
        }
    }
}

void runtime::cpu::pass::CPUMemoryAssignment::propagate_in_place_concat(
    shared_ptr<ngraph::op::Op> op, size_t output_index)
{
    if (op->description() == "Concat")
    {
        auto output_tensor = &op->get_output_tensor();
        NGRAPH_ASSERT(m_tensor_to_key_map.find(output_tensor) != m_tensor_to_key_map.end());
        auto output_key = m_tensor_to_key_map[output_tensor];

        auto offset = output_tensor->get_pool_offset();
        size_t arg_index = 0;
        for (auto arg : op->get_arguments())
        {
            auto input_tensor = &arg->get_output_tensor();
            NGRAPH_ASSERT(m_tensor_to_key_map.find(input_tensor) != m_tensor_to_key_map.end());
            auto input_key = m_tensor_to_key_map[input_tensor];
            // same set, in place concat allowed
            if (input_key == output_key)
            {
                auto old_offset = input_tensor->get_pool_offset();
                input_tensor->set_pool_offset(offset);
                NGRAPH_DEBUG << "cpu_memory_assignment: change offset, old offset is " << old_offset
                             << ", new offset is " << offset;
                offset += input_tensor->size();

                // check if need to propagate backward
                if (arg->is_op())
                {
                    auto arg_op = std::static_pointer_cast<ngraph::op::Op>(arg);
                    if (auto arg_op_annotations = arg_op->get_op_annotations())
                    {
                        auto arg_in_place_oi_pairs = arg_op_annotations->get_in_place_oi_pairs();
                        if (arg_in_place_oi_pairs.size() > 0)
                        {
                            NGRAPH_DEBUG
                                << "cpu_memory_assignment: call propagate_in_place_concat for "
                                << arg->get_name();
                            auto input = &op->get_inputs().at(arg_index);
                            auto arg_output_index = input->get_output().get_index();
                            propagate_in_place_concat(arg_op, arg_output_index);
                        }
                    }
                }
            }
            arg_index++;
        }
    }
    else
    {
        // other in place ops
        auto op_annotations = op->get_op_annotations();
        for (auto oi_pair : op_annotations->get_in_place_oi_pairs())
        {
            if (oi_pair.output != output_index || oi_pair.destructive)
            {
                continue;
            }

            auto input_tensor = &op->get_inputs().at(oi_pair.input).get_tensor();
            NGRAPH_ASSERT(m_tensor_to_key_map.find(input_tensor) != m_tensor_to_key_map.end());
            auto input_key = m_tensor_to_key_map[input_tensor];
            auto output_tensor = &op->get_outputs().at(oi_pair.output).get_tensor();
            NGRAPH_ASSERT(m_tensor_to_key_map.find(output_tensor) != m_tensor_to_key_map.end());
            auto output_key = m_tensor_to_key_map[output_tensor];

            // same set, in place op allowed
            if (input_key == output_key)
            {
                auto old_offset = input_tensor->get_pool_offset();
                auto new_offset = output_tensor->get_pool_offset();
                input_tensor->set_pool_offset(new_offset);
                NGRAPH_DEBUG << "cpu_memory_assignment: change offset, old offset is " << old_offset
                             << ", new offset is " << new_offset;
                auto input = &op->get_inputs().at(oi_pair.input);
                auto arg = input->get_node();

                // check if need to propagate backward
                if (arg->is_op())
                {
                    auto arg_op = std::static_pointer_cast<ngraph::op::Op>(arg);
                    if (auto arg_op_annotations = arg_op->get_op_annotations())
                    {
                        auto arg_in_place_oi_pairs = arg_op_annotations->get_in_place_oi_pairs();
                        if (arg_in_place_oi_pairs.size() > 0)
                        {
                            auto arg_output_index = input->get_output().get_index();
                            NGRAPH_DEBUG
                                << "cpu_memory_assignment: call propagate_in_place_concat for "
                                << arg->get_name();
                            propagate_in_place_concat(arg_op, arg_output_index);
                        }
                    }
                }
            }
        }
    }
}

//slice
void runtime::cpu::pass::CPUMemoryAssignment::process_in_place_slice(
    std::list<std::shared_ptr<Node>> nodes)
{
    for (shared_ptr<Node>& node : nodes)
    {
        if (node->description() == "Slice")
        {
            auto slice = std::static_pointer_cast<ngraph::op::Slice>(node);
            if (auto op_annotations = slice->get_op_annotations())
            {
                auto in_place_oi_pairs = op_annotations->get_in_place_oi_pairs();
                if (in_place_oi_pairs.size() > 0)
                {
                    auto input = &slice->get_inputs().at(0);
                    auto arg = input->get_output().get_node();
                    auto index = input->get_output().get_index();
                    auto input_tensor = &arg->get_output_tensor(index);
                    NGRAPH_ASSERT(m_tensor_to_key_map.find(input_tensor) !=
                                  m_tensor_to_key_map.end());
                    auto input_key = m_tensor_to_key_map[input_tensor];
                    auto output_tensor = &slice->get_output_tensor();
                    NGRAPH_ASSERT(m_tensor_to_key_map.find(output_tensor) !=
                                  m_tensor_to_key_map.end());
                    auto output_key = m_tensor_to_key_map[output_tensor];

                    // same set, in place slice allowed
                    if (input_key == output_key)
                    {
                        NGRAPH_ASSERT(m_key_to_tensors_set_map.find(output_key) !=
                                      m_key_to_tensors_set_map.end());
                        auto offset = input_tensor->get_pool_offset();
                        auto lower_bounds = slice->get_lower_bounds();
                        auto start = 0, accumulated = 1;
                        auto in_shape = slice->get_input_shape(0);
                        for (int i = in_shape.size() - 1; i >= 0; i--)
                        {
                            start += lower_bounds[i] * accumulated;
                            accumulated *= in_shape[i];
                        }

                        auto old_offset = output_tensor->get_pool_offset();
                        offset += slice->get_element_type().size() * start;
                        output_tensor->set_pool_offset(offset);
                        NGRAPH_DEBUG
                            << "cpu_memory_assignment: slice, change offset, old offset is "
                            << old_offset << ", new offset is " << offset;

                        // check if need to propagate forward
                        for (size_t i = 0; i < slice->get_output_size(); ++i)
                        {
                            auto slice_output = &slice->get_outputs().at(i);
                            for (auto slice_output_input : slice_output->get_inputs())
                            {
                                NGRAPH_DEBUG
                                    << "cpu_memory_assignment: call propagate_in_place_slice "
                                       "for output "
                                    << i << " of " << slice->get_name();
                                propagate_in_place_slice(slice_output_input,
                                                         slice_output_input->get_index());
                            }
                        }
                    }
                }
            }
        }
    }
}

void runtime::cpu::pass::CPUMemoryAssignment::propagate_in_place_slice(
    ngraph::descriptor::Input* input, size_t input_index)
{
    std::deque<std::pair<ngraph::descriptor::Input*, size_t>> stack;
    stack.push_front(std::pair<ngraph::descriptor::Input*, size_t>(input, input_index));

    while (stack.size() > 0)
    {
        ngraph::descriptor::Input* in = stack.front().first;
        auto index = stack.front().second;
        stack.pop_front();

        auto node = in->get_node();
        // let process_in_place_slice handle slice.
        if (!node->is_op() || node->description() == "Slice")
        {
            continue;
        }
        auto op = std::static_pointer_cast<ngraph::op::Op>(node);
        if (auto op_annotations = op->get_op_annotations())
        {
            for (auto oi_pair : op_annotations->get_in_place_oi_pairs())
            {
                if (oi_pair.input == index)
                {
                    auto input_tensor = &op->get_inputs().at(oi_pair.input).get_tensor();
                    NGRAPH_ASSERT(m_tensor_to_key_map.find(input_tensor) !=
                                  m_tensor_to_key_map.end());
                    auto input_key = m_tensor_to_key_map[input_tensor];
                    size_t output_index = oi_pair.output;
                    auto output_tensor = &op->get_outputs().at(output_index).get_tensor();
                    NGRAPH_ASSERT(m_tensor_to_key_map.find(output_tensor) !=
                                  m_tensor_to_key_map.end());
                    auto output_key = m_tensor_to_key_map[output_tensor];

                    // same set, in place op allowed
                    if (input_key == output_key)
                    {
                        output_tensor->set_pool_offset(input_tensor->get_pool_offset());
                        for (size_t i = 0; i < op->get_output_size(); ++i)
                        {
                            auto op_output = &op->get_outputs().at(i);
                            for (auto op_output_input : op_output->get_inputs())
                            {
                                stack.push_front(std::pair<ngraph::descriptor::Input*, size_t>(
                                    op_output_input, op_output_input->get_index()));
                            }
                        }
                    }
                }
            }
        }
    }
}

bool runtime::cpu::pass::CPUMemoryAssignment::run_on_function(shared_ptr<ngraph::Function> function)
{
    list<shared_ptr<Node>> ops = function->get_ordered_ops();
    unordered_set<descriptor::Tensor*> in_place_slice_chain;

    // build buffer sets maps
    size_t count = 0;
    for (auto it = ops.begin(); it != ops.end(); it++)
    {
        const shared_ptr<Node>& node = *it;
        if (node->is_parameter())
        {
            auto output_tensor = &node->get_output_tensor();
            auto ele = std::pair<CPUTensorRole, unordered_set<descriptor::Tensor*>>(
                CPUTensorRole::INPUT, unordered_set<descriptor::Tensor*>({output_tensor}));
            m_key_to_tensors_set_map[count] = ele;
            m_tensor_to_key_map[output_tensor] = count;
            count++;
        }
        else if (node->is_constant())
        {
            auto output_tensor = &node->get_output_tensor();
            auto ele = std::pair<CPUTensorRole, unordered_set<descriptor::Tensor*>>(
                CPUTensorRole::CONSTANT, unordered_set<descriptor::Tensor*>({output_tensor}));
            m_key_to_tensors_set_map[count] = ele;
            m_tensor_to_key_map[output_tensor] = count;
            count++;
        }
        else if (node->is_output())
        {
            auto output_tensor = &node->get_output_tensor();
            auto input_tensor = &node->get_inputs().at(0).get_tensor();
            NGRAPH_ASSERT(m_tensor_to_key_map.find(input_tensor) != m_tensor_to_key_map.end() &&
                          m_tensor_to_key_map[input_tensor] <= count);
            auto key = m_tensor_to_key_map[input_tensor];
            NGRAPH_ASSERT(m_key_to_tensors_set_map.find(key) != m_key_to_tensors_set_map.end());
            auto pair = m_key_to_tensors_set_map[key];
            if (pair.first != CPUTensorRole::INTERMEDIATE ||
                in_place_slice_chain.find(input_tensor) != in_place_slice_chain.end())
            {
                // tensor of function output should not be in the same set as function input, constant, output, or in place slice,
                // because they cannot share the same memory buffer
                auto ele = std::pair<CPUTensorRole, unordered_set<descriptor::Tensor*>>(
                    CPUTensorRole::OUTPUT, unordered_set<descriptor::Tensor*>({output_tensor}));
                m_key_to_tensors_set_map[count] = ele;
                m_tensor_to_key_map[output_tensor] = count;
                count++;
            }
            else
            {
                //in place output
                m_key_to_tensors_set_map[key].first = CPUTensorRole::OUTPUT;
                m_key_to_tensors_set_map[key].second.insert(output_tensor);
                m_tensor_to_key_map[output_tensor] = key;
            }
        }
        else if (node->is_op())
        {
            auto op = std::static_pointer_cast<op::Op>(node);
            if (auto op_annotations = op->get_op_annotations())
            {
                auto in_place_oi_pairs = op_annotations->get_in_place_oi_pairs();
                if (in_place_oi_pairs.size() > 0)
                {
                    auto cacheable = op_annotations->is_cacheable();

                    // in place concat
                    if (node->description() == "Concat")
                    {
                        auto output_tensor = &node->get_output_tensor();
                        auto ele = std::pair<CPUTensorRole, unordered_set<descriptor::Tensor*>>(
                            CPUTensorRole::INTERMEDIATE,
                            unordered_set<descriptor::Tensor*>({output_tensor}));
                        for (auto& arg : node->get_arguments())
                        {
                            // when reusing memory, check cacheability
                            if (!m_disable_memory_sharing && arg->is_op())
                            {
                                auto arg_op = std::static_pointer_cast<op::Op>(arg);
                                if (auto arg_op_annotations = arg_op->get_op_annotations())
                                {
                                    // when reusing memory, ops with different cacheabilities should not be in the same set.
                                    if (cacheable != arg_op_annotations->is_cacheable())
                                    {
                                        continue;
                                    }
                                }
                            }
                            // no in-place concat if arg is in in_place_slice_chain,
                            // because in-place slice before in-place concat cannot use the memory buffer of concat.
                            // in-place slice after in-place concat can use the memory buffer of concat.
                            auto input_tensor = &arg->get_output_tensor();
                            if (in_place_slice_chain.find(input_tensor) !=
                                in_place_slice_chain.end())
                            {
                                NGRAPH_DEBUG << "cpu_memory_assignment: no in place concat after "
                                                "in place slice";
                                continue;
                            }

                            NGRAPH_ASSERT(m_tensor_to_key_map.find(input_tensor) !=
                                              m_tensor_to_key_map.end() &&
                                          m_tensor_to_key_map[input_tensor] <= count);
                            auto key = m_tensor_to_key_map[input_tensor];

                            // tensor set is erased from the map when processing previous arg
                            if (m_key_to_tensors_set_map.find(key) ==
                                m_key_to_tensors_set_map.end())
                            {
                                continue;
                            }
                            auto pair = m_key_to_tensors_set_map[key];
                            // no in-place concat if arg is from parameter or constant
                            if (pair.first == CPUTensorRole::INPUT ||
                                pair.first == CPUTensorRole::CONSTANT)
                            {
                                continue;
                            }
                            // in-place concat
                            // move tensors in the set containing the input tensor to the set of output tensor
                            // then erase that input tensor set
                            for (auto tensor : pair.second)
                            {
                                m_tensor_to_key_map[tensor] = count;
                                ele.second.insert(tensor);
                            }
                            m_key_to_tensors_set_map.erase(key);
                        }
                        // put the set containing output tensor into the map
                        m_key_to_tensors_set_map[count] = ele;
                        m_tensor_to_key_map[output_tensor] = count;
                        count++;
                    }
                    else
                    {
                        // other in place ops
                        for (auto oi_pair : op_annotations->get_in_place_oi_pairs())
                        {
                            auto input_tensor = &node->get_inputs().at(oi_pair.input).get_tensor();
                            auto output_tensor =
                                &node->get_outputs().at(oi_pair.output).get_tensor();

                            // keep track of tensors sharing the memory buffer with in-place slice output tensor
                            if (in_place_slice_chain.find(input_tensor) !=
                                in_place_slice_chain.end())
                            {
                                in_place_slice_chain.insert(output_tensor);
                            }

                            // if destructive, do not put input tensor and output tensor into the same set.
                            if (!oi_pair.destructive)
                            {
                                bool no_in_place = false;
                                auto input_node = node->get_inputs().at(oi_pair.input).get_node();
                                // when reusing memory, check cacheability
                                if (!m_disable_memory_sharing && input_node->is_op())
                                {
                                    auto input_op = std::static_pointer_cast<op::Op>(input_node);
                                    if (auto input_op_annotations = input_op->get_op_annotations())
                                    {
                                        // when reusing memory, ops with different cacheabilities should not be in the same set.
                                        if (cacheable != input_op_annotations->is_cacheable())
                                        {
                                            NGRAPH_DEBUG << "cpu_memory_assignment: no in place "
                                                            "due to cacheability";
                                            no_in_place = true;
                                        }
                                    }
                                }
                                if (!no_in_place)
                                {
                                    if (node->description() == "Slice")
                                    {
                                        // build in place slice chain
                                        in_place_slice_chain.insert(output_tensor);
                                    }
                                    NGRAPH_ASSERT(m_tensor_to_key_map.find(input_tensor) !=
                                                  m_tensor_to_key_map.end());
                                    auto key = m_tensor_to_key_map[input_tensor];
                                    m_key_to_tensors_set_map[key].second.insert(output_tensor);
                                    m_tensor_to_key_map[output_tensor] = key;
                                }
                            }
                        }
                    }
                }
            }
            // process output tensors
            for (auto i = 0; i < node->get_output_size(); i++)
            {
                auto output_tensor = &node->get_outputs().at(i).get_tensor();
                // not in place, create a new set and insert into the map
                if (m_tensor_to_key_map.find(output_tensor) == m_tensor_to_key_map.end())
                {
                    auto ele = std::pair<CPUTensorRole, unordered_set<descriptor::Tensor*>>(
                        CPUTensorRole::INTERMEDIATE,
                        unordered_set<descriptor::Tensor*>({output_tensor}));
                    m_key_to_tensors_set_map[count] = ele;
                    m_tensor_to_key_map[output_tensor] = count;
                    count++;
                }
            }
        }
        else
        {
            // function call
            for (auto i = 0; i < node->get_output_size(); i++)
            {
                auto output_tensor = &node->get_outputs().at(i).get_tensor();
                auto ele = std::pair<CPUTensorRole, unordered_set<descriptor::Tensor*>>(
                    CPUTensorRole::INTERMEDIATE,
                    unordered_set<descriptor::Tensor*>({output_tensor}));
                m_key_to_tensors_set_map[count] = ele;
                m_tensor_to_key_map[output_tensor] = count;
                count++;
            }
        }
    }

    auto find_role = [](CPUTensorRole tensor_role) -> string {
        switch (tensor_role)
        {
        case CPUTensorRole::INPUT: return string("CPUTensorRole::INPUT");
        case CPUTensorRole::INTERMEDIATE: return string("CPUTensorRole::INTERMEDIATE");
        case CPUTensorRole::CONSTANT: return string("CPUTensorRole::CONSTANT");
        case CPUTensorRole::OUTPUT: return string("CPUTensorRole::OUTPUT");
        }
        throw runtime_error("unhandled CPU tensor role");
    };

    //liveness analysis
    unordered_set<size_t> allocated_sets;
    unordered_set<size_t> freed_sets;
    NGRAPH_DEBUG << "cpu_memory_assignment: m_key_to_tensors_set_map:";
    for (auto& ele : m_key_to_tensors_set_map)
    {
        NGRAPH_DEBUG << "key : " << ele.first << "; label: " << find_role(ele.second.first)
                     << "; sets: {";
        for (auto& ele_t : ele.second.second)
        {
            NGRAPH_DEBUG << ele_t->get_name() << " ";
        }
        NGRAPH_DEBUG << "}";
        if (ele.second.first != CPUTensorRole::INTERMEDIATE)
        {
            // do not allocate and free memory for function inputs, outputs, constants and tensors
            // sharing memory with them.
            allocated_sets.insert(ele.first);
            freed_sets.insert(ele.first);
        }
    }

    // forward pass to build liveness_new_list
    for (auto it = ops.begin(); it != ops.end(); it++)
    {
        const shared_ptr<Node>& node = *it;
        node->liveness_new_list.clear();

        for (size_t i = 0; i < node->get_output_size(); ++i)
        {
            auto tensor = &node->get_output_tensor(i);
            auto key = m_tensor_to_key_map[tensor];
            if (allocated_sets.find(key) == allocated_sets.end())
            {
                node->liveness_new_list.insert(tensor);
                allocated_sets.insert(key);
            }
        }
    }

    // backward pass to build liveness_free_list
    for (auto it = ops.rbegin(); it != ops.rend(); it++)
    {
        const shared_ptr<Node>& node = *it;
        node->liveness_free_list.clear();

        for (descriptor::Input& input_decl : node->get_inputs())
        {
            auto tensor = &input_decl.get_tensor();
            auto key = m_tensor_to_key_map[tensor];
            if (freed_sets.find(key) == freed_sets.end())
            {
                node->liveness_free_list.insert(tensor);
                freed_sets.insert(key);
            }
        }
    }

    // memory assignment using liveness analysis result

    // memory manager for non-cacheable ops, memory allocation will be freed when not longer in use
    ngraph::pass::MemoryManager mm(m_alignment, m_disable_memory_sharing);
    // memory manager for cacheable ops, memory allocation will never be freed
    ngraph::pass::MemoryManager mm_caching(m_alignment, true);

    // reuse memory
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
    }

    for (shared_ptr<Node> node : function->get_ordered_ops())
    {
        if (node->is_parameter() || node->is_constant() || node->is_output())
        {
            continue;
        }
        // handle destructive oi pair
        unordered_set<descriptor::Tensor*> no_free;
        unordered_set<descriptor::Tensor*> no_new;

        if (node->is_op())
        {
            auto op = std::static_pointer_cast<op::Op>(node);
            if (auto op_annotations = op->get_op_annotations())
            {
                for (auto oi_pair : op_annotations->get_in_place_oi_pairs())
                {
                    auto output_tensor = &node->get_outputs().at(oi_pair.output).get_tensor();
                    auto input_tensor = &node->get_inputs().at(oi_pair.input).get_tensor();
                    auto input_node = node->get_inputs().at(oi_pair.input).get_output().get_node();

                    if (oi_pair.destructive && node->liveness_free_list.count(input_tensor) != 0 &&
                        node->liveness_new_list.count(output_tensor) != 0)
                    {
                        if (input_node->is_op())
                        {
                            auto input_op = std::static_pointer_cast<op::Op>(input_node);
                            if (auto input_op_annotations = input_op->get_op_annotations())
                            {
                                // when input is cacheable, do not allow destructive oi
                                if (input_op_annotations->is_cacheable())
                                {
                                    NGRAPH_DEBUG << "cpu_memory_assignment: cacheable input, no "
                                                    "destructive oi";
                                    continue;
                                }
                                // when reusing memory, ops with different cacheabilities are using different memory manager
                                // and should not share the same buffer.
                                else if (!m_disable_memory_sharing &&
                                         op_annotations->is_cacheable())
                                {
                                    continue;
                                }
                            }
                        }
                        NGRAPH_ASSERT(m_tensor_to_key_map.find(input_tensor) !=
                                      m_tensor_to_key_map.end());
                        auto input_key = m_tensor_to_key_map[input_tensor];
                        NGRAPH_ASSERT(m_tensor_to_key_map.find(output_tensor) !=
                                      m_tensor_to_key_map.end());
                        auto output_key = m_tensor_to_key_map[output_tensor];

                        NGRAPH_ASSERT(m_key_to_tensors_set_map.find(input_key) !=
                                      m_key_to_tensors_set_map.end());
                        // do not modify function inputs and constants, so no destructive oi
                        if (m_key_to_tensors_set_map[input_key].first == CPUTensorRole::INPUT ||
                            m_key_to_tensors_set_map[input_key].first == CPUTensorRole::CONSTANT)
                        {
                            NGRAPH_DEBUG << "cpu_memory_assignment: input is function input or "
                                            "constant, no destructive oi";
                            continue;
                        }
                        auto input_set = m_key_to_tensors_set_map[input_key].second;
                        // check buffer sizes, if required output buffer is larger than input buffer, do not reuse input buffer
                        // get the largest tensor size, which is the size of the memory buffer for the set
                        size_t input_size = input_tensor->size();
                        // get the smallest offset, which is the offset of the memory buffer for the set
                        size_t offset = input_tensor->get_pool_offset();
                        for (auto e : input_set)
                        {
                            if (e->size() > input_size)
                            {
                                input_size = e->size();
                            }
                            if (e->get_pool_offset() < offset)
                            {
                                offset = e->get_pool_offset();
                            }
                        }
                        NGRAPH_ASSERT(m_key_to_tensors_set_map.find(output_key) !=
                                      m_key_to_tensors_set_map.end());
                        auto output_set = m_key_to_tensors_set_map[output_key].second;
                        size_t output_size = input_tensor->size();
                        // get the largest tensor size, which is the size of memory buffer for the set
                        for (auto e : output_set)
                        {
                            if (e->size() > output_size)
                            {
                                output_size = e->size();
                            }
                        }
                        if (input_size < output_size)
                        {
                            continue;
                        }
                        NGRAPH_DEBUG << "cpu_memory_assignment: last use of input tensor, "
                                        "destructive oi allowed:";
                        NGRAPH_DEBUG << "input_tensor is " << input_tensor->get_name();
                        NGRAPH_DEBUG << "output_tensor is " << output_tensor->get_name();
                        no_free.insert(input_tensor);
                        no_new.insert(output_tensor);

                        // set the tensor offset for tensors in the set containing the output tensor to the starting offset
                        // of the set of input tensor.
                        // do not combine those two sets.
                        // change the label of output tensor set to that of input tensor set
                        m_key_to_tensors_set_map[output_key].first =
                            m_key_to_tensors_set_map[input_key].first;
                        for (auto& ele_t : m_key_to_tensors_set_map[output_key].second)
                        {
                            ele_t->set_pool_offset(offset);
                        }
                    }
                }
            }
        }

        for (descriptor::Tensor* tensor : node->liveness_new_list)
        {
            if (no_new.find(tensor) != no_new.end())
            {
                continue;
            }
            size_t offset = 0;
            NGRAPH_ASSERT(m_tensor_to_key_map.find(tensor) != m_tensor_to_key_map.end());
            auto key = m_tensor_to_key_map[tensor];
            NGRAPH_ASSERT(m_key_to_tensors_set_map.find(key) != m_key_to_tensors_set_map.end());
            auto tensors_set = m_key_to_tensors_set_map[key].second;
            size_t size = tensor->size();
            for (auto e : tensors_set)
            {
                if (e->size() > size)
                {
                    size = e->size();
                }
            }
            if (m_tensor_caching.count(tensor) != 0)
            {
                offset = mm_caching.allocate(size);
            }
            else
            {
                offset = mm.allocate(size);
            }
            tensor->set_pool_offset(offset);
            for (auto& e : tensors_set)
            {
                e->set_pool_offset(offset);
            }
        }

        // when reusing memory, free when done
        if (!m_disable_memory_sharing && node->is_op())
        {
            for (descriptor::Tensor* tensor : node->liveness_free_list)
            {
                if (no_free.find(tensor) != no_free.end())
                {
                    continue;
                }
                if (m_tensor_caching.empty() ||
                    (!m_tensor_caching.empty() && m_tensor_caching.count(tensor) == 0))
                {
                    mm.free(tensor->get_pool_offset());
                }
            }
        }
    }

    // update offsets in concat and slice tensors set.
    // In place concatenation optimization
    process_in_place_concat(ops);

    // In place slice optimization
    process_in_place_slice(ops);

    //update the offset for intermediate tensors in tensor_caching
    auto start = mm.max_allocated();
    for (auto item : m_tensor_caching)
    {
        NGRAPH_ASSERT(m_tensor_to_key_map.find(item) != m_tensor_to_key_map.end());
        auto key = m_tensor_to_key_map[item];
        NGRAPH_ASSERT(m_key_to_tensors_set_map.find(key) != m_key_to_tensors_set_map.end());
        if (m_key_to_tensors_set_map[key].first == CPUTensorRole::INTERMEDIATE)
        {
            auto new_offset = item->get_pool_offset() + start;
            item->set_pool_offset(new_offset);
        }
    }

    NGRAPH_DEBUG << "cpu_memory_assignemnt: max allocated for mm is " << mm.max_allocated();
    NGRAPH_DEBUG << "cpu_memory_assignment: max allocated for mm_caching is "
                 << mm_caching.max_allocated();
    NGRAPH_DEBUG << "cpu_memory_assignment: max allocated in total is "
                 << mm.max_allocated() + mm_caching.max_allocated();

    function->set_temporary_pool_size(mm.max_allocated() + mm_caching.max_allocated());

    return false;
}
