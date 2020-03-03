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
    unordered_map<size_t, std::pair<TensorRole, unordered_set<descriptor::Tensor*>>>&
        bufferID_to_tensorSets,
    unordered_map<descriptor::Tensor*, size_t>& tensor_to_bufferID,
    size_t alignment,
    bool disable_memory_sharing)
    : m_alignment(alignment)
    , m_disable_memory_sharing(disable_memory_sharing)
    , m_bufferID_to_tensorSets(bufferID_to_tensorSets)
    , m_tensor_to_bufferID(tensor_to_bufferID)
{
    if (m_alignment == 0)
    {
        throw invalid_argument("Memory alignment must be > 0");
    }
}

size_t runtime::cpu::pass::CPUMemoryAssignment::get_bufferID(descriptor::Tensor* tensor)
{
    auto tensor_it = m_tensor_to_bufferID.find(tensor);
    NGRAPH_CHECK(tensor_it != m_tensor_to_bufferID.end());
    return tensor_it->second;
}

void runtime::cpu::pass::CPUMemoryAssignment::process_in_place_concat(
    std::vector<std::shared_ptr<Node>> nodes)
{
    for (shared_ptr<Node> node : nodes)
    {
        if (auto concat = as_type_ptr<op::Concat>(node))
        {
            if (auto op_annotations = concat->get_op_annotations())
            {
                auto in_place_oi_pairs = op_annotations->get_in_place_oi_pairs();
                if (in_place_oi_pairs.size() > 0)
                {
                    // check if it is the last in place concat
                    bool found_last_concat = true;
                    for (auto user : concat->get_users())
                    {
                        if (auto user_concat = as_type_ptr<op::Concat>(user))
                        {
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
                        auto output_tensor = &concat->output(0).get_tensor();
                        auto output_bufferID = get_bufferID(output_tensor);

                        auto offset = output_tensor->get_pool_offset();
                        size_t arg_index = 0;
                        for (auto arg : concat->input_values())
                        {
                            auto input_tensor = &arg.get_tensor();
                            auto input_bufferID = get_bufferID(input_tensor);
                            // same set, in place concat allowed
                            if (input_bufferID == output_bufferID)
                            {
                                auto old_offset = input_tensor->get_pool_offset();
                                input_tensor->set_pool_offset(offset);
                                NGRAPH_DEBUG
                                    << "cpu_memory_assignment: change offset, old offset is "
                                    << old_offset << ", new offset is " << offset << std::endl;

                                // check if need to propagate backward
                                auto arg_op = arg.get_node_shared_ptr();
                                if (auto arg_op_annotations = arg_op->get_op_annotations())
                                {
                                    auto arg_in_place_oi_pairs =
                                        arg_op_annotations->get_in_place_oi_pairs();
                                    if (arg_in_place_oi_pairs.size() > 0)
                                    {
                                        NGRAPH_DEBUG << "cpu_memory_assignment: call "
                                                        "propagate_in_place_concat for "
                                                     << *arg_op;
                                        propagate_in_place_concat(arg);
                                    }
                                }
                            }
                            offset += input_tensor->size();
                            arg_index++;
                        }
                    }
                }
            }
        }
    }
}

void runtime::cpu::pass::CPUMemoryAssignment::propagate_in_place_concat(const Output<Node>& output)
{
    auto op = output.get_node_shared_ptr();
    if (is_type<op::Concat>(op))
    {
        auto output_tensor = &op->output(0).get_tensor();
        auto output_bufferID = get_bufferID(output_tensor);

        auto offset = output_tensor->get_pool_offset();
        size_t arg_index = 0;
        for (auto arg : op->input_values())
        {
            auto input_tensor = &arg.get_tensor();
            auto input_bufferID = get_bufferID(input_tensor);
            // same set, in place concat allowed
            if (input_bufferID == output_bufferID)
            {
                auto old_offset = input_tensor->get_pool_offset();
                input_tensor->set_pool_offset(offset);
                NGRAPH_DEBUG << "cpu_memory_assignment: change offset, old offset is " << old_offset
                             << ", new offset is " << offset;

                // check if need to propagate backward
                auto arg_op = arg.get_node_shared_ptr();
                if (auto arg_op_annotations = arg_op->get_op_annotations())
                {
                    auto arg_in_place_oi_pairs = arg_op_annotations->get_in_place_oi_pairs();
                    if (arg_in_place_oi_pairs.size() > 0)
                    {
                        NGRAPH_DEBUG << "cpu_memory_assignment: call propagate_in_place_concat for "
                                     << *arg_op;
                        propagate_in_place_concat(arg);
                    }
                }
            }
            offset += input_tensor->size();
            arg_index++;
        }
    }
    else
    {
        // other in place ops
        auto op_annotations = op->get_op_annotations();
        for (auto oi_pair : op_annotations->get_in_place_oi_pairs())
        {
            if (oi_pair.output != output.get_index() || oi_pair.destructive)
            {
                continue;
            }

            auto input_tensor = &op->input_value(oi_pair.input).get_tensor();
            auto input_bufferID = get_bufferID(input_tensor);
            auto output_tensor = &op->output(oi_pair.output).get_tensor();
            auto output_bufferID = get_bufferID(output_tensor);

            // same set, in place op allowed
            if (input_bufferID == output_bufferID)
            {
                auto old_offset = input_tensor->get_pool_offset();
                auto new_offset = output_tensor->get_pool_offset();
                input_tensor->set_pool_offset(new_offset);
                NGRAPH_DEBUG << "cpu_memory_assignment: change offset, old offset is " << old_offset
                             << ", new offset is " << new_offset;
                auto input = op->input_value(oi_pair.input);
                auto arg_op = input.get_node_shared_ptr();

                // check if need to propagate backward
                if (auto arg_op_annotations = arg_op->get_op_annotations())
                {
                    auto arg_in_place_oi_pairs = arg_op_annotations->get_in_place_oi_pairs();
                    if (arg_in_place_oi_pairs.size() > 0)
                    {
                        NGRAPH_DEBUG << "cpu_memory_assignment: call propagate_in_place_concat for "
                                     << *arg_op;
                        propagate_in_place_concat(input);
                    }
                }
            }
        }
    }
}

// slice
void runtime::cpu::pass::CPUMemoryAssignment::process_in_place_slice(
    std::vector<std::shared_ptr<Node>> nodes)
{
    for (shared_ptr<Node>& node : nodes)
    {
        if (auto slice = as_type_ptr<op::Slice>(node))
        {
            if (auto op_annotations = slice->get_op_annotations())
            {
                auto in_place_oi_pairs = op_annotations->get_in_place_oi_pairs();
                if (in_place_oi_pairs.size() > 0)
                {
                    auto input = slice->input_value(0);
                    auto arg = input.get_node_shared_ptr();
                    auto input_tensor = &input.get_tensor();
                    auto input_bufferID = get_bufferID(input_tensor);
                    auto output_tensor = &slice->output(0).get_tensor();
                    auto output_bufferID = get_bufferID(output_tensor);

                    // same set, in place slice allowed
                    if (input_bufferID == output_bufferID)
                    {
                        NGRAPH_CHECK(m_bufferID_to_tensorSets.find(output_bufferID) !=
                                     m_bufferID_to_tensorSets.end());
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
                        for (auto slice_output : slice->outputs())
                        {
                            for (auto slice_output_input : slice_output.get_target_inputs())
                            {
                                NGRAPH_DEBUG
                                    << "cpu_memory_assignment: call propagate_in_place_slice "
                                       "for output "
                                    << slice_output.get_index() << " of " << *slice;
                                propagate_in_place_slice(slice_output_input);
                            }
                        }
                    }
                }
            }
        }
    }
}

void runtime::cpu::pass::CPUMemoryAssignment::propagate_in_place_slice(const Input<Node>& input)
{
    std::deque<Input<Node>> stack;
    stack.push_front(input);

    while (stack.size() > 0)
    {
        Input<Node> in = stack.front();
        stack.pop_front();

        auto node = in.get_node();
        // let process_in_place_slice handle slice.
        if (is_type<op::Slice>(node))
        {
            continue;
        }
        if (auto op_annotations = node->get_op_annotations())
        {
            for (auto oi_pair : op_annotations->get_in_place_oi_pairs())
            {
                if (oi_pair.input == in.get_index())
                {
                    auto input_tensor = &node->input(oi_pair.input).get_tensor();
                    auto input_bufferID = get_bufferID(input_tensor);
                    size_t output_index = oi_pair.output;
                    auto output_tensor = &node->output(output_index).get_tensor();
                    auto output_bufferID = get_bufferID(output_tensor);

                    // same set, in place op allowed
                    if (input_bufferID == output_bufferID)
                    {
                        output_tensor->set_pool_offset(input_tensor->get_pool_offset());
                        for (auto op_output : node->outputs())
                        {
                            for (auto op_output_input : op_output.get_target_inputs())
                            {
                                stack.push_front(op_output_input);
                            }
                        }
                    }
                }
            }
        }
    }
}

// This function processes each node and puts its output tensors into one buffer set accordingly.
// All the tensors in the same buffer set share the same memory buffer.
// Output tensor is put into the set of input tensor when the operation is non-destructive in-place.
// If the operation is destructive in-place or not in-place, a new buffer set is created for the
// output tensor. Each buffer set has a bufferID which starts at 0 and increments by 1 each time a
// new set is created. bufferID_to_tensorSets maps bufferID to the pair of TensorRole and buffer
// set. TensorRole is INPUT, CONSTANT, OUTPUT, or INTERMEDIATE, which tells from where the memory
// buffer comes. tensor_to_bufferID maps tensor to the ID of the buffer set it belongs to.
void runtime::cpu::pass::CPUMemoryAssignment::build_buffer_sets_maps(vector<shared_ptr<Node>>& ops)
{
    unordered_set<descriptor::Tensor*> in_place_slice_chain;
    size_t count = 0;
    for (auto it = ops.begin(); it != ops.end(); it++)
    {
        const shared_ptr<Node>& node = *it;
        if (node->is_parameter())
        {
            auto output_tensor = &node->output(0).get_tensor();
            auto ele = std::pair<TensorRole, unordered_set<descriptor::Tensor*>>(
                TensorRole::INPUT, unordered_set<descriptor::Tensor*>({output_tensor}));
            m_bufferID_to_tensorSets[count] = ele;
            m_tensor_to_bufferID[output_tensor] = count;
            count++;
        }
        else if (is_type<op::Constant>(node))
        {
            auto output_tensor = &node->output(0).get_tensor();
            auto ele = std::pair<TensorRole, unordered_set<descriptor::Tensor*>>(
                TensorRole::CONSTANT, unordered_set<descriptor::Tensor*>({output_tensor}));
            m_bufferID_to_tensorSets[count] = ele;
            m_tensor_to_bufferID[output_tensor] = count;
            count++;
        }
        else if (node->is_output())
        {
            auto output_tensor = &node->output(0).get_tensor();
            auto input_tensor = &node->input(0).get_tensor();
            auto bufferID = get_bufferID(input_tensor);
            NGRAPH_CHECK(bufferID <= count);

            auto input_buffer_it = m_bufferID_to_tensorSets.find(bufferID);
            NGRAPH_CHECK(input_buffer_it != m_bufferID_to_tensorSets.end());
            auto pair = input_buffer_it->second;
            if (pair.first != TensorRole::INTERMEDIATE ||
                in_place_slice_chain.find(input_tensor) != in_place_slice_chain.end())
            {
                // tensor of function output should not be in the same set as function input,
                // constant, output, or in place slice, because they cannot share the same memory
                // buffer
                auto ele = std::pair<TensorRole, unordered_set<descriptor::Tensor*>>(
                    TensorRole::OUTPUT, unordered_set<descriptor::Tensor*>({output_tensor}));
                m_bufferID_to_tensorSets[count] = ele;
                m_tensor_to_bufferID[output_tensor] = count;
                count++;
            }
            else
            {
                // in place output
                m_bufferID_to_tensorSets[bufferID].first = TensorRole::OUTPUT;
                m_bufferID_to_tensorSets[bufferID].second.insert(output_tensor);
                m_tensor_to_bufferID[output_tensor] = bufferID;
            }
        }
        else
        {
            NGRAPH_CHECK(node->is_op());
            auto op = std::static_pointer_cast<op::Op>(node);
            if (auto op_annotations = op->get_op_annotations())
            {
                auto in_place_oi_pairs = op_annotations->get_in_place_oi_pairs();
                if (in_place_oi_pairs.size() > 0)
                {
                    auto cacheable = op_annotations->is_cacheable();

                    // in place concat
                    if (is_type<op::Concat>(node))
                    {
                        auto output_tensor = &node->output(0).get_tensor();
                        auto ele = std::pair<TensorRole, unordered_set<descriptor::Tensor*>>(
                            TensorRole::INTERMEDIATE,
                            unordered_set<descriptor::Tensor*>({output_tensor}));
                        for (auto& arg : node->input_values())
                        {
                            // when reusing memory, check cacheability
                            if (!m_disable_memory_sharing)
                            {
                                auto arg_op = arg.get_node_shared_ptr();
                                if (auto arg_op_annotations = arg_op->get_op_annotations())
                                {
                                    // when reusing memory, ops with different cacheabilities should
                                    // not be in the same set.
                                    if (cacheable != arg_op_annotations->is_cacheable())
                                    {
                                        continue;
                                    }
                                }
                            }
                            // no in-place concat if arg is in in_place_slice_chain,
                            // because in-place slice before in-place concat cannot use the memory
                            // buffer of concat. In-place slice after in-place concat can use the
                            // memory buffer of concat.
                            auto input_tensor = &arg.get_tensor();
                            if (in_place_slice_chain.find(input_tensor) !=
                                in_place_slice_chain.end())
                            {
                                NGRAPH_DEBUG << "cpu_memory_assignment: no in place concat after "
                                                "in place slice";
                                continue;
                            }

                            auto bufferID = get_bufferID(input_tensor);

                            // tensor set is erased from the map when processing previous arg
                            if (m_bufferID_to_tensorSets.find(bufferID) ==
                                m_bufferID_to_tensorSets.end())
                            {
                                continue;
                            }
                            auto pair = m_bufferID_to_tensorSets[bufferID];
                            // no in-place concat if arg is from parameter or constant
                            if (pair.first == TensorRole::INPUT ||
                                pair.first == TensorRole::CONSTANT)
                            {
                                continue;
                            }
                            // in-place concat
                            // move tensors in the set containing the input tensor to the set of
                            // output tensor
                            // then erase that input tensor set
                            for (auto tensor : pair.second)
                            {
                                m_tensor_to_bufferID[tensor] = count;
                                ele.second.insert(tensor);
                            }
                            m_bufferID_to_tensorSets.erase(bufferID);
                        }
                        // put the set containing output tensor into the map
                        m_bufferID_to_tensorSets[count] = ele;
                        m_tensor_to_bufferID[output_tensor] = count;
                        count++;
                    }
                    else
                    {
                        // other in place ops
                        for (auto oi_pair : op_annotations->get_in_place_oi_pairs())
                        {
                            auto input_tensor = &node->input_value(oi_pair.input).get_tensor();
                            auto output_tensor = &node->output(oi_pair.output).get_tensor();

                            // if destructive, do not put input tensor and output tensor into the
                            // same set.
                            if (!oi_pair.destructive)
                            {
                                bool no_in_place = false;
                                auto input_op =
                                    node->input_value(oi_pair.input).get_node_shared_ptr();
                                // when reusing memory, check cacheability
                                if (!m_disable_memory_sharing)
                                {
                                    if (auto input_op_annotations = input_op->get_op_annotations())
                                    {
                                        // when reusing memory, ops with different cacheabilities
                                        // should not be in the same set.
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
                                    auto bufferID = get_bufferID(input_tensor);
                                    auto input_buffer_it = m_bufferID_to_tensorSets.find(bufferID);
                                    NGRAPH_CHECK(input_buffer_it != m_bufferID_to_tensorSets.end());

                                    if (is_type<op::Slice>(node))
                                    {
                                        if (input_buffer_it->second.first != TensorRole::CONSTANT)
                                        {
                                            // build in place slice chain
                                            in_place_slice_chain.insert(output_tensor);
                                            input_buffer_it->second.second.insert(output_tensor);
                                            m_tensor_to_bufferID[output_tensor] = bufferID;
                                        }
                                    }
                                    else
                                    {
                                        // keep track of tensors sharing the memory buffer with
                                        // in-place slice output tensor
                                        if (in_place_slice_chain.find(input_tensor) !=
                                            in_place_slice_chain.end())
                                        {
                                            in_place_slice_chain.insert(output_tensor);
                                        }
                                        input_buffer_it->second.second.insert(output_tensor);
                                        m_tensor_to_bufferID[output_tensor] = bufferID;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // process output tensors
            for (auto node_output : node->outputs())
            {
                auto output_tensor = &node_output.get_tensor();
                // not in place, create a new set and insert into the map
                if (m_tensor_to_bufferID.find(output_tensor) == m_tensor_to_bufferID.end())
                {
                    auto ele = std::pair<TensorRole, unordered_set<descriptor::Tensor*>>(
                        TensorRole::INTERMEDIATE,
                        unordered_set<descriptor::Tensor*>({output_tensor}));
                    m_bufferID_to_tensorSets[count] = ele;
                    m_tensor_to_bufferID[output_tensor] = count;
                    count++;
                }
            }
        }
    }
}

void runtime::cpu::pass::CPUMemoryAssignment::liveness_analysis(
    std::vector<std::shared_ptr<Node>>& ops)
{
    auto find_role = [](TensorRole tensor_role) -> string {
        switch (tensor_role)
        {
        case TensorRole::INPUT: return string("TensorRole::INPUT");
        case TensorRole::INTERMEDIATE: return string("TensorRole::INTERMEDIATE");
        case TensorRole::CONSTANT: return string("TensorRole::CONSTANT");
        case TensorRole::OUTPUT: return string("TensorRole::OUTPUT");
        case TensorRole::UNKNOWN:
        default: throw runtime_error("unhandled CPU tensor role");
        }
    };

    // liveness analysis
    unordered_set<size_t> allocated_sets;
    unordered_set<size_t> freed_sets;
    NGRAPH_DEBUG << "cpu_memory_assignment: m_bufferID_to_tensorSets:";
    for (auto& ele : m_bufferID_to_tensorSets)
    {
        NGRAPH_DEBUG << "bufferID : " << ele.first << "; label: " << find_role(ele.second.first)
                     << "; sets: {";
        for (auto& ele_t : ele.second.second)
        {
            NGRAPH_DEBUG << ele_t->get_name() << " ";
        }
        NGRAPH_DEBUG << "}";
        if (ele.second.first != TensorRole::INTERMEDIATE)
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

        for (auto node_output : node->outputs())
        {
            auto tensor = &node_output.get_tensor();
            auto bufferID = get_bufferID(tensor);
            if (allocated_sets.find(bufferID) == allocated_sets.end())
            {
                node->liveness_new_list.insert(tensor);
                allocated_sets.insert(bufferID);
            }
        }
    }

    // backward pass to build liveness_free_list
    for (auto it = ops.rbegin(); it != ops.rend(); it++)
    {
        const shared_ptr<Node>& node = *it;
        node->liveness_free_list.clear();

        for (auto input_value : node->input_values())
        {
            auto tensor = &input_value.get_tensor();
            auto bufferID = get_bufferID(tensor);
            if (freed_sets.find(bufferID) == freed_sets.end())
            {
                node->liveness_free_list.insert(tensor);
                freed_sets.insert(bufferID);
            }
        }
    }
}

bool runtime::cpu::pass::CPUMemoryAssignment::run_on_function(shared_ptr<ngraph::Function> function)
{
    auto ops = function->get_ordered_ops();

    build_buffer_sets_maps(ops);
    liveness_analysis(ops);

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
                        for (auto output : node->outputs())
                        {
                            m_tensor_caching.insert(&output.get_tensor());
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

        auto op = std::static_pointer_cast<op::Op>(node);
        if (auto op_annotations = op->get_op_annotations())
        {
            for (auto oi_pair : op_annotations->get_in_place_oi_pairs())
            {
                auto output_tensor = &node->output(oi_pair.output).get_tensor();
                auto input_tensor = &node->input_value(oi_pair.input).get_tensor();
                auto input_op = node->input_value(oi_pair.input).get_node_shared_ptr();

                if (oi_pair.destructive && node->liveness_free_list.count(input_tensor) != 0 &&
                    node->liveness_new_list.count(output_tensor) != 0)
                {
                    if (auto input_op_annotations = input_op->get_op_annotations())
                    {
                        // when reusing memory, ops with different cacheabilities are using
                        // different memory manager
                        // and should not share the same buffer.
                        if (!m_disable_memory_sharing &&
                            input_op_annotations->is_cacheable() != op_annotations->is_cacheable())
                        {
                            NGRAPH_DEBUG << "cpu_memory_assignment: reusing memory with "
                                            "input and output have different cacheabilities, no "
                                            "destructive oi";
                            continue;
                        }
                    }

                    auto input_bufferID = get_bufferID(input_tensor);
                    auto output_bufferID = get_bufferID(output_tensor);

                    auto input_buffer_it = m_bufferID_to_tensorSets.find(input_bufferID);
                    NGRAPH_CHECK(input_buffer_it != m_bufferID_to_tensorSets.end());
                    // do not modify function inputs and constants, so no destructive oi
                    if (input_buffer_it->second.first == TensorRole::INPUT ||
                        input_buffer_it->second.first == TensorRole::CONSTANT)
                    {
                        NGRAPH_DEBUG << "cpu_memory_assignment: input is function input or "
                                        "constant, no destructive oi";
                        continue;
                    }
                    auto input_set = input_buffer_it->second.second;
                    // check buffer sizes, if required output buffer is larger than input
                    // buffer, do not reuse input buffer get the largest tensor size, which is
                    // the size of the memory buffer for the set
                    size_t input_size = input_tensor->size();
                    // get the smallest offset, which is the offset of the memory buffer for the
                    // set
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
                    auto output_buffer_it = m_bufferID_to_tensorSets.find(output_bufferID);
                    NGRAPH_CHECK(output_buffer_it != m_bufferID_to_tensorSets.end());
                    auto output_set = output_buffer_it->second.second;
                    size_t output_size = input_tensor->size();
                    // get the largest tensor size, which is the size of memory buffer for the
                    // set
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

                    // set the tensor offset for tensors in the set containing the output tensor
                    // to the starting offset
                    // of the set of input tensor.
                    // do not combine those two sets.
                    // change the label of output tensor set to that of input tensor set
                    output_buffer_it->second.first = input_buffer_it->second.first;
                    for (auto& ele_t : output_set)
                    {
                        ele_t->set_pool_offset(offset);
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
            auto bufferID = get_bufferID(tensor);
            auto buffer_it = m_bufferID_to_tensorSets.find(bufferID);
            NGRAPH_CHECK(buffer_it != m_bufferID_to_tensorSets.end());
            auto tensor_set = buffer_it->second.second;
            size_t size = tensor->size();
            for (auto e : tensor_set)
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
            for (auto& e : tensor_set)
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

    // update the offset for intermediate tensors in tensor_caching
    auto start = mm.max_allocated();
    for (auto item : m_tensor_caching)
    {
        auto bufferID = get_bufferID(item);
        auto buffer_it = m_bufferID_to_tensorSets.find(bufferID);
        NGRAPH_CHECK(buffer_it != m_bufferID_to_tensorSets.end());

        if (buffer_it->second.first == TensorRole::INTERMEDIATE)
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
