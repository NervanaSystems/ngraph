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
#include "ngraph/op/slice.hpp"
#include "ngraph/pass/liveness.hpp"
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
    // memory manager for non-cacheable ops, memory allocation will be freed when not longer in use
    ngraph::pass::MemoryManager mm(m_alignment, m_disable_memory_sharing);
    // memory manager for cacheable ops, memory allocation will never be freed
    ngraph::pass::MemoryManager mm_caching(m_alignment, true);

    // Tensors should not be freed due to the following reasons:
    // Several tensors may have the same offset because of in place propagation, only one of them should be freed.
    // Tensors get offset 0 from parameter or constant due to in place propagation and should never be freed.
    std::set<const descriptor::Tensor*> io_no_free;

    if (!m_disable_memory_sharing)
    {
        // Set of tensors in the chain of in place propagation ops where the beginning of the chain is parameter, constant,
        // or a node whose output has multiple users.
        std::set<const descriptor::Tensor*> io_from_param_or_const_or_multi;
        // build caching map from cacheability
        for (shared_ptr<Node> node : function->get_ordered_ops())
        {
            if (node->is_op())
            {
                auto op = std::static_pointer_cast<op::Op>(node);
                auto op_annotations = op->get_op_annotations();
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

        // add other tensors due to in place propagation
        for (shared_ptr<Node> node : function->get_ordered_ops())
        {
            if (node->is_op())
            {
                auto op = std::static_pointer_cast<op::Op>(node);
                auto op_annotations = op->get_op_annotations();

                //if it is last concat node of in-place-concat-chain, put it in caching map
                auto in_place_oi_pairs = op_annotations->get_in_place_oi_pairs();
                if (in_place_oi_pairs.size() > 0)
                {
                    if (auto concat = std::dynamic_pointer_cast<ngraph::op::Concat>(node))
                    {
                        auto cpu_op_annotations =
                            std::static_pointer_cast<runtime::cpu::CPUOpAnnotations>(
                                op_annotations);
                        if (!cpu_op_annotations->can_free_memory())
                        {
                            shared_ptr<descriptor::Tensor> tv = node->get_output_tensor_ptr(0);
                            m_tensor_caching.insert(tv.get());
                        }
                    }
                    else
                    {
                        for (auto oi_pair : op_annotations->get_in_place_oi_pairs())
                        {
                            auto output = &node->get_outputs().at(oi_pair.output).get_tensor();
                            auto input = &node->get_inputs().at(oi_pair.input).get_tensor();
                            auto input_node =
                                node->get_inputs().at(oi_pair.input).get_output().get_node();

                            if ((node->liveness_free_list.count(input) != 0 ||
                                 !oi_pair.destructive) &&
                                node->liveness_new_list.count(output) != 0)

                            {
                                // input tensor and output tensor have the same offset, should not free input tensor, only free output tensor.
                                io_no_free.insert(input);

                                auto input_output_inputs =
                                    node->get_inputs().at(oi_pair.input).get_output().get_inputs();
                                // tensors in the chain of in place propagation ops where the beginning of the chain is parameter, constant,
                                // or a node whose output has multiple users. Those tensors should not be freed.
                                if (input_node->is_parameter() || input_node->is_constant() ||
                                    input_output_inputs.size() > 1 ||
                                    io_from_param_or_const_or_multi.count(input))
                                {
                                    io_no_free.insert(output);
                                    io_from_param_or_const_or_multi.insert(output);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    for (shared_ptr<Node> node : function->get_ordered_ops())
    {
        std::map<descriptor::Tensor*, descriptor::Tensor*> in_place_outputs;
        std::set<const descriptor::Tensor*> reused_inputs;

        if (node->is_op())
        {
            auto op = std::static_pointer_cast<op::Op>(node);
            // concat in_place_oi should be treated differently
            if (!std::dynamic_pointer_cast<op::Concat>(node))
            {
                if (auto op_annotations = op->get_op_annotations())
                {
                    for (auto oi_pair : op_annotations->get_in_place_oi_pairs())
                    {
                        auto output = &node->get_outputs().at(oi_pair.output).get_tensor();
                        auto input = &node->get_inputs().at(oi_pair.input).get_tensor();
                        auto input_node =
                            node->get_inputs().at(oi_pair.input).get_output().get_node();

                        // For destructive kernel, this should be the last use
                        // Non-destructive kernels can pass through

                        if ((node->liveness_free_list.count(input) != 0 || !oi_pair.destructive) &&
                            node->liveness_new_list.count(output) != 0)

                        {
                            in_place_outputs.insert({output, input});
                            reused_inputs.insert(input);
                        }
                    }
                }
            }
        }

        for (descriptor::Tensor* tensor : node->liveness_new_list)
        {
            size_t offset = 0;
            if (in_place_outputs.count(tensor))
            {
                offset = in_place_outputs.at(tensor)->get_pool_offset();
                // For input tensor and output tensor pair,
                // if the input tensor is not cacheable, the output tensor is not cacheable;
                // if the input tensor is cacheable, the output tensor could be cacaheable or not. If the output tensor is not cacheable,
                // need to put output tensor into the caching map for in place op. Otherwise, mm will try to free memory allocated by mm_caching.
                // For example, suppose we have op1 \
                //                              op2 - op3, op1 is cacheable, op2 and op3 are not. Due to in place propagation, op3 gets the offset
                // from op1, which is 1000 allocated by mm_caching. There is another 1000 allocated to op4 by mm. If op3 is not put into caching
                // map, mm will be called to free 1000, which is allocated to op4.
                if (m_tensor_caching.count(in_place_outputs.at(tensor)))
                {
                    m_tensor_caching.insert(tensor);
                }
            }
            else if (m_tensor_caching.count(tensor) != 0)
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
                if (reused_inputs.count(tensor) == 0 && io_no_free.count(tensor) == 0 &&
                    (m_tensor_caching.empty() ||
                     (!m_tensor_caching.empty() && m_tensor_caching.count(tensor) == 0)))
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

    NGRAPH_DEBUG << "cpu_memory_assignemnt: max allocated for mm is " << mm.max_allocated();
    NGRAPH_DEBUG << "cpu_memory_assignment: max allocated for mm_caching is "
                 << mm_caching.max_allocated();
    NGRAPH_DEBUG << "cpu_memory_assignment: max allocated in total is "
                 << mm.max_allocated() + mm_caching.max_allocated();

    function->set_temporary_pool_size(mm.max_allocated() + mm_caching.max_allocated());

    return false;
}
