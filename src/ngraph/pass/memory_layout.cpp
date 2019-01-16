//*****************************************************************************
// Copyright 2017-2019 Intel Corporation
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
#include <execinfo.h>
#include <iomanip>
#include <sstream>

#include "ngraph/log.hpp"
#include "ngraph/log.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/pass/liveness.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/memory_layout.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

static stringstream ss_verbose;
static stringstream ss_concise;

void print_trace(void)
{
    void* array[10];
    size_t size;
    char** strings;
    size_t i;

    size = backtrace(array, 10);
    strings = backtrace_symbols(array, size);

    printf("Obtained %zd stack frames.\n", size);

    for (i = 0; i < size; i++)
        printf("%s\n", strings[i]);

    free(strings);
}

pass::MemoryLayout::MemoryLayout(size_t alignment, bool disable_memory_sharing)
    : m_alignment(alignment)
    , m_disable_memory_sharing(disable_memory_sharing)
{
    if (m_alignment == 0)
    {
        throw invalid_argument("Memory alignment must be > 0");
    }
}

bool pass::MemoryLayout::run_on_function(shared_ptr<ngraph::Function> function)
{
    MemoryManager mm(m_alignment, m_disable_memory_sharing);

    size_t total_mem_alloc_all_nodes_requested = 0;
    size_t total_mem_alloc_all_nodes_actual = 0;
    size_t total_mem_free_all_nodes = 0;

    std::set<size_t> sizes_requested;
    std::map<size_t, size_t> size_num_nodes;
    std::map<size_t, std::pair<size_t, std::set<std::string>>> size_nodes_adding_to_maxalloc;

    for (shared_ptr<Node> node : function->get_ordered_ops())
    {
        std::map<descriptor::Tensor*, descriptor::Tensor*> in_place_outputs;
        std::set<const descriptor::Tensor*> reused_inputs;

        if (node->is_op())
        {
            auto op = std::static_pointer_cast<op::Op>(node);
            // concat and slice in_place_oi should be treated differently
            if (!std::dynamic_pointer_cast<op::Concat>(node) &&
                !std::dynamic_pointer_cast<op::Slice>(node))
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
                        // Non-destructive kernels can pass through if memory sharing is disabled
                        if ((node->liveness_free_list.count(input) != 0 ||
                             std::dynamic_pointer_cast<op::GetOutputElement>(node) ||
                             (m_disable_memory_sharing && !oi_pair.destructive)) &&
                            node->liveness_new_list.count(output) != 0)
                        {
                            in_place_outputs.insert({output, input});
                            reused_inputs.insert(input);
                        }
                    }
                }
            }
        }

        size_t mem_alloc_requested = 0;
        size_t mem_alloc_actual = 0;
        string str_;
        for (descriptor::Tensor* tensor : node->liveness_new_list)
        {
            size_t is_inplace = 0;
            if (in_place_outputs.count(tensor))
            {
                str_ = "INPLACE for tensor " + tensor->get_name();
            }
            else
            {
                str_ = "ALLOCATE for tensor " + tensor->get_name();
                is_inplace = 1;
            }
            ss_verbose << "- Node " << node->get_name() << " has requested to allocate "
                       << std::setprecision(5)
                       << static_cast<float>(tensor->size() / 1024.0f / 1024.0f) << " MiB "
                       << "for tensor " << tensor->get_name() << " of shape ("
                       << join(tensor->get_shape()) << ")" << std::endl;
            sizes_requested.insert(tensor->size());
            if (size_num_nodes.find(tensor->size()) == size_num_nodes.end())
            {
                size_num_nodes.insert(std::make_pair(tensor->size(), 1));
            }
            else
            {
                size_num_nodes[tensor->size()]++;
            }
            mem_alloc_requested += tensor->size();
            mem_alloc_actual += is_inplace * tensor->size();

            ss_verbose << str_ << std::endl;
            size_t offset = in_place_outputs.count(tensor)
                                ? in_place_outputs.at(tensor)->get_pool_offset()
                                : mm.allocate(tensor->size(), node, size_nodes_adding_to_maxalloc);
            tensor->set_pool_offset(offset);
        }

        ss_verbose << "->Total Memory alloc requested by node " << node->get_name() << " = "
                   << static_cast<float>(mem_alloc_requested) / 1024.0f / 1024.0f << " MiB"
                   << std::endl;
        ss_verbose << "->Total Memory actually requested to allocate by node " << node->get_name()
                   << " = " << static_cast<float>(mem_alloc_actual) / 1024.0f / 1024.0f << " MiB"
                   << std::endl;

        total_mem_alloc_all_nodes_requested += mem_alloc_requested;
        total_mem_alloc_all_nodes_actual += mem_alloc_actual;

        size_t mem_free = 0;
        if (!m_disable_memory_sharing)
        {
            for (const descriptor::Tensor* tensor : node->liveness_free_list)
            {
                if (reused_inputs.count(tensor) == 0)
                {
                    ss_verbose << "- Node " << node->get_name() << " has requested to free "
                               << std::setprecision(5)
                               << static_cast<float>(tensor->size() / 1024.0f / 1024.0f) << " MiB "
                               << "for tensor " << tensor->get_name() << " of shape ("
                               << join(tensor->get_shape()) << ")" << std::endl;
                    mem_free += tensor->size();
                    mm.free(tensor->get_pool_offset());
                }
            }
        }
        ss_verbose << "Total Memory free for node " << node->get_name() << " = "
                   << static_cast<float>(mem_free) / 1024.0f / 1024.0f << " MiB" << std::endl;
        total_mem_free_all_nodes += mem_free;

        ss_verbose << std::endl;
    }

    function->set_temporary_pool_size(mm.max_allocated());

    ss_concise << "Total Memory alloc requested by all nodes combined = "
               << static_cast<float>(total_mem_alloc_all_nodes_requested) / 1024.0f / 1024.0f
               << " MiB " << std::endl;
    ss_concise << "Total Memory actually requested to allocate by all nodes combined = "
               << static_cast<float>(total_mem_alloc_all_nodes_actual) / 1024.0f / 1024.0f
               << " MiB " << std::endl;
    ss_concise << "Total Memory free for all nodes = "
               << static_cast<float>(total_mem_free_all_nodes) / 1024.0f / 1024.0f << " MiB "
               << std::endl;
    ss_concise << "Net Memory for all nodes = "
               << (total_mem_alloc_all_nodes_actual - total_mem_free_all_nodes) << std::endl;
    ss_concise << std::endl;

    ss_concise << "Set of discrete sizes in MiB that are requested to be allocated: " << std::endl;
    ss_concise << "[";
    for (auto it : sizes_requested)
    {
        ss_concise << static_cast<float>(it) / 1024.0f / 1024.0f << ", ";
    }
    ss_concise << "]" << std::endl;
    ss_concise << std::endl;

    ss_concise
        << "Map of discrete alloc request sizes (in MiB) and the number of nodes requesting them: "
        << std::endl;
    ss_concise << "[";
    for (auto itr : size_num_nodes)
    {
        ss_concise << "(" << static_cast<float>(itr.first) / 1024.0f / 1024.0f << ", " << itr.second
                   << "), ";
    }
    ss_concise << "]" << std::endl;
    ss_concise << std::endl;

    ss_concise << "Map of discrete alloc request sizes (in MiB) and the number of nodes actually "
                  "contributing to the maximum allocation value: "
               << std::endl;
    ss_concise << "[";

    for (auto itr_ : size_nodes_adding_to_maxalloc)
    {
        ss_concise << "(" << static_cast<float>(itr_.first) / 1024.0f / 1024.0f << ", "
                   << itr_.second.first << ", {";
        for (auto itr_1 : itr_.second.second)
        {
            ss_concise << itr_1 << ", ";
        }
        ss_concise << "})";
    }
    ss_concise << "]" << std::endl;

    for (auto itr_2 : size_nodes_adding_to_maxalloc)
    {
        string n = (itr_2.second.first == 1) ? " node" : " nodes";
        ss_concise << itr_2.second.first << n << " of types ";
        for (auto itr_3 : itr_2.second.second)
        {
            ss_concise << itr_3 << ", ";
        }
        ss_concise << "contribute to the max allocated size by requesting "
                   << static_cast<float>(itr_2.first) / 1024.0f / 1024.0f << " MiB" << std::endl;
    }
    ss_verbose << ss_concise.str();
    std::cout << ss_verbose.str() << std::endl;
    //std::cout << ss_concise.str() << std::endl;

    return false;
}

pass::MemoryManager::node::node(size_t size, block_state state)
    : m_size{size}
    , m_state{state}
{
}

pass::MemoryManager::MemoryManager(size_t alignment, bool disable_memory_reuse)
    : m_alignment{alignment}
    , m_scheme{disable_memory_reuse ? allocation_scheme::NO_REUSE : allocation_scheme::FIRST_FIT}
    , m_max_allocated{0}
{
    std::cout << "Constructor of Memory Manager called where m_max_allocated is set to 0"
              << std::endl;
    if (m_alignment == 0)
    {
        throw invalid_argument("Memory alignment must be > 0");
    }
    m_node_list.emplace_back(numeric_limits<size_t>::max(), block_state::FREE);
}

size_t pass::MemoryManager::allocate(size_t size)
{
    /*     std::cout << std::endl;
    std::cout << "TRACE from pass::MemoryManager::allocate(size_t size): " << std::endl;
    print_trace(); */
    std::cout << "In pass::MemoryManager::allocate, memory of size "
              << static_cast<float>(size) / 1024.0f / 1024.0f << " MiB is being requested "
              << std::endl;
    size_t rc;
    switch (m_scheme)
    {
    case allocation_scheme::FIRST_FIT: rc = first_fit(size); break;
    case allocation_scheme::BEST_FIT: rc = best_fit(size); break;
    case allocation_scheme::NO_REUSE: rc = no_reuse_allocator(size); break;
    }
    return rc;
}

size_t pass::MemoryManager::allocate(size_t size,
                                     const std::shared_ptr<ngraph::Node>& node_in_use,
                                     std::map<size_t, std::pair<size_t, std::set<std::string>>>& s)
{
    ss_verbose << "In pass::MemoryManager::allocate,";
    size_t rc;
    switch (m_scheme)
    {
    case allocation_scheme::FIRST_FIT: rc = first_fit(size, node_in_use, s); break;
    case allocation_scheme::BEST_FIT: rc = best_fit(size); break;
    case allocation_scheme::NO_REUSE: rc = no_reuse_allocator(size); break;
    }
    return rc;
}

size_t pass::MemoryManager::no_reuse_allocator(size_t size)
{
    size_t offset = m_max_allocated;
    m_max_allocated += align(size, m_alignment);
    return offset;
}

size_t pass::MemoryManager::best_fit(size_t size)
{
    size = align(size, m_alignment);
    size_t offset = 0;
    size_t min_delta = numeric_limits<size_t>::max();
    auto best_fit = m_node_list.end();
    size_t best_offset = offset;
    for (auto it = m_node_list.begin(); it != m_node_list.end(); ++it)
    {
        if (it->m_state == block_state::FREE && it->m_size >= size)
        {
            size_t delta = it->m_size - size;
            if (delta < min_delta)
            {
                min_delta = delta;
                best_fit = it;
                best_offset = offset;
            }
        }
        offset += it->m_size;
    }

    if (best_fit == m_node_list.end())
    {
        throw bad_alloc();
    }

    if (min_delta == 0)
    {
        // exact fit
        best_fit->m_state = block_state::ALLOCATED;
    }
    else
    {
        m_node_list.insert(best_fit, node{size, block_state::ALLOCATED});
        best_fit->m_size -= size;
    }
    m_max_allocated = max(m_max_allocated, best_offset + size);

    return best_offset;
}

size_t pass::MemoryManager::first_fit(size_t size)
{
    std::pair<size_t, bool> r = first_fit_private(size);
    if (r.second)
    {
        std::cout << "NO ADDITIONAL MEM ADDED IN THIS ALLOCATE CALL" << std::endl;
    }
    else
    {
        std::cout << "ADDITIONAL MEM ADDED IN THIS ALLOCATE CALL" << std::endl;
    }

    return r.first;
}

size_t pass::MemoryManager::first_fit(size_t size,
                                      const std::shared_ptr<ngraph::Node>& node_in_use,
                                      std::map<size_t, std::pair<size_t, std::set<std::string>>>& s)
{
    std::pair<size_t, bool> r = first_fit_private(size);
    if (r.second)
    {
        ss_verbose << "NO ADDITIONAL MEM ADDED IN THIS ALLOCATE CALL" << std::endl;
    }
    else
    {
        ss_verbose << "ADDITIONAL MEM ADDED IN THIS ALLOCATE CALL" << std::endl;
        if (s.find(size) == s.end())
        {
            std::set<std::string> node_desc{node_in_use->description()};
            s.insert(std::make_pair(size, std::make_pair(1, node_desc)));
        }
        else
        {
            s[size].first++;
            s[size].second.insert(node_in_use->description());
        }
    }

    return r.first;
}

std::pair<size_t, bool> pass::MemoryManager::first_fit_private(size_t size)
{
    size = align(size, m_alignment);
    size_t offset = 0;
    bool found = false;
    for (auto it = m_node_list.begin(); it != m_node_list.end(); ++it)
    {
        if (it->m_state == block_state::FREE && it->m_size >= size)
        {
            if (it->m_size > size)
            {
                m_node_list.insert(it, node{size, block_state::ALLOCATED});
                it->m_size -= size;
            }
            else
            {
                // exact fit
                it->m_state = block_state::ALLOCATED;
            }

            found = true;
            break;
        }
        offset += it->m_size;
    }
    if (!found)
    {
        throw bad_alloc();
    }
    ss_verbose << "in first_fit, offset returned = " << offset
               << " Bytes = " << (offset / 1024.0f / 1024.0f) << " MiB " << std::endl;
    ss_verbose << "And, size requested was " << size << " Bytes = " << (size / 1024.0f / 1024.0f)
               << " MiB " << std::endl;
    ss_verbose << "Max allocated is being set in first fit: "
               << " Before set: " << (m_max_allocated / 1024.0f / 1024.0f) << " MiB ";
    size_t before_set = m_max_allocated;
    m_max_allocated = max(m_max_allocated, offset + size);
    ss_verbose << " After set: " << (m_max_allocated / 1024.0f / 1024.0f) << " MiB " << std::endl;
    size_t after_set = m_max_allocated;
    bool is_not_alloc = (before_set == after_set) ? true : false;

    return std::make_pair(offset, is_not_alloc);
}

void pass::MemoryManager::free(size_t offset)
{
    size_t search_offset = 0;
    bool found = false;
    for (auto it = m_node_list.begin(); it != m_node_list.end(); ++it)
    {
        if (offset == search_offset)
        {
            list<node>::iterator it_next = next(it);
            if (it == m_node_list.begin())
            {
                // free the first node in the list
                it->m_state = block_state::FREE;
            }
            else
            {
                // node has predecessor
                list<node>::iterator it_prev = prev(it);
                if (it_prev->m_state == block_state::FREE)
                {
                    it->m_size += it_prev->m_size;
                    m_node_list.erase(it_prev);
                }
            }
            if (it_next != m_node_list.end() && it_next->m_state == block_state::FREE)
            {
                // join this node with next
                it->m_size += it_next->m_size;
                m_node_list.erase(it_next);
            }
            it->m_state = block_state::FREE;
            found = true;
            break;
        }
        search_offset += it->m_size;
    }
    if (!found)
    {
        throw runtime_error("bad free");
    }
}

void pass::MemoryManager::dump(ostream& out)
{
    for (const node& n : m_node_list)
    {
        out << "size=" << n.m_size << ", ";
        out << (n.m_state == block_state::FREE ? "FREE" : "ALLOCATED");
        out << "\n";
    }
}

size_t pass::MemoryManager::align(size_t size, size_t alignment)
{
    if (alignment == 0)
    {
        throw invalid_argument("alignment must be > 0");
    }
    if (size == 0)
    {
        size = alignment;
    }
    else
    {
        auto remainder = size % alignment;
        if (remainder > 0)
        {
            size += (alignment - remainder);
        }
    }
    return size;
}
