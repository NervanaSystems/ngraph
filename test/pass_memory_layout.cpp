// ----------------------------------------------------------------------------
// Copyright 2017 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "ngraph/ngraph.hpp"
#include "ngraph/pass/liveness.hpp"
#include "ngraph/pass/assign_tensors.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/propagate_types.hpp"
#include "ngraph/pass/topological_sort.hpp"
#include "ngraph/pass/liveness.hpp"
#include "ngraph/pass/visualize_tree.hpp"
#include "ngraph/pass/dump_sorted.hpp"
#include "ngraph/pass/memory_layout.hpp"
#include "test_tools.hpp"

using namespace ngraph;
using namespace std;

static vector<pass::MemoryManager::node> get_node_list(const pass::MemoryManager& mm)
{
    vector<pass::MemoryManager::node> rc;
    rc.insert(rc.end(), mm.begin(), mm.end());
    return rc;
}

TEST(memory_manager, allocate)
{
    pass::MemoryManager mm{1};

    // Special case, allocating size zero bumps the size of the alloc up to the alignment size
    EXPECT_EQ(0, mm.allocate(0));
    EXPECT_EQ(1, mm.allocate(10));
    EXPECT_EQ(11, mm.allocate(10));
    EXPECT_EQ(21, mm.allocate(10));
}

TEST(memory_manager, free_first_allocated)
{
    pass::MemoryManager mm{1};

    EXPECT_EQ(0, mm.allocate(10));
    EXPECT_EQ(10, mm.allocate(10));
    EXPECT_EQ(3, mm.get_node_list().size());

    mm.free(0);

    auto node_list = get_node_list(mm);
    EXPECT_EQ(3, node_list.size());
    EXPECT_TRUE(node_list[0].is_free());
    EXPECT_FALSE(node_list[1].is_free());
    EXPECT_TRUE(node_list[2].is_free());
}

TEST(memory_manager, free_middle_allocated)
{
    pass::MemoryManager mm{1};

    EXPECT_EQ(0, mm.allocate(10));
    EXPECT_EQ(10, mm.allocate(10));
    EXPECT_EQ(20, mm.allocate(10));
    EXPECT_EQ(30, mm.allocate(10));
    EXPECT_EQ(40, mm.allocate(10));
    EXPECT_EQ(6, mm.get_node_list().size());

    mm.free(10);

    auto node_list = get_node_list(mm);
    EXPECT_EQ(6, node_list.size());
    EXPECT_FALSE(node_list[0].is_free());
    EXPECT_TRUE(node_list[1].is_free());
    EXPECT_FALSE(node_list[2].is_free());
    EXPECT_FALSE(node_list[3].is_free());
    EXPECT_FALSE(node_list[4].is_free());
}

TEST(memory_manager, free_last_allocated)
{
    pass::MemoryManager mm{1};

    EXPECT_EQ(0, mm.allocate(10));
    EXPECT_EQ(10, mm.allocate(10));
    EXPECT_EQ(20, mm.allocate(10));
    EXPECT_EQ(30, mm.allocate(10));
    EXPECT_EQ(40, mm.allocate(10));
    EXPECT_EQ(6, mm.get_node_list().size());

    mm.free(40);

    auto node_list = get_node_list(mm);
    EXPECT_EQ(5, node_list.size());
    EXPECT_FALSE(node_list[0].is_free());
    EXPECT_FALSE(node_list[1].is_free());
    EXPECT_FALSE(node_list[2].is_free());
    EXPECT_FALSE(node_list[3].is_free());
    EXPECT_TRUE(node_list[4].is_free());
}

TEST(memory_manager, free_first_free)
{
    pass::MemoryManager mm{1};

    EXPECT_EQ(0, mm.allocate(10));
    EXPECT_EQ(10, mm.allocate(10));
    EXPECT_EQ(20, mm.allocate(10));
    EXPECT_EQ(30, mm.allocate(10));
    EXPECT_EQ(40, mm.allocate(10));
    EXPECT_EQ(6, mm.get_node_list().size());

    mm.free(10);
    mm.free(0);

    auto node_list = get_node_list(mm);
    EXPECT_EQ(5, node_list.size());
    EXPECT_TRUE(node_list[0].is_free());
    EXPECT_FALSE(node_list[1].is_free());
    EXPECT_FALSE(node_list[2].is_free());
    EXPECT_FALSE(node_list[3].is_free());
}

TEST(memory_manager, free_middle_free)
{
    pass::MemoryManager mm{1};

    EXPECT_EQ(0, mm.allocate(10));
    EXPECT_EQ(10, mm.allocate(10));
    EXPECT_EQ(20, mm.allocate(10));
    EXPECT_EQ(30, mm.allocate(10));
    EXPECT_EQ(40, mm.allocate(10));
    EXPECT_EQ(6, mm.get_node_list().size());

    mm.free(0);
    mm.free(20);
    mm.free(10);

    auto node_list = get_node_list(mm);
    EXPECT_EQ(4, node_list.size());
    EXPECT_TRUE(node_list[0].is_free());
    EXPECT_FALSE(node_list[1].is_free());
    EXPECT_FALSE(node_list[2].is_free());
}

TEST(memory_manager, max_allocated)
{
    pass::MemoryManager mm{1};

    EXPECT_EQ(0, mm.allocate(10));
    EXPECT_EQ(10, mm.allocate(10));
    EXPECT_EQ(20, mm.allocate(10));
    EXPECT_EQ(30, mm.allocate(10));
    EXPECT_EQ(40, mm.allocate(10));
    EXPECT_EQ(6, mm.get_node_list().size());

    mm.free(0);
    mm.free(20);
    mm.free(10);

    EXPECT_EQ(mm.max_allocated(), 50);
}

TEST(memory_manager, bad_free)
{
    pass::MemoryManager mm{1};

    EXPECT_THROW(mm.free(10), std::runtime_error);
}

TEST(memory_manager, align)
{
    EXPECT_EQ(8, pass::MemoryManager::align(0, 8));
    EXPECT_EQ(8, pass::MemoryManager::align(1, 8));
    EXPECT_EQ(8, pass::MemoryManager::align(2, 8));
    EXPECT_EQ(8, pass::MemoryManager::align(3, 8));
    EXPECT_EQ(8, pass::MemoryManager::align(4, 8));
    EXPECT_EQ(8, pass::MemoryManager::align(5, 8));
    EXPECT_EQ(8, pass::MemoryManager::align(6, 8));
    EXPECT_EQ(8, pass::MemoryManager::align(7, 8));
    EXPECT_EQ(8, pass::MemoryManager::align(8, 8));
    EXPECT_EQ(16, pass::MemoryManager::align(9, 8));
}

TEST(memory_manager, memory_align)
{
    pass::MemoryManager mm{64};

    EXPECT_EQ(0, mm.allocate(4));
    EXPECT_EQ(64, mm.allocate(4));
    EXPECT_EQ(128, mm.allocate(4));
}

TEST(memory_layout, basic)
{
    string dump_file = "memory_layout.txt";
    pass::Manager pass_manager;
    auto          topological_sort = make_shared<pass::TopologicalSort>();
    auto          propagate_types  = make_shared<pass::PropagateTypes>();
    auto          assign_tensors   = make_shared<pass::AssignTensors>();
    auto          liveness         = make_shared<pass::Liveness>();
    auto          memory_layout    = make_shared<pass::MemoryLayout>();
    auto          dump_sorted      = make_shared<pass::DumpSorted>(dump_file);

    pass_manager.register_pass(topological_sort);
    pass_manager.register_pass(propagate_types);
    pass_manager.register_pass(assign_tensors);
    pass_manager.register_pass(liveness);
    pass_manager.register_pass(memory_layout);
    pass_manager.register_pass(dump_sorted);

    auto graph = make_test_graph();
    pass_manager.run_passes(graph);
    auto sorted = pass_manager.get_call_graph();
    size_t temporary_pool_size = pass_manager.get_state().get_temporary_pool_size();
    EXPECT_EQ(12, temporary_pool_size);
}
