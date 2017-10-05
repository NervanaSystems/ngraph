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
#include "ngraph/pass/assign_tensors.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/propagate_types.hpp"
#include "ngraph/pass/topological_sort.hpp"
#include "test_tools.hpp"

using namespace ngraph;
using namespace std;

TEST(pass_manager, add)
{
    pass::Manager pass_manager;

    pass_manager.register_pass<pass::TopologicalSort>();
    pass_manager.register_pass<pass::PropagateTypes>();
    pass_manager.register_pass<pass::AssignTensors>();

    auto graph = make_test_graph();
    size_t node_count = get_node_count(graph->get_result());
    pass_manager.run_passes(graph.get());
    auto sorted = graph->get_ordered_ops();
    EXPECT_EQ(node_count, sorted.size());
    EXPECT_TRUE(validate_list(sorted));
}

TEST(pass_manager, dependency)
{
    pass::Manager pass_manager;

    pass_manager.register_pass<pass::TopologicalSort>();
    EXPECT_THROW(pass_manager.register_pass<pass::AssignTensors>(), runtime_error);
}
