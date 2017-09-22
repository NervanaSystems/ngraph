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
#include <memory>

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/pass/assign_tensors.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/propagate_types.hpp"
#include "ngraph/pass/liveness.hpp"
#include "ngraph/pass/topological_sort.hpp"
#include "ngraph/function.hpp"
#include "test_tools.hpp"

using namespace std;
using namespace ngraph;
using namespace ngraph::descriptor;

TEST(tensor, size)
{
    pass::Manager pass_manager;
    auto          topological_sort = make_shared<pass::TopologicalSort>();
    auto          propagate_types  = make_shared<pass::PropagateTypes>();
    auto          assign_tensors   = make_shared<pass::AssignTensors>();
    auto          liveness         = make_shared<pass::Liveness>();

    pass_manager.register_pass(topological_sort);
    pass_manager.register_pass(propagate_types);
    pass_manager.register_pass(assign_tensors);
    pass_manager.register_pass(liveness);

    {
        auto arg0 = make_shared<op::Parameter>(element::Float32::element_type(), Shape{2, 3});
        auto add  = make_shared<op::Add>(arg0, arg0);
        auto f0   = make_shared<Function>(add, op::Parameters{arg0});

        pass_manager.run_passes(f0);

        auto outputs = arg0->get_outputs();
        ASSERT_EQ(1, outputs.size());
        Tensor& output = outputs[0].get_tensor();
        EXPECT_EQ(2 * 3 * 4, output.size());
    }

    {
        auto arg0 = make_shared<op::Parameter>(element::Float32::element_type(), Shape{});
        auto add  = make_shared<op::Add>(arg0, arg0);
        auto f0   = make_shared<Function>(add, op::Parameters{arg0});

        pass_manager.run_passes(f0);

        auto outputs = arg0->get_outputs();
        ASSERT_EQ(1, outputs.size());
        Tensor& output = outputs[0].get_tensor();
        EXPECT_EQ(1 * 4, output.size());
    }

    {
        auto arg0 = make_shared<op::Parameter>(element::Float32::element_type(), Shape{1});
        auto add  = make_shared<op::Add>(arg0, arg0);
        auto f0   = make_shared<Function>(add, op::Parameters{arg0});

        pass_manager.run_passes(f0);

        auto outputs = arg0->get_outputs();
        ASSERT_EQ(1, outputs.size());
        Tensor& output = outputs[0].get_tensor();
        EXPECT_EQ(1 * 4, output.size());
    }
}
