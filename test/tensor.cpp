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
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "ngraph/function.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/pass/assign_tensors.hpp"
#include "ngraph/pass/liveness.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/propagate_types.hpp"
#include "ngraph/pass/topological_sort.hpp"
#include "test_tools.hpp"

using namespace std;
using namespace ngraph;
using namespace ngraph::descriptor;

TEST(tensor, size)
{
    pass::Manager pass_manager;

    pass_manager.register_pass<pass::TopologicalSort>();
    pass_manager.register_pass<pass::PropagateTypes>();
    pass_manager.register_pass<pass::AssignTensors>();
    pass_manager.register_pass<pass::Liveness>();

    {
        auto arg0 = make_shared<op::Parameter>(element::Float32::element_type(), Shape{2, 3});
        auto add = make_shared<op::Add>(arg0, arg0);
        auto rt = make_shared<TensorViewType>(element::Float32::element_type(), Shape{2, 3});
        auto f0 = make_shared<Function>(add, rt, op::Parameters{arg0});

        pass_manager.run_passes(f0);

        auto outputs = arg0->get_outputs();
        ASSERT_EQ(1, outputs.size());
        Tensor& output = outputs[0].get_tensor();
        EXPECT_EQ(2 * 3 * 4, output.size());
    }

    {
        auto arg0 = make_shared<op::Parameter>(element::Float32::element_type(), Shape{});
        auto add = make_shared<op::Add>(arg0, arg0);
        auto rt = make_shared<TensorViewType>(element::Float32::element_type(), Shape{});
        auto f0 = make_shared<Function>(add, rt, op::Parameters{arg0});

        pass_manager.run_passes(f0);

        auto outputs = arg0->get_outputs();
        ASSERT_EQ(1, outputs.size());
        Tensor& output = outputs[0].get_tensor();
        EXPECT_EQ(1 * 4, output.size());
    }

    {
        auto arg0 = make_shared<op::Parameter>(element::Float32::element_type(), Shape{1});
        auto add = make_shared<op::Add>(arg0, arg0);
        auto rt = make_shared<TensorViewType>(element::Float32::element_type(), Shape{1});
        auto f0 = make_shared<Function>(add, rt, op::Parameters{arg0});

        pass_manager.run_passes(f0);

        auto outputs = arg0->get_outputs();
        ASSERT_EQ(1, outputs.size());
        Tensor& output = outputs[0].get_tensor();
        EXPECT_EQ(1 * 4, output.size());
    }
}
