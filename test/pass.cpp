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

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <list>
#include <memory>

#include "gtest/gtest.h"
#include "ngraph/op/add.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/pass/constant_to_broadcast.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/visualize_tree.hpp"
#include "ngraph/serializer.hpp"
#include "util/test_tools.hpp"

using namespace ngraph;
using namespace std;

TEST(pass, visualize_tree)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto C = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>((A + B) * C, ParameterVector{A, B, C});

    ngraph::pass::Manager pm;
    pm.register_pass<pass::VisualizeTree>("test_viz.png");
    pm.run_passes(f);
}

TEST(pass, constant_to_broadcast)
{
    Shape shape{128, 256, 1, 1};
    vector<float> v = {3};
    auto c = make_shared<op::Constant>(element::f32, shape, v);
    auto f = make_shared<Function>(c, ParameterVector{});

    {
        ngraph::pass::Manager pm;
        pm.register_pass<pass::VisualizeTree>("pre_constant_to_broadcast.png");
        pm.run_passes(f);
    }
    {
        ngraph::pass::Manager pm;
        pm.register_pass<pass::ConstantToBroadcast>();
        EXPECT_EQ(count_ops_of_type<op::Broadcast>(f), 0);
        pm.run_passes(f);
        EXPECT_EQ(count_ops_of_type<op::Broadcast>(f), 1);
    }
    {
        ngraph::pass::Manager pm;
        pm.register_pass<pass::VisualizeTree>("post_constant_to_broadcast.png");
        pm.run_passes(f);
    }
}
