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

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "ngraph/ngraph.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/shape_relevance.hpp"

using namespace ngraph;
using namespace std;

TEST(shape_relevance, simple)
{
    auto param0 = make_shared<op::v0::Parameter>(element::f32, Shape{4, 6});
    auto param1 = make_shared<op::v0::Parameter>(element::f32, Shape{4, 6});
    auto x = make_shared<op::v1::Add>(param0, param1);

    auto f = make_shared<Function>(x, ParameterVector{param0, param1});

    pass::Manager manager;
    manager.register_pass<pass::ShapeRelevance>();
    manager.run_passes(f);

    ASSERT_FALSE(param0->is_relevant_to_shapes());
    ASSERT_FALSE(param1->is_relevant_to_shapes());
}

TEST(shape_relevance, param_direct)
{
    auto param0 = make_shared<op::v0::Parameter>(element::f32, Shape{4, 6});
    auto param1 = make_shared<op::v0::Parameter>(element::i64, Shape{4});
    auto x = make_shared<op::v1::Reshape>(param0, param1, true);

    auto f = make_shared<Function>(x, ParameterVector{param0, param1});

    pass::Manager manager;
    manager.register_pass<pass::ShapeRelevance>();
    manager.run_passes(f);

    ASSERT_FALSE(param0->is_relevant_to_shapes());
    ASSERT_TRUE(param1->is_relevant_to_shapes());
}

TEST(shape_relevance, param_indirect)
{
    auto param0 = make_shared<op::v0::Parameter>(element::f32, Shape{4, 6});
    auto param1 = make_shared<op::v0::Parameter>(element::i64, Shape{4});
    auto param2 = make_shared<op::v0::Parameter>(element::i64, Shape{2});

    auto c = make_shared<op::v0::Concat>(OutputVector{param1, param2}, 0);
    auto x = make_shared<op::v1::Reshape>(param0, c, true);

    auto f = make_shared<Function>(x, ParameterVector{param0, param1, param2});

    pass::Manager manager;
    manager.register_pass<pass::ShapeRelevance>();
    manager.run_passes(f);

    ASSERT_FALSE(param0->is_relevant_to_shapes());
    ASSERT_TRUE(param1->is_relevant_to_shapes());
    ASSERT_TRUE(param2->is_relevant_to_shapes());
}

TEST(shape_relevance, param_shape_of_direct_v0)
{
    auto param0 = make_shared<op::v0::Parameter>(element::f32, Shape{4, 6});

    auto x = make_shared<op::v1::Reshape>(param0, make_shared<op::v0::ShapeOf>(param0), true);

    auto f = make_shared<Function>(x, ParameterVector{param0});

    pass::Manager manager;
    manager.register_pass<pass::ShapeRelevance>();
    manager.run_passes(f);

    ASSERT_FALSE(param0->is_relevant_to_shapes());
}

TEST(shape_relevance, param_shape_of_direct_v3)
{
    auto param0 = make_shared<op::v0::Parameter>(element::f32, Shape{4, 6});

    auto x = make_shared<op::v1::Reshape>(param0, make_shared<op::v3::ShapeOf>(param0), true);

    auto f = make_shared<Function>(x, ParameterVector{param0});

    pass::Manager manager;
    manager.register_pass<pass::ShapeRelevance>();
    manager.run_passes(f);

    ASSERT_FALSE(param0->is_relevant_to_shapes());
}

TEST(shape_relevance, param_shape_of_direct_i32_v3)
{
    auto param0 = make_shared<op::v0::Parameter>(element::f32, Shape{4, 6});

    auto x = make_shared<op::v1::Reshape>(
        param0, make_shared<op::v3::ShapeOf>(param0, element::i32), true);

    auto f = make_shared<Function>(x, ParameterVector{param0});

    pass::Manager manager;
    manager.register_pass<pass::ShapeRelevance>();
    manager.run_passes(f);

    ASSERT_FALSE(param0->is_relevant_to_shapes());
}

TEST(shape_relevance, param_shape_of_indirect_v0)
{
    auto param0 = make_shared<op::v0::Parameter>(element::f32, Shape{4, 6});

    auto s = make_shared<op::v0::ShapeOf>(param0);
    auto r = make_shared<op::v0::Reverse>(s, AxisSet{0});
    auto x = make_shared<op::v1::Reshape>(param0, r, true);

    auto f = make_shared<Function>(x, ParameterVector{param0});

    pass::Manager manager;
    manager.register_pass<pass::ShapeRelevance>();
    manager.run_passes(f);

    ASSERT_FALSE(param0->is_relevant_to_shapes());
}

TEST(shape_relevance, param_shape_of_indirect_v3)
{
    auto param0 = make_shared<op::v0::Parameter>(element::f32, Shape{4, 6});

    auto s = make_shared<op::v3::ShapeOf>(param0);
    auto r = make_shared<op::v0::Reverse>(s, AxisSet{0});
    auto x = make_shared<op::v1::Reshape>(param0, r, true);

    auto f = make_shared<Function>(x, ParameterVector{param0});

    pass::Manager manager;
    manager.register_pass<pass::ShapeRelevance>();
    manager.run_passes(f);

    ASSERT_FALSE(param0->is_relevant_to_shapes());
}

TEST(shape_relevance, param_shape_of_indirect_i32_v3)
{
    auto param0 = make_shared<op::v0::Parameter>(element::f32, Shape{4, 6});

    auto s = make_shared<op::v3::ShapeOf>(param0, element::i32);
    auto r = make_shared<op::v0::Reverse>(s, AxisSet{0});
    auto x = make_shared<op::v1::Reshape>(param0, r, true);

    auto f = make_shared<Function>(x, ParameterVector{param0});

    pass::Manager manager;
    manager.register_pass<pass::ShapeRelevance>();
    manager.run_passes(f);

    ASSERT_FALSE(param0->is_relevant_to_shapes());
}
