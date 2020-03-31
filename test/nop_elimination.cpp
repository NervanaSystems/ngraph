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

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/nop_elimination.hpp"
#include "util/test_tools.hpp"

using namespace ngraph;
using namespace std;

TEST(nop_elimination, eliminate_pad)
{
    Shape shape_a{2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    CoordinateDiff padding_below{0};
    CoordinateDiff padding_above{0};
    auto f = make_shared<Function>(make_shared<op::Pad>(A, B, padding_below, padding_above),
                                   ParameterVector{A, B});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::NopElimination>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::Pad>(f), 0);
}

TEST(nop_elimination, eliminate_sum)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Sum>(A, AxisSet{}), ParameterVector{A});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::NopElimination>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::Sum>(f), 0);
}

TEST(nop_elimination, eliminate_convert)
{
    Shape shape{};
    auto type = element::f32;
    auto A = make_shared<op::Parameter>(type, shape);
    auto f = make_shared<Function>(make_shared<op::Convert>(A, element::f32), ParameterVector{A});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::NopElimination>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::Convert>(f), 0);
}

TEST(nop_elimination, eliminate_slice)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Slice>(A, Coordinate{0, 0}, Coordinate{2, 2}),
                                   ParameterVector{A});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::NopElimination>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::Slice>(f), 0);
}

TEST(nop_elimination, eliminate_broadcast)
{
    Shape shape{};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f =
        make_shared<Function>(make_shared<op::Broadcast>(A, shape, AxisSet{}), ParameterVector{A});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::NopElimination>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::Broadcast>(f), 0);
}

TEST(nop_elimination, eliminate_stop_gradient)
{
    Shape shape{};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::StopGradient>(A), ParameterVector{A});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::NopElimination>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::StopGradient>(f), 0);
}

TEST(nop_elimination, eliminate_shapeof_gather_axis_one)
{
    auto A = make_shared<op::Parameter>(element::f32, Shape{2, 3});
    auto index = make_shared<op::Constant>(element::i64, Shape{}, vector<float>{0});
    auto axis = make_shared<op::Constant>(element::i64, Shape{}, vector<float>{0});
    auto gather = make_shared<op::v1::Gather>(A, index, axis);
    auto shapeof = make_shared<op::ShapeOf>(gather);
    auto f = make_shared<Function>(NodeVector{shapeof}, ParameterVector{A});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::Validate>();
    /* pass_manager.register_pass<pass::NopElimination>(); */
    pass_manager.run_passes(f);

    auto backend = runtime::Backend::create("CPU");
    auto a = backend->create_tensor(element::f32, Shape{2, 3});
    copy_data(a, vector<float>{1, -2, 0, -4.75f});
    auto handle = backend->compile(f);
    auto result = backend->create_tensor(element::i64, Shape{1});
    handle->call_with_validate({result}, {a});
    EXPECT_EQ((vector<int64_t>{2}), read_vector<int64_t>(result));

    /* ASSERT_EQ(count_ops_of_type<op::StopGradient>(f), 0); */
}

TEST(nop_elimination, pass_property)
{
    auto pass = std::make_shared<ngraph::pass::NopElimination>();
    ASSERT_TRUE(pass->get_property(pass::PassProperty::REQUIRE_STATIC_SHAPE));
    ASSERT_FALSE(pass->get_property(pass::PassProperty::CHANGE_DYNAMIC_STATE));
}
