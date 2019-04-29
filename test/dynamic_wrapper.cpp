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

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/runtime/dynamic_wrapper/dynamic_wrapper_backend.hpp"
#include "util/all_close_f.hpp"
#include "util/test_tools.hpp"

using namespace ngraph;
using namespace std;

TEST(dynamic_wrapper, create)
{
    auto backend = runtime::Backend::create("INTERPRETER", true);
    ASSERT_NE(backend, nullptr);
    ASSERT_TRUE(backend->supports_dynamic_tensors());
}

TEST(dynamic_wrapper, create_no_dynamic)
{
    auto backend = runtime::Backend::create("INTERPRETER");
    ASSERT_NE(backend, nullptr);
    ASSERT_FALSE(backend->supports_dynamic_tensors());
}

TEST(dynamic_wrapper, create_dynamic_tensor)
{
    auto backend = runtime::Backend::create("INTERPRETER", true);
    auto t = backend->create_dynamic_tensor(element::f32, PartialShape{2, Dimension::dynamic(), 3});
    ASSERT_TRUE(t->get_partial_shape().same_scheme(PartialShape{2, Dimension::dynamic(), 3}));
}

TEST(dynamic_wrapper, abc_static)
{
    auto a = make_shared<op::Parameter>(element::f32, Shape{2, 3, 3});
    auto b = make_shared<op::Parameter>(element::f32, Shape{2, 3, 3});
    auto c = make_shared<op::Parameter>(element::f32, Shape{2, 3, 3});

    auto a_plus_b_times_c = (a + b) * c;

    auto f = make_shared<Function>(NodeVector{a_plus_b_times_c}, ParameterVector{a, b, c});

    auto backend = runtime::Backend::create("INTERPRETER", true);

    auto ex = backend->compile(f);

    auto t_a = backend->create_tensor(element::f32, Shape{2, 3, 3});
    auto t_b = backend->create_tensor(element::f32, Shape{2, 3, 3});
    auto t_c = backend->create_tensor(element::f32, Shape{2, 3, 3});
    auto t_r = backend->create_tensor(element::f32, Shape{2, 3, 3});

    copy_data(t_a, vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18});
    copy_data(t_b, vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18});
    copy_data(t_c, vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18});

    ex->call_with_validate({t_r}, {t_a, t_b, t_c});

    ASSERT_EQ(t_r->get_shape(), (Shape{2, 3, 3}));

    vector<float> expected_values(2 * 3 * 3);
    for (size_t i = 0; i < 2 * 3 * 3; i++)
    {
        expected_values[i] = ((i + 1) + (i + 1)) * (i + 1);
    }

    EXPECT_TRUE(test::all_close_f(read_vector<float>(t_r), expected_values));
}

TEST(dynamic_wrapper, abc)
{
    auto a = make_shared<op::Parameter>(element::f32, PartialShape{2, Dimension::dynamic(), 3});
    auto b = make_shared<op::Parameter>(element::f32, PartialShape{2, Dimension::dynamic(), 3});
    auto c = make_shared<op::Parameter>(element::f32, PartialShape{2, Dimension::dynamic(), 3});

    auto a_plus_b_times_c = (a + b) * c;

    auto f = make_shared<Function>(NodeVector{a_plus_b_times_c}, ParameterVector{a, b, c});

    auto backend = runtime::Backend::create("INTERPRETER", true);

    auto ex = backend->compile(f);

    auto t_r =
        backend->create_dynamic_tensor(element::f32, PartialShape{2, Dimension::dynamic(), 3});

    for (size_t middle_dim = 0; middle_dim < 5; middle_dim++)
    {
        vector<float> inputs(2 * middle_dim * 3);
        for (size_t i = 0; i < 2 * middle_dim * 3; i++)
        {
            inputs[i] = i + 1;
        }

        auto t_a = backend->create_tensor(element::f32, Shape{2, middle_dim, 3});
        auto t_b = backend->create_tensor(element::f32, Shape{2, middle_dim, 3});
        auto t_c = backend->create_tensor(element::f32, Shape{2, middle_dim, 3});

        copy_data(t_a, inputs);
        copy_data(t_b, inputs);
        copy_data(t_c, inputs);

        ex->call_with_validate({t_r}, {t_a, t_b, t_c});

        ASSERT_EQ(t_r->get_shape(), (Shape{2, middle_dim, 3}));
        auto results = read_vector<float>(t_r);

        vector<float> expected_values(2 * middle_dim * 3);
        for (size_t i = 0; i < 2 * middle_dim * 3; i++)
        {
            expected_values[i] = ((i + 1) + (i + 1)) * (i + 1);
        }

        EXPECT_TRUE(test::all_close_f(results, expected_values));
    }
}
