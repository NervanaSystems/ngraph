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
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <random>
#include <string>

#undef IN_NGRAPH_LIBRARY
#include "gtest/gtest.h"
#include "ngraph/frontend/fluid/operators/reduce_sum.hpp"
#include "ngraph/ngraph.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/ndarray.hpp"
#include "util/random.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

static std::mt19937_64 random_generator;

using namespace std;
using namespace ngraph;

static string s_manifest = "test.manifest";

NGRAPH_TEST(CPU, fluid_reduce_sum_dynamic)
{
    // Create a graph for f(x,axes:int32) = Sum(x,Convert<int64>(axes)).
    auto x = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto y = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    vector<int> dim = {-1};

    auto sum = make_shared<ngraph::fluid::ReduceSum>(x, dim, false, false);
    auto grad = make_shared<ngraph::fluid::ReduceSumGrad>(x, y, dim, false, false);
    ASSERT_TRUE(sum->get_output_partial_shape(0).rank().is_dynamic());
    ASSERT_TRUE(grad->get_output_partial_shape(0).rank().is_dynamic());

    auto f = make_shared<Function>(NodeVector{sum}, ParameterVector{x});
    auto g = make_shared<Function>(NodeVector{grad}, ParameterVector{x, y});

    auto backend = runtime::Backend::create("CPU", true);

    auto ex = backend->compile(f);
    auto gex = backend->compile(g);

    auto t_r = backend->create_dynamic_tensor(element::f32, PartialShape::dynamic());
    auto t_gr = backend->create_dynamic_tensor(element::f32, PartialShape::dynamic());

    std::vector<Shape> x_shapes{Shape{2, 3}, Shape{5}};
    std::vector<Shape> y_shapes{Shape{2}, Shape{}};
    std::vector<std::vector<float>> inputs{{1, 2, 3, 4, 5, 6}, {1, 2, 3, 4, 5}};
    std::vector<std::vector<float>> grads{{1, 2}, {1}};
    std::vector<Shape> expected_result_shapes{Shape{2}, Shape{}};
    std::vector<Shape> expected_gresult_shapes{Shape{2, 3}, Shape{5}};
    std::vector<std::vector<float>> expected_results{{6, 15}, {15}};
    std::vector<std::vector<float>> expected_gresults{{1, 1, 1, 2, 2, 2}, {1, 1, 1, 1, 1}};

    for (size_t i = 0; i < x_shapes.size(); i++)
    {
        auto t_x = backend->create_tensor(element::f32, x_shapes[i]);
        auto t_y = backend->create_tensor(element::f32, y_shapes[i]);

        copy_data(t_x, inputs[i]);
        copy_data(t_y, grads[i]);

        ex->call_with_validate({t_r}, {t_x});
        gex->call_with_validate({t_gr}, {t_x, t_y});

        ASSERT_EQ(t_r->get_shape(), expected_result_shapes[i]);
        ASSERT_EQ(t_gr->get_shape(), expected_gresult_shapes[i]);

        auto results = read_vector<float>(t_r);
        auto gresults = read_vector<float>(t_gr);

        ASSERT_TRUE(test::all_close_f(results, expected_results[i], MIN_FLOAT_TOLERANCE_BITS));
        ASSERT_TRUE(test::all_close_f(gresults, expected_gresults[i], MIN_FLOAT_TOLERANCE_BITS));
    }
}

NGRAPH_TEST(CPU, fluid_reduce_sum_all_dynamic)
{
    // Create a graph for f(x,axes:int32) = Sum(x,Convert<int64>(axes)).
    auto x = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto y = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    vector<int> dim = {-1};

    auto sum = make_shared<ngraph::fluid::ReduceSum>(x, dim, true, false);
    auto grad = make_shared<ngraph::fluid::ReduceSumGrad>(x, y, dim, true, false);
    ASSERT_TRUE(sum->get_output_partial_shape(0).rank().is_dynamic());
    ASSERT_TRUE(grad->get_output_partial_shape(0).rank().is_dynamic());

    auto f = make_shared<Function>(NodeVector{sum}, ParameterVector{x});
    auto g = make_shared<Function>(NodeVector{grad}, ParameterVector{x, y});

    auto backend = runtime::Backend::create("CPU", true);

    auto ex = backend->compile(f);
    auto gex = backend->compile(g);

    auto t_r = backend->create_dynamic_tensor(element::f32, PartialShape::dynamic());
    auto t_gr = backend->create_dynamic_tensor(element::f32, PartialShape::dynamic());

    std::vector<Shape> x_shapes{Shape{2, 3}, Shape{5}};
    std::vector<Shape> y_shapes{Shape{}, Shape{}};
    std::vector<std::vector<float>> inputs{{1, 2, 3, 4, 5, 6}, {1, 2, 3, 4, 5}};
    std::vector<std::vector<float>> grads{{2}, {1}};
    std::vector<Shape> expected_result_shapes{Shape{}, Shape{}};
    std::vector<Shape> expected_gresult_shapes{Shape{2, 3}, Shape{5}};
    std::vector<std::vector<float>> expected_results{{21}, {15}};
    std::vector<std::vector<float>> expected_gresults{{2, 2, 2, 2, 2, 2}, {1, 1, 1, 1, 1}};

    for (size_t i = 0; i < x_shapes.size(); i++)
    {
        auto t_x = backend->create_tensor(element::f32, x_shapes[i]);
        auto t_y = backend->create_tensor(element::f32, y_shapes[i]);

        copy_data(t_x, inputs[i]);
        copy_data(t_y, grads[i]);

        ex->call_with_validate({t_r}, {t_x});
        gex->call_with_validate({t_gr}, {t_x, t_y});

        ASSERT_EQ(t_r->get_shape(), expected_result_shapes[i]);
        ASSERT_EQ(t_gr->get_shape(), expected_gresult_shapes[i]);

        auto results = read_vector<float>(t_r);
        auto gresults = read_vector<float>(t_gr);

        ASSERT_TRUE(test::all_close_f(results, expected_results[i], MIN_FLOAT_TOLERANCE_BITS));
        ASSERT_TRUE(test::all_close_f(gresults, expected_gresults[i], MIN_FLOAT_TOLERANCE_BITS));
    }
}

NGRAPH_TEST(CPU, fluid_reduce_sum_dynamic_keep_dim)
{
    // Create a graph for f(x,axes:int32) = Sum(x,Convert<int64>(axes)).
    auto x = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto y = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    vector<int> dim = {-1};

    auto sum = make_shared<ngraph::fluid::ReduceSum>(x, dim, false, true);
    auto grad = make_shared<ngraph::fluid::ReduceSumGrad>(x, y, dim, false, true);
    ASSERT_TRUE(sum->get_output_partial_shape(0).rank().is_dynamic());
    ASSERT_TRUE(grad->get_output_partial_shape(0).rank().is_dynamic());

    auto f = make_shared<Function>(NodeVector{sum}, ParameterVector{x});
    auto g = make_shared<Function>(NodeVector{grad}, ParameterVector{x, y});

    auto backend = runtime::Backend::create("CPU", true);

    auto ex = backend->compile(f);
    auto gex = backend->compile(g);

    auto t_r = backend->create_dynamic_tensor(element::f32, PartialShape::dynamic());
    auto t_gr = backend->create_dynamic_tensor(element::f32, PartialShape::dynamic());

    std::vector<Shape> x_shapes{Shape{2, 3}, Shape{5}};
    std::vector<Shape> y_shapes{Shape{2, 1}, Shape{1}};
    std::vector<std::vector<float>> inputs{{1, 2, 3, 4, 5, 6}, {1, 2, 3, 4, 5}};
    std::vector<std::vector<float>> grads{{1, 2}, {1}};
    std::vector<Shape> expected_result_shapes{Shape{2, 1}, Shape{1}};
    std::vector<Shape> expected_gresult_shapes{Shape{2, 3}, Shape{5}};
    std::vector<std::vector<float>> expected_results{{6, 15}, {15}};
    std::vector<std::vector<float>> expected_gresults{{1, 1, 1, 2, 2, 2}, {1, 1, 1, 1, 1}};

    for (size_t i = 0; i < x_shapes.size(); i++)
    {
        auto t_x = backend->create_tensor(element::f32, x_shapes[i]);
        auto t_y = backend->create_tensor(element::f32, y_shapes[i]);

        copy_data(t_x, inputs[i]);
        copy_data(t_y, grads[i]);

        ex->call_with_validate({t_r}, {t_x});
        gex->call_with_validate({t_gr}, {t_x, t_y});

        ASSERT_EQ(t_r->get_shape(), expected_result_shapes[i]);
        ASSERT_EQ(t_gr->get_shape(), expected_gresult_shapes[i]);

        auto results = read_vector<float>(t_r);
        auto gresults = read_vector<float>(t_gr);

        ASSERT_TRUE(test::all_close_f(results, expected_results[i], MIN_FLOAT_TOLERANCE_BITS));
        ASSERT_TRUE(test::all_close_f(gresults, expected_gresults[i], MIN_FLOAT_TOLERANCE_BITS));
    }
}

NGRAPH_TEST(CPU, fluid_reduce_sum_all_dynamic_keep_dim)
{
    // Create a graph for f(x,axes:int32) = Sum(x,Convert<int64>(axes)).
    auto x = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto y = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    vector<int> dim = {-1};

    auto sum = make_shared<ngraph::fluid::ReduceSum>(x, dim, true, true);
    auto grad = make_shared<ngraph::fluid::ReduceSumGrad>(x, y, dim, true, true);
    ASSERT_TRUE(sum->get_output_partial_shape(0).rank().is_dynamic());
    ASSERT_TRUE(grad->get_output_partial_shape(0).rank().is_dynamic());

    auto f = make_shared<Function>(NodeVector{sum}, ParameterVector{x});
    auto g = make_shared<Function>(NodeVector{grad}, ParameterVector{x, y});

    auto backend = runtime::Backend::create("CPU", true);

    auto ex = backend->compile(f);
    auto gex = backend->compile(g);

    auto t_r = backend->create_dynamic_tensor(element::f32, PartialShape::dynamic());
    auto t_gr = backend->create_dynamic_tensor(element::f32, PartialShape::dynamic());

    std::vector<Shape> x_shapes{Shape{2, 3}, Shape{5}};
    std::vector<Shape> y_shapes{Shape{1, 1}, Shape{1}};
    std::vector<std::vector<float>> inputs{{1, 2, 3, 4, 5, 6}, {1, 2, 3, 4, 5}};
    std::vector<std::vector<float>> grads{{2}, {1}};
    std::vector<Shape> expected_result_shapes{Shape{1, 1}, Shape{1}};
    std::vector<Shape> expected_gresult_shapes{Shape{2, 3}, Shape{5}};
    std::vector<std::vector<float>> expected_results{{21}, {15}};
    std::vector<std::vector<float>> expected_gresults{{2, 2, 2, 2, 2, 2}, {1, 1, 1, 1, 1}};

    for (size_t i = 0; i < x_shapes.size(); i++)
    {
        auto t_x = backend->create_tensor(element::f32, x_shapes[i]);
        auto t_y = backend->create_tensor(element::f32, y_shapes[i]);

        copy_data(t_x, inputs[i]);
        copy_data(t_y, grads[i]);

        ex->call_with_validate({t_r}, {t_x});
        gex->call_with_validate({t_gr}, {t_x, t_y});

        ASSERT_EQ(t_r->get_shape(), expected_result_shapes[i]);
        ASSERT_EQ(t_gr->get_shape(), expected_gresult_shapes[i]);

        auto results = read_vector<float>(t_r);
        auto gresults = read_vector<float>(t_gr);

        ASSERT_TRUE(test::all_close_f(results, expected_results[i], MIN_FLOAT_TOLERANCE_BITS));
        ASSERT_TRUE(test::all_close_f(gresults, expected_gresults[i], MIN_FLOAT_TOLERANCE_BITS));
    }
}
