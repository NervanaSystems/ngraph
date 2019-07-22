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
#include "util/all_close_f.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(dynamic_${BACKEND_NAME}, create)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}", true);
    ASSERT_NE(backend, nullptr);
    ASSERT_TRUE(backend->supports_dynamic_tensors());
}

NGRAPH_TEST(dynamic_${BACKEND_NAME}, create_no_dynamic)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    ASSERT_NE(backend, nullptr);
    ASSERT_FALSE(backend->supports_dynamic_tensors());
}

NGRAPH_TEST(dynamic_${BACKEND_NAME}, create_dynamic_tensor)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}", true);
    auto t = backend->create_dynamic_tensor(element::f32, PartialShape{2, Dimension::dynamic(), 3});
    ASSERT_TRUE(t->get_partial_shape().same_scheme(PartialShape{2, Dimension::dynamic(), 3}));
}

NGRAPH_TEST(dynamic_${BACKEND_NAME}, abc)
{
    //
    // Create a graph for f(a,b,c) = (a+b)*c, where a, b, c all have shape {2,?,3}.
    //
    auto a = make_shared<op::Parameter>(element::f32, PartialShape{2, Dimension::dynamic(), 3});
    auto b = make_shared<op::Parameter>(element::f32, PartialShape{2, Dimension::dynamic(), 3});
    auto c = make_shared<op::Parameter>(element::f32, PartialShape{2, Dimension::dynamic(), 3});

    auto a_plus_b_times_c = (a + b) * c;

    auto f = make_shared<Function>(NodeVector{a_plus_b_times_c}, ParameterVector{a, b, c});

    //
    // Get a backend with dynamic support, and compile f.
    //
    auto backend = runtime::Backend::create("${BACKEND_NAME}", true);

    auto ex = backend->compile(f);

    //
    // Create a dynamic output tensor with shape {2,?,3}.
    //
    auto t_r =
        backend->create_dynamic_tensor(element::f32, PartialShape{2, Dimension::dynamic(), 3});

    //
    // For each of n=[0,...,5), run the compiled executable against a test vector of shape
    // {2,n,3}, and check the results.
    //
    for (size_t middle_dim = 0; middle_dim < 5; middle_dim++)
    {
        // Fill in some test input values, which we'll use for a, b, and c.
        vector<float> inputs(2 * middle_dim * 3);
        for (size_t i = 0; i < 2 * middle_dim * 3; i++)
        {
            inputs[i] = i;
        }

        // Create static tensors for the inputs and copy data.
        auto t_a = backend->create_tensor(element::f32, Shape{2, middle_dim, 3});
        auto t_b = backend->create_tensor(element::f32, Shape{2, middle_dim, 3});
        auto t_c = backend->create_tensor(element::f32, Shape{2, middle_dim, 3});

        copy_data(t_a, inputs);
        copy_data(t_b, inputs);
        copy_data(t_c, inputs);

        // Call ex, writing result into t_r (note we're using the same t_r from outside the loop.)
        ex->call_with_validate({t_r}, {t_a, t_b, t_c});

        // After call, t_r should have a shape of {2,n,3}.
        ASSERT_EQ(t_r->get_shape(), (Shape{2, middle_dim, 3}));

        // Read out the results, and compare them against expected values.
        auto results = read_vector<float>(t_r);

        vector<float> expected_values(2 * middle_dim * 3);
        for (size_t i = 0; i < 2 * middle_dim * 3; i++)
        {
            expected_values[i] = (i + i) * i;
        }

        EXPECT_TRUE(test::all_close_f(results, expected_values));
    }
}

NGRAPH_TEST(dynamic_${BACKEND_NAME}, transpose)
{
    //
    // Create a graph for f(x,perm) = Transpose(x,Convert<i64>(perm)). We'll do the permutation in
    // i32 and cast it to i64, just for fun (and to mirror the TensorFlow test I am porting here).
    //
    auto x = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto perm = make_shared<op::Parameter>(element::i32, PartialShape{Dimension::dynamic()});
    auto perm_i64 = make_shared<op::Convert>(perm, element::i64);

    auto x_transpose = make_shared<op::Transpose>(x, perm_i64);

    auto f = make_shared<Function>(NodeVector{x_transpose}, ParameterVector{x, perm});

    auto backend = runtime::Backend::create("${BACKEND_NAME}", true);

    auto ex = backend->compile(f);

    auto t_r = backend->create_dynamic_tensor(element::f32, PartialShape::dynamic());

    std::vector<Shape> x_shapes{Shape{2, 3}, Shape{2, 3}, Shape{2, 2, 3}};
    std::vector<std::vector<int32_t>> perms{{0, 1}, {1, 0}, {2, 1, 0}};
    std::vector<std::vector<float>> inputs{
        {1, 2, 3, 4, 5, 6}, {1, 2, 3, 4, 5, 6}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}};
    std::vector<Shape> expected_result_shapes{Shape{2, 3}, Shape{3, 2}, {3, 2, 2}};
    // Generated with numpy, so don't worry. :)
    std::vector<std::vector<float>> expected_results{
        {1, 2, 3, 4, 5, 6}, {1, 4, 2, 5, 3, 6}, {1, 7, 4, 10, 2, 8, 5, 11, 3, 9, 6, 12}};

    for (size_t i = 0; i < x_shapes.size(); i++)
    {
        auto t_x = backend->create_tensor(element::f32, x_shapes[i]);
        auto t_perm = backend->create_tensor(element::i32, Shape{perms[i].size()});

        copy_data(t_x, inputs[i]);
        copy_data(t_perm, perms[i]);

        ex->call_with_validate({t_r}, {t_x, t_perm});

        ASSERT_EQ(t_r->get_shape(), expected_result_shapes[i]);

        auto results = read_vector<float>(t_r);

        ASSERT_TRUE(test::all_close_f(results, expected_results[i], MIN_FLOAT_TOLERANCE_BITS));
    }
}

NGRAPH_TEST(dynamic_${BACKEND_NAME}, broadcast)
{
    // Create a graph for
    //   f(x,shape:i32,axes:32) = Broadcast(x,Convert<i64>(shape),Convert<i64>(axes)).
    auto x = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto shape = make_shared<op::Parameter>(element::i32, PartialShape{Dimension::dynamic()});
    auto axes = make_shared<op::Parameter>(element::i32, PartialShape{Dimension::dynamic()});
    auto shape_i64 = make_shared<op::Convert>(shape, element::i64);
    auto axes_i64 = make_shared<op::Convert>(axes, element::i64);

    auto bc = make_shared<op::DynBroadcast>(x, shape_i64, axes_i64);

    auto f = make_shared<Function>(NodeVector{bc}, ParameterVector{x, shape, axes});

    auto backend = runtime::Backend::create("${BACKEND_NAME}", true);

    auto ex = backend->compile(f);

    auto t_r = backend->create_dynamic_tensor(element::f32, PartialShape::dynamic());

    std::vector<Shape> x_shapes{Shape{}, Shape{}, Shape{2}, Shape{2}};
    std::vector<std::vector<int32_t>> shapes{{2, 2}, {2, 2, 2}, {3, 2}, {2, 3}};
    std::vector<std::vector<int32_t>> axeses{{0, 1}, {0, 1, 2}, {0}, {1}};
    std::vector<std::vector<float>> inputs{{6}, {7}, {10, 11}, {10, 11}};
    std::vector<Shape> expected_result_shapes{
        Shape{2, 2}, Shape{2, 2, 2}, Shape{3, 2}, Shape{2, 3}};
    std::vector<std::vector<float>> expected_results{
        {6, 6, 6, 6}, {7, 7, 7, 7, 7, 7, 7, 7}, {10, 11, 10, 11, 10, 11}, {10, 10, 10, 11, 11, 11}};

    for (size_t i = 0; i < x_shapes.size(); i++)
    {
        auto t_x = backend->create_tensor(element::f32, x_shapes[i]);
        auto t_shape = backend->create_tensor(element::i32, Shape{shapes[i].size()});
        auto t_axes = backend->create_tensor(element::i32, Shape{axeses[i].size()});

        copy_data(t_x, inputs[i]);
        copy_data(t_shape, shapes[i]);
        copy_data(t_axes, axeses[i]);

        ex->call_with_validate({t_r}, {t_x, t_shape, t_axes});

        ASSERT_EQ(t_r->get_shape(), expected_result_shapes[i]);

        auto results = read_vector<float>(t_r);

        ASSERT_TRUE(test::all_close_f(results, expected_results[i], MIN_FLOAT_TOLERANCE_BITS));
    }
}

NGRAPH_TEST(dynamic_${BACKEND_NAME}, sum)
{
    // Create a graph for f(x,axes:int32) = Sum(x,Convert<int64>(axes)).
    auto x = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto axes = make_shared<op::Parameter>(element::i32, PartialShape{Dimension::dynamic()});
    auto axes_i64 = make_shared<op::Convert>(axes, element::i64);

    auto sum = make_shared<op::Sum>(x, axes_i64);
    ASSERT_TRUE(sum->get_output_partial_shape(0).rank().is_dynamic());

    auto f = make_shared<Function>(NodeVector{sum}, ParameterVector{x, axes});

    auto backend = runtime::Backend::create("${BACKEND_NAME}", true);

    auto ex = backend->compile(f);

    auto t_r = backend->create_dynamic_tensor(element::f32, PartialShape::dynamic());

    std::vector<Shape> x_shapes{
        Shape{2, 3}, Shape{2, 3}, Shape{2, 3}, Shape{2, 3}, Shape{5}, Shape{5}};
    std::vector<std::vector<int32_t>> axeses{{}, {0}, {1}, {0, 1}, {}, {0}};
    std::vector<std::vector<float>> inputs{{1, 2, 3, 4, 5, 6},
                                           {1, 2, 3, 4, 5, 6},
                                           {1, 2, 3, 4, 5, 6},
                                           {1, 2, 3, 4, 5, 6},
                                           {1, 2, 3, 4, 5},
                                           {1, 2, 3, 4, 5}};
    std::vector<Shape> expected_result_shapes{
        Shape{2, 3}, Shape{3}, Shape{2}, Shape{}, Shape{5}, Shape{}};
    std::vector<std::vector<float>> expected_results{
        {1, 2, 3, 4, 5, 6}, {5, 7, 9}, {6, 15}, {21}, {1, 2, 3, 4, 5}, {15}};

    for (size_t i = 0; i < x_shapes.size(); i++)
    {
        auto t_x = backend->create_tensor(element::f32, x_shapes[i]);
        auto t_axes = backend->create_tensor(element::i32, Shape{axeses[i].size()});

        copy_data(t_x, inputs[i]);
        copy_data(t_axes, axeses[i]);

        ex->call_with_validate({t_r}, {t_x, t_axes});

        ASSERT_EQ(t_r->get_shape(), expected_result_shapes[i]);

        auto results = read_vector<float>(t_r);

        ASSERT_TRUE(test::all_close_f(results, expected_results[i], MIN_FLOAT_TOLERANCE_BITS));
    }
}

NGRAPH_TEST(dynamic_${BACKEND_NAME}, all)
{
    // Create a graph for f(x,axes:int32) = All(x,Convert<int64>(axes)).
    auto x = make_shared<op::Parameter>(element::boolean, PartialShape::dynamic());
    auto axes = make_shared<op::Parameter>(element::i32, PartialShape{Dimension::dynamic()});
    auto axes_i64 = make_shared<op::Convert>(axes, element::i64);

    auto all = make_shared<op::All>(x, axes_i64);
    ASSERT_TRUE(all->get_output_partial_shape(0).rank().is_dynamic());

    auto f = make_shared<Function>(NodeVector{all}, ParameterVector{x, axes});

    auto backend = runtime::Backend::create("${BACKEND_NAME}", true);

    auto ex = backend->compile(f);

    auto t_r = backend->create_dynamic_tensor(element::boolean, PartialShape::dynamic());

    std::vector<Shape> x_shapes{
        Shape{2, 3}, Shape{2, 3}, Shape{2, 3}, Shape{2, 3}, Shape{5}, Shape{5}};
    std::vector<std::vector<int32_t>> axeses{{}, {0}, {1}, {0, 1}, {}, {0}};
    std::vector<std::vector<char>> inputs{{1, 0, 1, 0, 1, 0},
                                          {1, 0, 1, 0, 0, 1},
                                          {1, 0, 1, 1, 1, 1},
                                          {1, 0, 1, 0, 1, 0},
                                          {1, 0, 1, 0, 1},
                                          {1, 0, 1, 0, 1}};
    std::vector<Shape> expected_result_shapes{
        Shape{2, 3}, Shape{3}, Shape{2}, Shape{}, Shape{5}, Shape{}};
    std::vector<std::vector<char>> expected_results{
        {1, 0, 1, 0, 1, 0}, {0, 0, 1}, {0, 1}, {0}, {1, 0, 1, 0, 1}, {0}};

    for (size_t i = 0; i < x_shapes.size(); i++)
    {
        auto t_x = backend->create_tensor(element::boolean, x_shapes[i]);
        auto t_axes = backend->create_tensor(element::i32, Shape{axeses[i].size()});

        copy_data(t_x, inputs[i]);
        copy_data(t_axes, axeses[i]);

        ex->call_with_validate({t_r}, {t_x, t_axes});

        ASSERT_EQ(t_r->get_shape(), expected_result_shapes[i]);

        auto results = read_vector<char>(t_r);

        ASSERT_EQ(results, expected_results[i]);
    }
}

template <typename T>
struct RangeTest
{
    T start;
    T stop;
    T step;
    Shape expected_result_shape;
    std::vector<T> expected_result;
};

// TODO(amprocte): We should test this with more than just int32, but there is a bug in the
// handling of element type-changing that is currently blocking doing that easily.
NGRAPH_TEST(dynamic_${BACKEND_NAME}, range)
{
    // Create a graph for f(start,stop,step) = Range(start,stop,step).
    auto start = make_shared<op::Parameter>(element::i32, Shape{});
    auto stop = make_shared<op::Parameter>(element::i32, Shape{});
    auto step = make_shared<op::Parameter>(element::i32, Shape{});

    auto range = make_shared<op::Range>(start, stop, step);
    ASSERT_TRUE(range->get_output_partial_shape(0).same_scheme(PartialShape::dynamic(1)));

    auto f = make_shared<Function>(NodeVector{range}, ParameterVector{start, stop, step});

    auto backend = runtime::Backend::create("${BACKEND_NAME}", true);

    auto ex = backend->compile(f);

    auto t_r = backend->create_dynamic_tensor(element::i32, PartialShape::dynamic());

    std::vector<RangeTest<int32_t>> int32_tests = {
        RangeTest<int32_t>{0, 10, 1, Shape{10}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}},
        RangeTest<int32_t>{-5, 6, 3, Shape{4}, {-5, -2, 1, 4}},
        RangeTest<int32_t>{10, 0, 1, Shape{0}, {}},
        RangeTest<int32_t>{10, 5, -3, Shape{2}, {10, 7}}};

    for (auto& test : int32_tests)
    {
        auto t_start = backend->create_tensor(element::i32, Shape{});
        auto t_stop = backend->create_tensor(element::i32, Shape{});
        auto t_step = backend->create_tensor(element::i32, Shape{});

        copy_data(t_start, std::vector<int32_t>{test.start});
        copy_data(t_stop, std::vector<int32_t>{test.stop});
        copy_data(t_step, std::vector<int32_t>{test.step});

        ex->call_with_validate({t_r}, {t_start, t_stop, t_step});

        ASSERT_EQ(t_r->get_element_type(), element::i32);
        ASSERT_EQ(t_r->get_shape(), test.expected_result_shape);

        auto results = read_vector<int32_t>(t_r);

        ASSERT_EQ(results, test.expected_result);
    }
}

NGRAPH_TEST(dynamic_${BACKEND_NAME}, reshape)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}", true);

    auto build_graph = [&backend](bool zero_flag) {
        // Create a graph for f(x,shape) = DynReshape(x,shape,zero_flag=zero_flag).
        auto x = make_shared<op::Parameter>(element::i32, PartialShape::dynamic());
        auto shape = make_shared<op::Parameter>(element::i64, PartialShape::dynamic(1));

        auto dyn_reshape = make_shared<op::DynReshape>(x, shape, zero_flag);
        EXPECT_TRUE(dyn_reshape->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));

        auto f = make_shared<Function>(NodeVector{dyn_reshape}, ParameterVector{x, shape});

        auto ex = backend->compile(f);

        return ex;
    };

    auto t_r = backend->create_dynamic_tensor(element::i32, PartialShape::dynamic());

    auto ex_flag_off = build_graph(false);
    auto ex_flag_on = build_graph(true);

    std::vector<std::tuple<bool, Shape, std::vector<int32_t>, std::vector<int64_t>, Shape>> tests;

    tests.emplace_back(make_tuple(
        false, Shape{2, 3}, vector<int32_t>{1, 2, 3, 4, 5, 6}, vector<int64_t>{6}, Shape{6}));
    tests.emplace_back(make_tuple(
        true, Shape{2, 3}, vector<int32_t>{1, 2, 3, 4, 5, 6}, vector<int64_t>{6}, Shape{6}));
    tests.emplace_back(make_tuple(
        false, Shape{2, 3}, vector<int32_t>{1, 2, 3, 4, 5, 6}, vector<int64_t>{-1}, Shape{6}));
    tests.emplace_back(make_tuple(false,
                                  Shape{2, 3},
                                  vector<int32_t>{1, 2, 3, 4, 5, 6},
                                  vector<int64_t>{2, -1},
                                  Shape{2, 3}));
    tests.emplace_back(make_tuple(false,
                                  Shape{2, 3},
                                  vector<int32_t>{1, 2, 3, 4, 5, 6},
                                  vector<int64_t>{3, -1},
                                  Shape{3, 2}));
    tests.emplace_back(make_tuple(false,
                                  Shape{2, 3},
                                  vector<int32_t>{1, 2, 3, 4, 5, 6},
                                  vector<int64_t>{3, 2, -1},
                                  Shape{3, 2, 1}));
    tests.emplace_back(make_tuple(true,
                                  Shape{2, 3},
                                  vector<int32_t>{1, 2, 3, 4, 5, 6},
                                  vector<int64_t>{3, 2, -1},
                                  Shape{3, 2, 1}));
    tests.emplace_back(make_tuple(true,
                                  Shape{2, 3},
                                  vector<int32_t>{1, 2, 3, 4, 5, 6},
                                  vector<int64_t>{0, 0, -1},
                                  Shape{2, 3, 1}));
    tests.emplace_back(make_tuple(true,
                                  Shape{2, 3},
                                  vector<int32_t>{1, 2, 3, 4, 5, 6},
                                  vector<int64_t>{2, 0, -1},
                                  Shape{2, 3, 1}));
    tests.emplace_back(make_tuple(
        true, Shape{0, 3, 4}, vector<int32_t>{}, vector<int64_t>{3, -1, 2}, Shape{3, 0, 2}));

    for (auto& test : tests)
    {
        bool zero_flag = get<0>(test);
        const Shape& in_shape = get<1>(test);
        const std::vector<int32_t>& data = get<2>(test);
        const std::vector<int64_t>& dims = get<3>(test);
        const Shape& out_shape = get<4>(test);

        auto t_x = backend->create_tensor(element::i32, in_shape);
        auto t_shape = backend->create_tensor(element::i64, Shape{dims.size()});

        copy_data(t_x, data);
        copy_data(t_shape, dims);

        auto ex = zero_flag ? ex_flag_on : ex_flag_off;
        ex->call_with_validate({t_r}, {t_x, t_shape});

        ASSERT_EQ(t_r->get_element_type(), element::i32);
        ASSERT_EQ(t_r->get_shape(), out_shape);

        auto results = read_vector<int32_t>(t_r);

        ASSERT_EQ(results, data);
    }
}

static void axpy_test(const PartialShape& input_pshape, const std::vector<Shape>& input_shapes)
{
    auto a = make_shared<op::Parameter>(element::f32, input_pshape);
    auto x = make_shared<op::Parameter>(element::f32, input_pshape);
    auto y = make_shared<op::Parameter>(element::f32, input_pshape);

    auto axpy = a * x + y;

    auto f = make_shared<Function>(NodeVector{axpy}, ParameterVector{a, x, y});
    auto backend = runtime::Backend::create("${BACKEND_NAME}", true);
    auto ex = backend->compile(f);

    auto t_r = backend->create_dynamic_tensor(element::f32, input_pshape);

    for (auto& shape : input_shapes)
    {
        vector<float> inputs(shape_size(shape));
        for (size_t i = 0; i < shape_size(shape); i++)
        {
            inputs[i] = i;
        }

        auto t_a = backend->create_tensor(element::f32, shape);
        auto t_x = backend->create_tensor(element::f32, shape);
        auto t_y = backend->create_tensor(element::f32, shape);

        copy_data(t_a, inputs);
        copy_data(t_x, inputs);
        copy_data(t_y, inputs);

        ex->call_with_validate({t_r}, {t_a, t_x, t_y});

        ASSERT_EQ(t_r->get_shape(), shape);

        auto results = read_vector<float>(t_r);

        vector<float> expected_values(shape_size(shape));
        for (size_t i = 0; i < shape_size(shape); i++)
        {
            expected_values[i] = (i * i) + i;
        }

        EXPECT_TRUE(test::all_close_f(results, expected_values));
    }
}

NGRAPH_TEST(dynamic_${BACKEND_NAME}, axpy)
{
    // Test with shape {?, 3, 3}.
    axpy_test(PartialShape{Dimension::dynamic(), 3, 3}, {Shape{2, 3, 3}, Shape{5, 3, 3}});

    // Test with shape {?, ?, ?}.
    axpy_test(PartialShape::dynamic(3),
              {Shape{2, 3, 3}, Shape{5, 3, 3}, Shape{2, 5, 2}, Shape{8, 1, 8}});

    // Test with shape ?. (Rank unknown.)
    axpy_test(PartialShape::dynamic(),
              {Shape{2, 3, 3},
               Shape{5, 3, 3},
               Shape{2, 5, 2},
               Shape{8, 1, 8},
               Shape{5},
               Shape{8, 2},
               Shape{8, 2, 8, 2},
               Shape{2, 3, 4, 5, 2}});
}

static void to_vector_test(const PartialShape& input_pshape, const std::vector<Shape>& input_shapes)
{
    auto x = make_shared<op::Parameter>(element::f32, input_pshape);

    shared_ptr<Node> x_new_shape = make_shared<op::ShapeOf>(x);
    x_new_shape = make_shared<op::Product>(x_new_shape, AxisSet{0});
    x_new_shape = make_shared<op::Reshape>(x_new_shape, AxisVector{}, Shape{1});

    auto x_reshaped = make_shared<op::DynReshape>(x, x_new_shape);

    auto f = make_shared<Function>(NodeVector{x_reshaped}, ParameterVector{x});
    auto backend = runtime::Backend::create("${BACKEND_NAME}", true);
    auto ex = backend->compile(f);

    auto t_r = backend->create_dynamic_tensor(element::f32, PartialShape::dynamic(1));

    for (auto& shape : input_shapes)
    {
        vector<float> inputs(shape_size(shape));
        for (size_t i = 0; i < shape_size(shape); i++)
        {
            inputs[i] = i;
        }

        auto t_x = backend->create_tensor(element::f32, shape);

        copy_data(t_x, inputs);

        ex->call_with_validate({t_r}, {t_x});

        ASSERT_EQ(t_r->get_shape(), (Shape{shape_size(shape)}));

        auto results = read_vector<float>(t_r);

        EXPECT_TRUE(test::all_close_f(results, inputs));
    }
}

NGRAPH_TEST(dynamic_${BACKEND_NAME}, to_vector)
{
    // Test with shape {?, 3, 3}.
    to_vector_test(PartialShape{Dimension::dynamic(), 3, 3}, {Shape{2, 3, 3}, Shape{5, 3, 3}});

    // Test with shape {?, ?, ?}.
    to_vector_test(PartialShape::dynamic(3),
                   {Shape{2, 3, 3}, Shape{5, 3, 3}, Shape{2, 5, 2}, Shape{8, 1, 8}});

    // Test with shape ?. (Rank unknown.)
    to_vector_test(PartialShape::dynamic(),
                   {Shape{2, 3, 3},
                    Shape{5, 3, 3},
                    Shape{2, 5, 2},
                    Shape{8, 1, 8},
                    Shape{5},
                    Shape{8, 2},
                    Shape{8, 2, 8, 2},
                    Shape{2, 3, 4, 5, 2}});
}

static void reverse_shape_test(const PartialShape& input_pshape,
                               const std::vector<Shape>& input_shapes)
{
    auto x = make_shared<op::Parameter>(element::f32, input_pshape);

    shared_ptr<Node> x_new_shape = make_shared<op::ShapeOf>(x);
    x_new_shape = make_shared<op::Reverse>(x_new_shape, AxisSet{0});

    auto x_reshaped = make_shared<op::DynReshape>(x, x_new_shape);

    auto f = make_shared<Function>(NodeVector{x_reshaped}, ParameterVector{x});
    auto backend = runtime::Backend::create("${BACKEND_NAME}", true);
    auto ex = backend->compile(f);

    auto t_r = backend->create_dynamic_tensor(element::f32, PartialShape::dynamic());

    for (auto& shape : input_shapes)
    {
        vector<float> inputs(shape_size(shape));
        for (size_t i = 0; i < shape_size(shape); i++)
        {
            inputs[i] = i;
        }

        auto t_x = backend->create_tensor(element::f32, shape);

        copy_data(t_x, inputs);

        ex->call_with_validate({t_r}, {t_x});

        Shape expected_shape = shape;
        std::reverse(expected_shape.begin(), expected_shape.end());
        ASSERT_EQ(t_r->get_shape(), expected_shape);

        auto results = read_vector<float>(t_r);

        EXPECT_TRUE(test::all_close_f(results, inputs));
    }
}

NGRAPH_TEST(dynamic_${BACKEND_NAME}, reverse_shape)
{
    // Test with shape {?, 3, 3}.
    reverse_shape_test(PartialShape{Dimension::dynamic(), 3, 3}, {Shape{2, 3, 3}, Shape{5, 3, 3}});

    // Test with shape {?, ?, ?}.
    reverse_shape_test(PartialShape::dynamic(3),
                       {Shape{2, 3, 3}, Shape{5, 3, 3}, Shape{2, 5, 2}, Shape{8, 1, 8}});

    // Test with shape ?. (Rank unknown.)
    reverse_shape_test(PartialShape::dynamic(),
                       {Shape{2, 3, 3},
                        Shape{5, 3, 3},
                        Shape{2, 5, 2},
                        Shape{8, 1, 8},
                        Shape{5},
                        Shape{8, 2},
                        Shape{8, 2, 8, 2},
                        Shape{2, 3, 4, 5, 2}});
}
