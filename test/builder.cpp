/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/
#include "gtest/gtest.h"

#include "ngraph/ngraph.hpp"
#include "util/all_close.hpp"
#include "util/test_tools.hpp"

using namespace ngraph;
using namespace std;

shared_ptr<runtime::TensorView>
    make_reduce_result(function<shared_ptr<Node>(const shared_ptr<Node>&, const AxisSet&)> func)
{
    Shape shape_a{3, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{2};
    auto f = make_shared<Function>(func(A, {0}), op::ParameterVector{A});
    auto manager = runtime::Manager::get("INTERPRETER");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);
    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto result = backend->make_primary_tensor_view(element::f32, shape_rt);
    cf->call({result}, {a});

    return result;
}

shared_ptr<runtime::TensorView> make_reduce_result_true(
    function<shared_ptr<Node>(const shared_ptr<Node>&, const AxisSet&, bool)> func)
{
    Shape shape_a{3, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{2};
    auto f = make_shared<Function>(func(A, {0}, true), op::ParameterVector{A});
    auto manager = runtime::Manager::get("INTERPRETER");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);
    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto result = backend->make_primary_tensor_view(element::f32, shape_rt);
    cf->call({result}, {a});

    return result;
}

shared_ptr<runtime::TensorView> make_reduce_result_false(
    function<shared_ptr<Node>(const shared_ptr<Node>&, const AxisSet&, bool)> func)
{
    Shape shape_a{3, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{2};
    auto f = make_shared<Function>(func(A, {0}, false), op::ParameterVector{A});
    auto manager = runtime::Manager::get("INTERPRETER");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);
    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto result = backend->make_primary_tensor_view(element::f32, shape_rt);
    cf->call({result}, {a});

    return result;
}

TEST(builder, l2_norm)
{
    auto result = make_reduce_result(builder::l2_norm);
    ASSERT_TRUE(test::all_close((vector<float>{5.9160797831f, 7.48331477355f}),
                                read_vector<float>(result)));
}

TEST(builder, mean)
{
    auto result = make_reduce_result(builder::mean);
    ASSERT_TRUE(test::all_close((vector<float>{3, 4}), read_vector<float>(result)));
}

TEST(builder, std_dev)
{
    auto result = make_reduce_result_false(builder::std_dev);
    ASSERT_TRUE(test::all_close((vector<float>{1.63299316186f, 1.63299316186f}),
                                read_vector<float>(result)));
    result = make_reduce_result_true(builder::std_dev);
    ASSERT_TRUE(test::all_close((vector<float>{2, 2}), read_vector<float>(result)));
}

TEST(builder, variance)
{
    auto result = make_reduce_result_false(builder::variance);
    ASSERT_TRUE(test::all_close((vector<float>{2.66666666666f, 2.66666666666f}),
                                read_vector<float>(result)));
    result = make_reduce_result_true(builder::variance);
    ASSERT_TRUE(test::all_close((vector<float>{4, 4}), read_vector<float>(result)));
}

TEST(builder, numpy_transpose)
{
    // 2D Transpose
    Shape shape{2, 4};
    auto param = make_shared<op::Parameter>(element::f32, shape);
    auto transposed = dynamic_pointer_cast<op::Reshape>(builder::numpy_transpose(param));
    EXPECT_EQ(Shape({4, 2}), transposed->get_output_shape());

    // Multidimensional Transpose
    shape = Shape{2, 4, 8};
    param = make_shared<op::Parameter>(element::f32, shape);
    transposed = dynamic_pointer_cast<op::Reshape>(builder::numpy_transpose(param));
    EXPECT_EQ(Shape({8, 4, 2}), transposed->get_output_shape());

    // Dimshuffle
    shape = Shape{2, 4, 8};
    param = make_shared<op::Parameter>(element::f32, shape);
    transposed =
        dynamic_pointer_cast<op::Reshape>(builder::numpy_transpose(param, AxisVector{2, 0, 1}));
    EXPECT_EQ(Shape({8, 2, 4}), transposed->get_output_shape());

    // Bad Orders
    EXPECT_ANY_THROW(
        dynamic_pointer_cast<op::Reshape>(builder::numpy_transpose(param, AxisVector{2})));
    EXPECT_ANY_THROW(
        dynamic_pointer_cast<op::Reshape>(builder::numpy_transpose(param, AxisVector{2, 2, 1})));
}

TEST(builder, tensor_mask)
{
    Shape max_sequence_length{3};
    auto sequence_lengths = make_shared<op::Parameter>(element::u32, max_sequence_length);

    Shape mask_shape{3, 5};
    auto f = make_shared<Function>(builder::tensor_mask(sequence_lengths, 1, 0, mask_shape),
                                   op::ParameterVector{sequence_lengths});

    auto manager = runtime::Manager::get("INTERPRETER");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    auto sequence_lengths_data =
        backend->make_primary_tensor_view(element::u32, max_sequence_length);
    copy_data(sequence_lengths_data, vector<uint32_t>{1, 3, 2});
    auto result = backend->make_primary_tensor_view(element::boolean, mask_shape);

    cf->call({result}, {sequence_lengths_data});
    vector<char> expected{1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0};

    EXPECT_EQ(expected, read_vector<char>(result));
}

TEST(builder, rotate_left)
{
    Shape shape{3};
    auto A = make_shared<op::Parameter>(element::u32, shape);

    auto f = make_shared<Function>(builder::rotate_left(A, {1}), op::ParameterVector{A});

    auto manager = runtime::Manager::get("INTERPRETER");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    auto a = backend->make_primary_tensor_view(element::u32, shape);
    copy_data(a, vector<uint32_t>{1, 2, 3});
    auto result = backend->make_primary_tensor_view(element::u32, shape);

    cf->call({result}, {a});
    vector<uint32_t> expected{2, 3, 1};

    EXPECT_EQ(expected, read_vector<uint32_t>(result));
}

TEST(builder, over_rotate_left)
{
    Shape shape{3};
    auto A = make_shared<op::Parameter>(element::u32, shape);

    // 5 rotations on a size 3 axis = 2 rotations
    auto f = make_shared<Function>(builder::rotate_left(A, {5}), op::ParameterVector{A});

    auto manager = runtime::Manager::get("INTERPRETER");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    auto a = backend->make_primary_tensor_view(element::u32, shape);
    copy_data(a, vector<uint32_t>{3, 1, 2});
    auto result = backend->make_primary_tensor_view(element::u32, shape);

    cf->call({result}, {a});
    vector<uint32_t> expected{2, 3, 1};

    EXPECT_EQ(expected, read_vector<uint32_t>(result));
}

TEST(builder, rotate_left_2D)
{
    Shape shape{2, 3};
    auto A = make_shared<op::Parameter>(element::u32, shape);

    auto f = make_shared<Function>(builder::rotate_left(A, {0, 2}), op::ParameterVector{A});

    auto manager = runtime::Manager::get("INTERPRETER");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    auto a = backend->make_primary_tensor_view(element::u32, shape);
    copy_data(a, vector<uint32_t>{1, 2, 3, 4, 5, 6});
    auto result = backend->make_primary_tensor_view(element::u32, shape);

    cf->call({result}, {a});
    vector<uint32_t> expected{3, 1, 2, 6, 4, 5};

    EXPECT_EQ(expected, read_vector<uint32_t>(result));
}

TEST(builder, rotate_left_multi)
{
    Shape shape{2, 3};
    auto A = make_shared<op::Parameter>(element::u32, shape);

    auto f = make_shared<Function>(builder::rotate_left(A, {1, 2}), op::ParameterVector{A});

    auto manager = runtime::Manager::get("INTERPRETER");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    auto a = backend->make_primary_tensor_view(element::u32, shape);
    copy_data(a, vector<uint32_t>{1, 2, 3, 4, 5, 6});
    auto result = backend->make_primary_tensor_view(element::u32, shape);

    cf->call({result}, {a});
    vector<uint32_t> expected{6, 4, 5, 3, 1, 2};

    EXPECT_EQ(expected, read_vector<uint32_t>(result));
}
