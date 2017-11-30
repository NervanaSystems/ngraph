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
#include "gtest/gtest.h"

#include "ngraph/ngraph.hpp"
#include "util/all_close.hpp"

using namespace ngraph;
using namespace ngraph::test;
using namespace std;

template <typename T>
static void copy_data(shared_ptr<runtime::TensorView> tv, const vector<T>& data)
{
    size_t data_size = data.size() * sizeof(T);
    tv->write(data.data(), 0, data_size);
}

std::shared_ptr<ngraph::runtime::TensorView> make_reduce_result(
    std::function<std::shared_ptr<Node>(const std::shared_ptr<Node>&, const AxisSet&)> func)
{
    auto shape_a = Shape{3, 2};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape_a);
    auto shape_rt = Shape{2};
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_rt);
    auto f = make_shared<Function>(func(A, {0}), rt, op::Parameters{A});
    auto manager = runtime::Manager::get("NGVM");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);
    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape_rt);
    cf->call({a}, {result});

    return result;
}

std::shared_ptr<ngraph::runtime::TensorView> make_reduce_result_true(
    std::function<std::shared_ptr<Node>(const std::shared_ptr<Node>&, const AxisSet&, bool)> func)
{
    auto shape_a = Shape{3, 2};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape_a);
    auto shape_rt = Shape{2};
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_rt);
    auto f = make_shared<Function>(func(A, {0}, true), rt, op::Parameters{A});
    auto manager = runtime::Manager::get("NGVM");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);
    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape_rt);
    cf->call({a}, {result});

    return result;
}

std::shared_ptr<ngraph::runtime::TensorView> make_reduce_result_false(
    std::function<std::shared_ptr<Node>(const std::shared_ptr<Node>&, const AxisSet&, bool)> func)
{
    auto shape_a = Shape{3, 2};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape_a);
    auto shape_rt = Shape{2};
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_rt);
    auto f = make_shared<Function>(func(A, {0}, false), rt, op::Parameters{A});
    auto manager = runtime::Manager::get("NGVM");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);
    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape_rt);
    cf->call({a}, {result});

    return result;
}

TEST(builder, l2_norm)
{
    auto result = make_reduce_result(builder::l2_norm);
    ASSERT_TRUE(
        all_close((vector<float>{5.9160797831f, 7.48331477355f}), result->get_vector<float>()));
}

TEST(builder, mean)
{
    auto result = make_reduce_result(builder::mean);
    ASSERT_TRUE(all_close((vector<float>{3, 4}), result->get_vector<float>()));
}

TEST(builder, std_dev)
{
    auto result = make_reduce_result_false(builder::std_dev);
    ASSERT_TRUE(
        all_close((vector<float>{1.63299316186f, 1.63299316186f}), result->get_vector<float>()));
    result = make_reduce_result_true(builder::std_dev);
    ASSERT_TRUE(all_close((vector<float>{2, 2}), result->get_vector<float>()));
}

TEST(builder, variance)
{
    auto result = make_reduce_result_false(builder::variance);
    ASSERT_TRUE(
        all_close((vector<float>{2.66666666666f, 2.66666666666f}), result->get_vector<float>()));
    result = make_reduce_result_true(builder::variance);
    ASSERT_TRUE(all_close((vector<float>{4, 4}), result->get_vector<float>()));
}

TEST(builder, numpy_transpose)
{
    // 2D Transpose
    Shape shape{2, 4};
    auto param = std::make_shared<op::Parameter>(ngraph::element::Float32::element_type(), shape);
    auto transposed = std::dynamic_pointer_cast<op::Reshape>(builder::numpy_transpose(param));
    EXPECT_EQ(Shape({4, 2}), transposed->get_output_shape());

    // Multidimensional Transpose
    shape = Shape{2, 4, 8};
    param = std::make_shared<op::Parameter>(ngraph::element::Float32::element_type(), shape);
    transposed = std::dynamic_pointer_cast<op::Reshape>(builder::numpy_transpose(param));
    EXPECT_EQ(Shape({8, 4, 2}), transposed->get_output_shape());

    // Dimshuffle
    shape = Shape{2, 4, 8};
    param = std::make_shared<op::Parameter>(ngraph::element::Float32::element_type(), shape);
    transposed = std::dynamic_pointer_cast<op::Reshape>(
        builder::numpy_transpose(param, AxisVector{2, 0, 1}));
    EXPECT_EQ(Shape({8, 2, 4}), transposed->get_output_shape());

    // Bad Orders
    EXPECT_ANY_THROW(
        std::dynamic_pointer_cast<op::Reshape>(builder::numpy_transpose(param, AxisVector{2})));
    EXPECT_ANY_THROW(std::dynamic_pointer_cast<op::Reshape>(
        builder::numpy_transpose(param, AxisVector{2, 2, 1})));
}
