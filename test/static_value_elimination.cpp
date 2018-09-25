//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
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
#include "ngraph/pass/static_value_elimination.hpp"
#include "util/test_tools.hpp"

using namespace ngraph;
using namespace std;

TEST(static_value_elimination, eliminate_get_shape)
{
    // Make sure GetShape is being eliminated.
    Shape shape_a{2, 4, 6, 8};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto f = make_shared<Function>(make_shared<op::GetShape>(A), op::ParameterVector{A});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::StaticValueElimination>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);
    ASSERT_EQ(count_ops_of_type<op::GetShape>(f), 0);

    // Make sure result is correct.
    auto backend = runtime::Backend::create("INTERPRETER");

    auto a = backend->create_tensor(element::f32, shape_a);
    vector<float> a_data(2 * 4 * 6 * 8, 42);
    copy_data(a, a_data);
    auto result = backend->create_tensor(element::i64, Shape{4});

    backend->call_with_validate(f, {result}, {a});
    ASSERT_EQ((vector<int64_t>{2, 4, 6, 8}), read_vector<int64_t>(result));
}

TEST(static_value_elimination, eliminate_convert)
{
    // Make sure GetShape and Convert are being eliminated.
    Shape shape_a{2, 4, 6, 8};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto f =
        make_shared<Function>(make_shared<op::Convert>(make_shared<op::GetShape>(A), element::f32),
                              op::ParameterVector{A});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::StaticValueElimination>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);
    ASSERT_EQ(count_ops_of_type<op::GetShape>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Convert>(f), 0);

    // Make sure result is correct.
    auto backend = runtime::Backend::create("INTERPRETER");

    auto a = backend->create_tensor(element::f32, shape_a);
    vector<float> a_data(2 * 4 * 6 * 8, 42);
    copy_data(a, a_data);
    auto result = backend->create_tensor(element::f32, Shape{4});

    backend->call_with_validate(f, {result}, {a});
    ASSERT_EQ((vector<float>{2, 4, 6, 8}), read_vector<float>(result));
}

TEST(static_value_elimination, eliminate_reshape)
{
    // Make sure GetShape and Reshape are being eliminated.
    Shape shape_a{2, 4, 6, 8};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto f = make_shared<Function>(
        make_shared<op::Reshape>(make_shared<op::GetShape>(A), AxisVector{0}, Shape{4}),
        op::ParameterVector{A});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::StaticValueElimination>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);
    ASSERT_EQ(count_ops_of_type<op::GetShape>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Reshape>(f), 0);

    // Make sure result is correct.
    auto backend = runtime::Backend::create("INTERPRETER");

    auto a = backend->create_tensor(element::f32, shape_a);
    vector<float> a_data(2 * 4 * 6 * 8, 42);
    copy_data(a, a_data);
    auto result = backend->create_tensor(element::i64, Shape{4});

    backend->call_with_validate(f, {result}, {a});
    ASSERT_EQ((vector<int64_t>{2, 4, 6, 8}), read_vector<int64_t>(result));
}

TEST(static_value_elimination, eliminate_dyn_reshape)
{
    // Make sure GetShape and DynReshape are being eliminated.
    Shape shape_a{2, 4, 6, 8};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);

    std::vector<int64_t> k_data{4};
    auto K = std::make_shared<op::Constant>(element::i64, Shape{1}, k_data.data());

    auto f = make_shared<Function>(make_shared<op::DynReshape>(make_shared<op::GetShape>(A), K),
                                   op::ParameterVector{A});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::StaticValueElimination>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);
    ASSERT_EQ(count_ops_of_type<op::GetShape>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::DynReshape>(f), 0);

    // Make sure result is correct.
    auto backend = runtime::Backend::create("INTERPRETER");

    auto a = backend->create_tensor(element::f32, shape_a);
    vector<float> a_data(2 * 4 * 6 * 8, 42);
    copy_data(a, a_data);
    auto result = backend->create_tensor(element::i64, Shape{4});

    backend->call_with_validate(f, {result}, {a});
    ASSERT_EQ((vector<int64_t>{2, 4, 6, 8}), read_vector<int64_t>(result));
}

TEST(static_value_elimination, eliminate_slice)
{
    // Make sure GetShape and Slice are being eliminated.
    Shape shape_a{2, 4, 6, 8};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto f = make_shared<Function>(
        make_shared<op::Slice>(make_shared<op::GetShape>(A), Shape{1}, Shape{3}, Strides{1}),
        op::ParameterVector{A});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::StaticValueElimination>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);
    ASSERT_EQ(count_ops_of_type<op::GetShape>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Slice>(f), 0);

    // Make sure result is correct.
    auto backend = runtime::Backend::create("INTERPRETER");

    auto a = backend->create_tensor(element::f32, shape_a);
    vector<float> a_data(2 * 4 * 6 * 8, 42);
    copy_data(a, a_data);
    auto result = backend->create_tensor(element::i64, Shape{2});

    backend->call_with_validate(f, {result}, {a});
    ASSERT_EQ((vector<int64_t>{4, 6}), read_vector<int64_t>(result));
}

TEST(static_value_elimination, eliminate_concat)
{
    // Make sure GetShape and Concat are being eliminated.
    Shape shape_a{2, 4, 6, 8};
    Shape shape_b{3, 5, 7};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    auto f = make_shared<Function>(
        make_shared<op::Concat>(
            NodeVector{make_shared<op::GetShape>(A), make_shared<op::GetShape>(B)}, 0),
        op::ParameterVector{A, B});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::StaticValueElimination>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);
    ASSERT_EQ(count_ops_of_type<op::GetShape>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Concat>(f), 0);

    // Make sure result is correct.
    auto backend = runtime::Backend::create("INTERPRETER");

    auto a = backend->create_tensor(element::f32, shape_a);
    vector<float> a_data(2 * 4 * 6 * 8, 42);
    copy_data(a, a_data);
    auto b = backend->create_tensor(element::f32, shape_b);
    vector<float> b_data(3 * 5 * 7, 42);
    copy_data(b, b_data);
    auto result = backend->create_tensor(element::i64, Shape{7});

    backend->call_with_validate(f, {result}, {a, b});
    ASSERT_EQ((vector<int64_t>{2, 4, 6, 8, 3, 5, 7}), read_vector<int64_t>(result));
}

TEST(static_value_elimination, eliminate_multiply)
{
    // Make sure GetShape and Multiply are being eliminated.
    Shape shape_a{2, 4, 6, 8};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    std::vector<uint64_t> k_data{4, 1, 3, 2};
    auto K = make_shared<op::Constant>(element::i64, Shape{4}, k_data.data());

    auto f = make_shared<Function>(make_shared<op::Multiply>(make_shared<op::GetShape>(A), K),
                                   op::ParameterVector{A});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::StaticValueElimination>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);
    ASSERT_EQ(count_ops_of_type<op::GetShape>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Multiply>(f), 0);

    // Make sure result is correct.
    auto backend = runtime::Backend::create("INTERPRETER");

    auto a = backend->create_tensor(element::f32, shape_a);
    vector<float> a_data(2 * 4 * 6 * 8, 42);
    copy_data(a, a_data);
    auto result = backend->create_tensor(element::i64, Shape{4});

    backend->call_with_validate(f, {result}, {a});
    ASSERT_EQ((vector<int64_t>{8, 4, 18, 16}), read_vector<int64_t>(result));
}

TEST(static_value_elimination, eliminate_divide)
{
    // Make sure GetShape and Divide are being eliminated.
    Shape shape_a{2, 4, 6, 8};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    std::vector<uint64_t> k_data{2, 1, 3, 4};
    auto K = make_shared<op::Constant>(element::i64, Shape{4}, k_data.data());

    auto f = make_shared<Function>(make_shared<op::Divide>(make_shared<op::GetShape>(A), K),
                                   op::ParameterVector{A});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::StaticValueElimination>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);
    ASSERT_EQ(count_ops_of_type<op::GetShape>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Divide>(f), 0);

    // Make sure result is correct.
    auto backend = runtime::Backend::create("INTERPRETER");

    auto a = backend->create_tensor(element::f32, shape_a);
    vector<float> a_data(2 * 4 * 6 * 8, 42);
    copy_data(a, a_data);
    auto result = backend->create_tensor(element::i64, Shape{4});

    backend->call_with_validate(f, {result}, {a});
    ASSERT_EQ((vector<int64_t>{1, 4, 2, 2}), read_vector<int64_t>(result));
}

TEST(static_value_elimination, eliminate_broadcast)
{
    // Make sure Constant and Broadcast are being eliminated.
    std::vector<uint64_t> k_data{2};
    auto K = make_shared<op::Constant>(element::i64, Shape{}, k_data.data());
    auto br = make_shared<op::Broadcast>(K, Shape{5}, AxisSet{0});

    auto f = make_shared<Function>(br, op::ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::StaticValueElimination>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);
    ASSERT_EQ(count_ops_of_type<op::Broadcast>(f), 0);

    // Make sure result is correct.
    auto backend = runtime::Backend::create("INTERPRETER");

    auto result = backend->create_tensor(element::i64, Shape{5});

    backend->call_with_validate(f, {result}, {});
    ASSERT_EQ((vector<int64_t>{2, 2, 2, 2, 2}), read_vector<int64_t>(result));
}
