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

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <list>
#include <memory>

#include "gtest/gtest.h"
#include "ngraph/autodiff/adjoints.hpp"
#include "ngraph/file_util.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/batch_norm.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/op/reverse_sequence.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/visualize_tree.hpp"
#include "ngraph/runtime/cpu/pass/cpu_fusion.hpp"
#include "ngraph/serializer.hpp"
#include "ngraph/util.hpp"
#include "nlohmann/json.hpp"
#include "util/all_close.hpp"
#include "util/autodiff/backprop_function.hpp"
#include "util/autodiff/numeric_compare.hpp"
#include "util/random.hpp"
#include "util/test_tools.hpp"

using namespace ngraph;
using namespace std;

class UnhandledOp : public ngraph::op::Abs
{
public:
    UnhandledOp(const std::shared_ptr<Node>& arg)
        : Abs(arg)
    {
    }
};

TEST(cpu_test, unhandled_op)
{
    auto A = make_shared<op::Parameter>(element::f32, Shape{});
    auto unhandled = make_shared<UnhandledOp>(A);
    auto f = make_shared<Function>(unhandled, op::ParameterVector{A});
    auto backend = runtime::Backend::create("CPU");
    ASSERT_THROW(backend->compile(f), ngraph_error);
}

TEST(cpu_test, reverse_sequence_n2c3h4w2)
{
    Shape shape{2, 3, 4, 2};
    Shape seq_len_shape{4};
    auto A = make_shared<op::Parameter>(element::i32, shape);
    auto B = make_shared<op::Parameter>(element::i32, seq_len_shape);

    size_t batch_axis = 2;
    size_t sequence_axis = 1;
    auto rs = std::make_shared<op::ReverseSequence>(A, B, batch_axis, sequence_axis);

    auto f = make_shared<Function>(rs, op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("CPU");

    // Create some tensors for input/output
    shared_ptr<runtime::TensorView> a = backend->create_tensor(element::i32, shape);
    shared_ptr<runtime::TensorView> b = backend->create_tensor(element::i32, seq_len_shape);

    shared_ptr<runtime::TensorView> result = backend->create_tensor(element::i32, shape);

    std::vector<int> input{
        0,  0, 3,  0, 6,  0, 9,  0, 1,  0, 4,  0, 7,  0, 10, 0, 2,  0, 5,  0, 8,  0, 11, 0,
        12, 0, 15, 0, 18, 0, 21, 0, 13, 0, 16, 0, 19, 0, 22, 0, 14, 0, 17, 0, 20, 0, 23, 0,
    };

    std::vector<int> seq_lenghts{1, 2, 1, 2};
    copy_data(b, seq_lenghts);

    std::vector<int> expected{
        0,  0, 4,  0, 6,  0, 10, 0, 1,  0, 3,  0, 7,  0, 9,  0, 2,  0, 5,  0, 8,  0, 11, 0,

        12, 0, 16, 0, 18, 0, 22, 0, 13, 0, 15, 0, 19, 0, 21, 0, 14, 0, 17, 0, 20, 0, 23, 0};

    copy_data(a, input);

    backend->call(f, {result}, {a, b});
    EXPECT_EQ(read_vector<int>(result), expected);
}

TEST(cpu_test, reverse_sequence_n4c3h2w2)
{
    Shape shape{4, 3, 2, 2};
    auto A = make_shared<op::Parameter>(element::i32, shape);
    Shape seq_len_shape{4};
    auto B = make_shared<op::Parameter>(element::i32, seq_len_shape);

    size_t batch_axis = 0;
    size_t sequence_axis = 1;

    auto rs = std::make_shared<op::ReverseSequence>(A, B, batch_axis, sequence_axis);

    auto f = make_shared<Function>(rs, op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("CPU");

    // Create some tensors for input/output
    shared_ptr<runtime::TensorView> a = backend->create_tensor(element::i32, shape);
    shared_ptr<runtime::TensorView> b = backend->create_tensor(element::i32, seq_len_shape);

    shared_ptr<runtime::TensorView> result = backend->create_tensor(element::i32, shape);

    std::vector<int> seq_lenghts{1, 2, 3, 3};
    copy_data(b, seq_lenghts);

    std::vector<int> input{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                           16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                           32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47};

    std::vector<int> expected{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 16, 17, 18, 19,
                              12, 13, 14, 15, 20, 21, 22, 23, 32, 33, 34, 35, 28, 29, 30, 31,
                              24, 25, 26, 27, 44, 45, 46, 47, 40, 41, 42, 43, 36, 37, 38, 39};

    copy_data(a, input);

    backend->call(f, {result}, {a, b});
    EXPECT_EQ(read_vector<int>(result), expected);
}

TEST(cpu_test, reverse_sequence_n4d2c3h2w2)
{
    Shape shape{4, 2, 3, 2, 2};
    auto A = make_shared<op::Parameter>(element::i32, shape);
    Shape seq_len_shape{4};
    auto B = make_shared<op::Parameter>(element::i32, seq_len_shape);

    size_t batch_axis = 0;
    size_t sequence_axis = 2;

    auto rs = std::make_shared<op::ReverseSequence>(A, B, batch_axis, sequence_axis);

    auto f = make_shared<Function>(rs, op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("CPU");

    // Create some tensors for input/output
    shared_ptr<runtime::TensorView> a = backend->create_tensor(element::i32, shape);
    shared_ptr<runtime::TensorView> b = backend->create_tensor(element::i32, seq_len_shape);

    shared_ptr<runtime::TensorView> result = backend->create_tensor(element::i32, shape);

    std::vector<int> input{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                           16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                           32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
                           48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
                           64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
                           80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95};

    std::vector<int> expected{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                              16, 17, 18, 19, 20, 21, 22, 23, 28, 29, 30, 31, 24, 25, 26, 27,
                              32, 33, 34, 35, 40, 41, 42, 43, 36, 37, 38, 39, 44, 45, 46, 47,
                              48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
                              64, 65, 66, 67, 68, 69, 70, 71, 76, 77, 78, 79, 72, 73, 74, 75,
                              80, 81, 82, 83, 88, 89, 90, 91, 84, 85, 86, 87, 92, 93, 94, 95};

    copy_data(a, input);

    std::vector<int> seq_lenghts{1, 2, 1, 2};
    copy_data(b, seq_lenghts);

    backend->call(f, {result}, {a, b});
    EXPECT_EQ(read_vector<int>(result), expected);
}

TEST(cpu_test, backwards_reverse_sequence_n3_c2_h3)
{
    Shape shape{3, 2, 3};
    auto backend = runtime::Backend::create("CPU");

    auto A = make_shared<op::Parameter>(element::i32, shape);
    Shape seq_len_shape{2};
    auto B = make_shared<op::Parameter>(element::i32, seq_len_shape);

    size_t batch_axis = 1;
    size_t sequence_axis = 0;

    auto rs = std::make_shared<op::ReverseSequence>(A, B, batch_axis, sequence_axis);
    auto f = make_shared<Function>(rs, op::ParameterVector{A, B});

    shared_ptr<runtime::TensorView> a = backend->create_tensor(element::i32, shape);
    shared_ptr<runtime::TensorView> b = backend->create_tensor(element::i32, seq_len_shape);
    shared_ptr<runtime::TensorView> c = backend->create_tensor(element::i32, shape);
    shared_ptr<runtime::TensorView> da = backend->create_tensor(element::i32, shape);
    shared_ptr<runtime::TensorView> db = backend->create_tensor(element::i32, seq_len_shape);

    //input values don't matter
    vector<int> va(shape_size(shape), 0);

    vector<int> vc{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};
    vector<int> expected{13, 14, 15, 16, 17, 18, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6};

    copy_data(c, vc);
    copy_data(a, va);

    std::vector<int> seq_lenghts{3, 3};
    copy_data(b, seq_lenghts);

    auto C = make_shared<op::Parameter>(element::i32, shape);
    auto df = autodiff::backprop_function(f);
    backend->call(df, {da, db}, {a, b, c});
    ASSERT_EQ(read_vector<int>(da), expected);
}

TEST(cpu_test, backwards_reverse_sequence_n4d2c3h2w2)
{
    Shape shape{4, 2, 3, 2, 2};
    auto backend = runtime::Backend::create("CPU");

    auto A = make_shared<op::Parameter>(element::i32, shape);
    Shape seq_len_shape{4};
    auto B = make_shared<op::Parameter>(element::i32, seq_len_shape);

    size_t batch_axis = 0;
    size_t sequence_axis = 2;

    auto rs = std::make_shared<op::ReverseSequence>(A, B, batch_axis, sequence_axis);
    auto f = make_shared<Function>(rs, op::ParameterVector{A, B});

    shared_ptr<runtime::TensorView> a = backend->create_tensor(element::i32, shape);
    shared_ptr<runtime::TensorView> b = backend->create_tensor(element::i32, seq_len_shape);
    shared_ptr<runtime::TensorView> c = backend->create_tensor(element::i32, shape);
    shared_ptr<runtime::TensorView> da = backend->create_tensor(element::i32, shape);
    shared_ptr<runtime::TensorView> db = backend->create_tensor(element::i32, seq_len_shape);

    //input values don't matter
    vector<int> va(shape_size(shape), 0);

    std::vector<int> vc{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                        32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
                        48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
                        64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
                        80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95};

    std::vector<int> expected{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                              16, 17, 18, 19, 20, 21, 22, 23, 28, 29, 30, 31, 24, 25, 26, 27,
                              32, 33, 34, 35, 40, 41, 42, 43, 36, 37, 38, 39, 44, 45, 46, 47,
                              48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
                              64, 65, 66, 67, 68, 69, 70, 71, 76, 77, 78, 79, 72, 73, 74, 75,
                              80, 81, 82, 83, 88, 89, 90, 91, 84, 85, 86, 87, 92, 93, 94, 95};

    copy_data(c, vc);
    copy_data(a, va);

    std::vector<int> seq_lenghts{1, 2, 1, 2};
    copy_data(b, seq_lenghts);

    auto C = make_shared<op::Parameter>(element::i32, shape);
    auto df = autodiff::backprop_function(f);
    backend->call(df, {da, db}, {a, b, c});
    ASSERT_EQ(read_vector<int>(da), expected);
}
