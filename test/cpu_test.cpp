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
#include "misc.hpp"
#include "ngraph/autodiff/adjoints.hpp"
#include "ngraph/file_util.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/batch_norm.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/visualize_tree.hpp"
#include "ngraph/runtime/cpu/op/convert_layout.hpp"
#include "ngraph/serializer.hpp"
#include "ngraph/util.hpp"
#include "nlohmann/json.hpp"
#include "util/all_close.hpp"
#include "util/autodiff/backprop_function.hpp"
#include "util/autodiff/numeric_compare.hpp"
#include "util/ndarray.hpp"
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

static void compare_backends(std::shared_ptr<Function>& f1,
                             std::shared_ptr<Function>& f2,
                             const string backend1,
                             const string backend2,
                             float rtol = 1e-5,
                             float atol = 1e-8)
{
    test::Uniform<float> rng(-1.0f, 1.0f);
    vector<vector<float>> args;
    for (shared_ptr<op::Parameter> param : f1->get_parameters())
    {
        vector<float> tensor_val(shape_size(param->get_shape()));
        rng.initialize(tensor_val);
        args.push_back(tensor_val);
    }
    auto f1_results = execute(f1, args, backend1);
    auto f2_results = execute(f2, args, backend2);

    for (size_t i = 0; i < f1_results.size(); i++)
    {
        EXPECT_TRUE(test::all_close(f1_results.at(i), f2_results.at(i), rtol, atol));
    }
}

TEST(cpu_test, unhandled_op)
{
    auto A = make_shared<op::Parameter>(element::f32, Shape{});
    auto unhandled = make_shared<UnhandledOp>(A);
    auto f = make_shared<Function>(unhandled, ParameterVector{A});
    auto backend = runtime::Backend::create("CPU");
    ASSERT_THROW(backend->compile(f), unsupported_op);
}

TEST(cpu_test, trivial_in_place_relu)
{
    auto A = make_shared<op::Parameter>(element::f32, Shape{16, 1});
    auto B = make_shared<op::Parameter>(element::f32, Shape{16, 1});
    auto add = A + B;
    auto relu = make_shared<op::Relu>(add);
    auto f = make_shared<Function>(relu, ParameterVector{A, B});
    auto backend = runtime::Backend::create("CPU");
    (backend->compile(f));
    ASSERT_EQ(relu->get_outputs().at(0).get_tensor().get_pool_offset(),
              add->get_outputs().at(0).get_tensor().get_pool_offset());
}

#ifndef NGRAPH_HALIDE
TEST(cpu_test, trivial_in_place_relu_fail)
{
    auto A = make_shared<op::Parameter>(element::f32, Shape{16, 1});
    auto B = make_shared<op::Parameter>(element::f32, Shape{16, 1});
    auto add = A + B;
    auto relu = make_shared<op::Relu>(add);
    auto add2 = relu + add;
    auto f = make_shared<Function>(add2, ParameterVector{A, B});
    auto backend = runtime::Backend::create("CPU");
    (backend->compile(f));
    ASSERT_NE(relu->get_outputs().at(0).get_tensor().get_pool_offset(),
              add->get_outputs().at(0).get_tensor().get_pool_offset());
}
#endif

#ifdef NGRAPH_TBB_ENABLE
TEST(cpu_test, abc_tbb)
{
    // Force TBB flow graph generation in the CPU backend
    // This has no effect on other backends
    bool use_tbb = (getenv("NGRAPH_CPU_USE_TBB") != nullptr);
    if (!use_tbb)
    {
        set_environment("NGRAPH_CPU_USE_TBB", "1", 1);
    }

    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto C = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>((A + B) * C, ParameterVector{A, B, C});

    auto backend = runtime::Backend::create("CPU");

    // Create some tensors for input/output
    shared_ptr<runtime::Tensor> a = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> b = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> c = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> result = backend->create_tensor(element::f32, shape);

    copy_data(a, test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
    copy_data(b, test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());
    copy_data(c, test::NDArray<float, 2>({{9, 10}, {11, 12}}).get_vector());

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a, b, c});
    EXPECT_EQ(read_vector<float>(result),
              (test::NDArray<float, 2>({{54, 80}, {110, 144}})).get_vector());

    backend->call_with_validate(handle, {result}, {b, a, c});
    EXPECT_EQ(read_vector<float>(result),
              (test::NDArray<float, 2>({{54, 80}, {110, 144}})).get_vector());

    backend->call_with_validate(handle, {result}, {a, c, b});
    EXPECT_EQ(read_vector<float>(result),
              (test::NDArray<float, 2>({{50, 72}, {98, 128}})).get_vector());

    if (!use_tbb)
    {
        unset_environment("NGRAPH_CPU_USE_TBB");
    }
}
#endif // NGRAPH_TBB_ENABLE

TEST(cpu_test, mkldnn_layouts)
{
    Shape shape_a{1, 16, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{32, 16, 1, 1};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{1, 32, 2, 2};
    auto conv1 = make_shared<op::Convolution>(A,
                                              B,
                                              Strides{1, 1},
                                              Strides{1, 1},
                                              CoordinateDiff{0, 0},
                                              CoordinateDiff{0, 0},
                                              Strides{1, 1});
    Shape pool_shape{1, 1};
    auto pool1 = make_shared<op::AvgPool>(conv1, pool_shape);
    auto pool1_result = make_shared<op::Result>(pool1);
    // Request result in default layout
    pool1_result->set_needs_default_layout(true);
    auto f = make_shared<Function>(ResultVector{pool1_result}, ParameterVector{A, B});

    auto backend = runtime::Backend::create("CPU");

    vector<float> input(64, 1.0f);
    vector<float> weights;
    vector<float> rv(128);
    for (int i = 0; i < 128; i++)
    {
        weights.push_back(0.0f);
    }
    for (int i = 0; i < 384; i++)
    {
        weights.push_back(1.0f);
    }

    auto a = backend->create_tensor(element::f32, shape_a, input.data());
    auto b = backend->create_tensor(element::f32, shape_b, weights.data());
    auto result = backend->create_tensor(element::f32, shape_r, rv.data());

    vector<float> expected_result;
    for (int i = 0; i < 32; i++)
    {
        expected_result.push_back(0.0f);
    }
    for (int i = 0; i < 96; i++)
    {
        expected_result.push_back(16.0f);
    }

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a, b});

    EXPECT_EQ(vector<float>{expected_result}, rv);
}

TEST(cpu_test, reshape_layout_optimizations1)
{
    // Squeeze outermost dimension
    auto make_function = []() -> std::shared_ptr<Function> {
        auto A = make_shared<op::Parameter>(element::f32, Shape{1, 16, 2, 2});
        auto B = make_shared<op::Parameter>(element::f32, Shape{32, 16, 1, 1});
        auto conv = make_shared<op::Convolution>(A,
                                                 B,
                                                 Strides{1, 1},
                                                 Strides{1, 1},
                                                 CoordinateDiff{0, 0},
                                                 CoordinateDiff{0, 0},
                                                 Strides{1, 1});
        auto squeeze = make_shared<op::Reshape>(conv, AxisVector{0, 1, 2, 3}, Shape{32, 2, 2});
        return make_shared<Function>(NodeVector{squeeze}, ParameterVector{A, B});
    };

    auto backend = runtime::Backend::create("CPU");
    auto cpu_f = make_function();
    auto int_f = make_function();

    test::Uniform<float> rng(-100.0f, 100.0f);
    vector<vector<float>> args;
    for (shared_ptr<op::Parameter> param : cpu_f->get_parameters())
    {
        vector<float> tensor_val(shape_size(param->get_shape()));
        rng.initialize(tensor_val);
        args.push_back(tensor_val);
    }
    auto int_results = execute(int_f, args, "INTERPRETER");
    auto cpu_results = execute(cpu_f, args, "CPU");
    // Two convert layouts for inputs and weights of convolution.
    EXPECT_EQ(count_ops_of_type<runtime::cpu::op::ConvertLayout>(cpu_f), 2);
    for (size_t i = 0; i < cpu_results.size(); i++)
    {
        EXPECT_TRUE(test::all_close(cpu_results.at(i), int_results.at(i), 1.0e-4f, 1.0e-4f));
    }
}

TEST(cpu_test, reshape_layout_optimizations2)
{
    // ExpandDims - inner most and internal dims
    auto make_function = []() -> std::shared_ptr<Function> {
        auto A = make_shared<op::Parameter>(element::f32, Shape{1, 16, 2, 2});
        auto B = make_shared<op::Parameter>(element::f32, Shape{32, 16, 1, 1});
        auto conv = make_shared<op::Convolution>(A,
                                                 B,
                                                 Strides{1, 1},
                                                 Strides{1, 1},
                                                 CoordinateDiff{0, 0},
                                                 CoordinateDiff{0, 0},
                                                 Strides{1, 1});
        auto expand =
            make_shared<op::Reshape>(conv, AxisVector{0, 1, 2, 3}, Shape{1, 32, 2, 1, 2, 1});
        return make_shared<Function>(NodeVector{expand}, ParameterVector{A, B});
    };

    auto backend = runtime::Backend::create("CPU");
    auto cpu_f = make_function();
    auto int_f = make_function();

    test::Uniform<float> rng(-100.0f, 100.0f);
    vector<vector<float>> args;
    for (shared_ptr<op::Parameter> param : cpu_f->get_parameters())
    {
        vector<float> tensor_val(shape_size(param->get_shape()));
        rng.initialize(tensor_val);
        args.push_back(tensor_val);
    }
    auto int_results = execute(int_f, args, "INTERPRETER");
    auto cpu_results = execute(cpu_f, args, "CPU");
    EXPECT_EQ(count_ops_of_type<runtime::cpu::op::ConvertLayout>(cpu_f), 2);
    for (size_t i = 0; i < cpu_results.size(); i++)
    {
        EXPECT_TRUE(test::all_close(cpu_results.at(i), int_results.at(i), 1.0e-4f, 1.0e-4f));
    }
}

TEST(cpu_test, reshape_layout_optimizations3)
{
    // Squeeze padded dimension
    auto make_function = []() -> std::shared_ptr<Function> {
        auto A = make_shared<op::Parameter>(element::f32, Shape{1, 16, 2, 2});
        auto B = make_shared<op::Parameter>(element::f32, Shape{1, 16, 1, 1});
        auto conv = make_shared<op::Convolution>(A,
                                                 B,
                                                 Strides{1, 1},
                                                 Strides{1, 1},
                                                 CoordinateDiff{0, 0},
                                                 CoordinateDiff{0, 0},
                                                 Strides{1, 1});
        auto squeeze = make_shared<op::Reshape>(conv, AxisVector{0, 1, 2, 3}, Shape{2, 2});
        return make_shared<Function>(NodeVector{squeeze}, ParameterVector{A, B});
    };

    auto backend = runtime::Backend::create("CPU");
    auto cpu_f = make_function();
    auto int_f = make_function();

    test::Uniform<float> rng(-100.0f, 100.0f);
    vector<vector<float>> args;
    for (shared_ptr<op::Parameter> param : cpu_f->get_parameters())
    {
        vector<float> tensor_val(shape_size(param->get_shape()));
        rng.initialize(tensor_val);
        args.push_back(tensor_val);
    }
    auto int_results = execute(int_f, args, "INTERPRETER");
    auto cpu_results = execute(cpu_f, args, "CPU");
    // Two convert layouts for inputs and weights of convolution.
    // One convert layout after convolution
    EXPECT_EQ(count_ops_of_type<runtime::cpu::op::ConvertLayout>(cpu_f), 3);
    for (size_t i = 0; i < cpu_results.size(); i++)
    {
        EXPECT_TRUE(test::all_close(cpu_results.at(i), int_results.at(i), 1.0e-4f, 1.0e-4f));
    }
}

TEST(cpu_test, reshape_layout_optimizations4)
{
    // Squeeze and expand dimensions. Ensure no extra conversions downstream
    auto make_function = []() -> std::shared_ptr<Function> {
        auto A = make_shared<op::Parameter>(element::f32, Shape{1, 16, 1, 8});
        auto B1 = make_shared<op::Parameter>(element::f32, Shape{32, 16, 1, 1});
        auto conv1 = make_shared<op::Convolution>(A,
                                                  B1,
                                                  Strides{1, 1},
                                                  Strides{1, 1},
                                                  CoordinateDiff{0, 0},
                                                  CoordinateDiff{0, 0},
                                                  Strides{1, 1});
        auto squeeze = make_shared<op::Reshape>(conv1, AxisVector{0, 1, 2, 3}, Shape{32, 1, 8});
        auto relu = make_shared<op::Relu>(squeeze);
        auto expand = make_shared<op::Reshape>(relu, AxisVector{0, 1, 2}, Shape{1, 32, 1, 8});
        auto B2 = make_shared<op::Parameter>(element::f32, Shape{8, 32, 1, 1});
        auto conv2 = make_shared<op::Convolution>(expand,
                                                  B2,
                                                  Strides{1, 1},
                                                  Strides{1, 1},
                                                  CoordinateDiff{0, 0},
                                                  CoordinateDiff{0, 0},
                                                  Strides{1, 1});
        return make_shared<Function>(NodeVector{conv2}, ParameterVector{A, B1, B2});
    };

    auto backend = runtime::Backend::create("CPU");
    auto cpu_f = make_function();
    auto int_f = make_function();

    test::Uniform<float> rng(-100.0f, 100.0f);
    vector<vector<float>> args;
    for (shared_ptr<op::Parameter> param : cpu_f->get_parameters())
    {
        vector<float> tensor_val(shape_size(param->get_shape()));
        rng.initialize(tensor_val);
        args.push_back(tensor_val);
    }
    auto int_results = execute(int_f, args, "INTERPRETER");
    auto cpu_results = execute(cpu_f, args, "CPU");

    for (size_t i = 0; i < cpu_results.size(); i++)
    {
        EXPECT_TRUE(test::all_close(cpu_results.at(i), int_results.at(i), 1.0e-4f, 1.0e-4f));
    }
    EXPECT_LE(count_ops_of_type<runtime::cpu::op::ConvertLayout>(cpu_f), 4);
}

TEST(cpu_test, reshape_layout_optimizations5)
{
    auto make_function = []() -> std::shared_ptr<Function> {
        auto A = make_shared<op::Parameter>(element::f32, Shape{1, 16, 1, 8});
        auto B1 = make_shared<op::Parameter>(element::f32, Shape{32, 16, 1, 1});
        auto conv1 = make_shared<op::Convolution>(A,
                                                  B1,
                                                  Strides{1, 1},
                                                  Strides{1, 1},
                                                  CoordinateDiff{0, 0},
                                                  CoordinateDiff{0, 0},
                                                  Strides{1, 1});
        auto expand =
            make_shared<op::Reshape>(conv1, AxisVector{0, 1, 2, 3}, Shape{1, 1, 32, 1, 8});
        auto relu = make_shared<op::Relu>(expand);
        auto squeeze =
            make_shared<op::Reshape>(relu, AxisVector{0, 1, 2, 3, 4}, Shape{1, 32, 1, 8});
        auto B2 = make_shared<op::Parameter>(element::f32, Shape{8, 32, 1, 1});
        auto conv2 = make_shared<op::Convolution>(squeeze,
                                                  B2,
                                                  Strides{1, 1},
                                                  Strides{1, 1},
                                                  CoordinateDiff{0, 0},
                                                  CoordinateDiff{0, 0},
                                                  Strides{1, 1});
        return make_shared<Function>(NodeVector{conv2}, ParameterVector{A, B1, B2});
    };

    auto backend = runtime::Backend::create("CPU");
    auto cpu_f = make_function();
    auto int_f = make_function();

    test::Uniform<float> rng(-100.0f, 100.0f);
    vector<vector<float>> args;
    for (shared_ptr<op::Parameter> param : cpu_f->get_parameters())
    {
        vector<float> tensor_val(shape_size(param->get_shape()));
        rng.initialize(tensor_val);
        args.push_back(tensor_val);
    }
    auto int_results = execute(int_f, args, "INTERPRETER");
    auto cpu_results = execute(cpu_f, args, "CPU");

    for (size_t i = 0; i < cpu_results.size(); i++)
    {
        EXPECT_TRUE(test::all_close(cpu_results.at(i), int_results.at(i), 1.0e-4f, 1.0e-4f));
    }
    EXPECT_LE(count_ops_of_type<runtime::cpu::op::ConvertLayout>(cpu_f), 4);
}

TEST(cpu_test, reshape_layout_optimizations6)
{
    // Squeeze and expand dimensions. Ensure no extra conversions downstream
    auto make_function = []() -> std::shared_ptr<Function> {
        auto A = make_shared<op::Parameter>(element::f32, Shape{2, 4, 3, 2});
        auto mul = make_shared<op::Multiply>(A, A);
        auto sum = make_shared<op::Sum>(mul, AxisVector{0});
        auto reshape = make_shared<op::Reshape>(sum, AxisVector{0, 1, 2}, Shape{1, 4, 3, 2});
        auto sqrt = make_shared<op::Sqrt>(reshape);
        return make_shared<Function>(NodeVector{sqrt}, ParameterVector{A});
    };

    auto backend = runtime::Backend::create("CPU");
    auto cpu_f = make_function();
    auto int_f = make_function();

    test::Uniform<float> rng(-100.0f, 100.0f);
    vector<vector<float>> args;
    for (shared_ptr<op::Parameter> param : cpu_f->get_parameters())
    {
        vector<float> tensor_val(shape_size(param->get_shape()));
        rng.initialize(tensor_val);
        args.push_back(tensor_val);
    }
    auto int_results = execute(int_f, args, "INTERPRETER");
    auto cpu_results = execute(cpu_f, args, "CPU");

    for (size_t i = 0; i < cpu_results.size(); i++)
    {
        EXPECT_TRUE(test::all_close(cpu_results.at(i), int_results.at(i)));
    }
    EXPECT_EQ(count_ops_of_type<runtime::cpu::op::ConvertLayout>(cpu_f), 0);
}

TEST(cpu_test, reshape_layout_optimizations7)
{
    // Expand multiple dimensions. Ensure no extra conversions downstream
    auto make_function = []() -> std::shared_ptr<Function> {
        auto A = make_shared<op::Parameter>(element::f32, Shape{1, 4, 10, 6, 10});
        auto mul = make_shared<op::Multiply>(A, A);
        auto sum = make_shared<op::Sum>(mul, AxisVector{0, 1});
        auto reshape = make_shared<op::Reshape>(sum, AxisVector{0, 1, 2}, Shape{1, 1, 10, 6, 10});
        return make_shared<Function>(NodeVector{reshape}, ParameterVector{A});
    };

    auto backend = runtime::Backend::create("CPU");
    auto cpu_f = make_function();
    auto int_f = make_function();

    test::Uniform<float> rng(-100.0f, 100.0f);
    vector<vector<float>> args;
    for (shared_ptr<op::Parameter> param : cpu_f->get_parameters())
    {
        vector<float> tensor_val(shape_size(param->get_shape()));
        rng.initialize(tensor_val);
        args.push_back(tensor_val);
    }
    auto int_results = execute(int_f, args, "INTERPRETER");
    auto cpu_results = execute(cpu_f, args, "CPU");

    for (size_t i = 0; i < cpu_results.size(); i++)
    {
        EXPECT_TRUE(test::all_close(cpu_results.at(i), int_results.at(i)));
    }
    EXPECT_EQ(count_ops_of_type<runtime::cpu::op::ConvertLayout>(cpu_f), 0);
}

TEST(cpu_test, DISABLED_collapse_dims1)
{
    // Expand multiple dimensions. Ensure no extra conversions downstream
    auto make_function = []() -> std::shared_ptr<Function> {
        auto A = make_shared<op::Parameter>(element::f32, Shape{1, 4, 10, 6, 10});
        auto sum1 = make_shared<op::Sum>(A, AxisVector{1});    // Shape{1, 10, 6, 10}
        auto sum2 = make_shared<op::Sum>(sum1, AxisVector{0}); // Shape{10, 6, 10}
        return make_shared<Function>(NodeVector{sum2}, ParameterVector{A});
    };

    auto backend = runtime::Backend::create("CPU");
    auto cpu_f = make_function();
    auto int_f = make_function();

    test::Uniform<float> rng(-100.0f, 100.0f);
    vector<vector<float>> args;
    for (shared_ptr<op::Parameter> param : cpu_f->get_parameters())
    {
        vector<float> tensor_val(shape_size(param->get_shape()));
        rng.initialize(tensor_val);
        args.push_back(tensor_val);
    }
    auto int_results = execute(int_f, args, "INTERPRETER");
    auto cpu_results = execute(cpu_f, args, "CPU");

    for (size_t i = 0; i < cpu_results.size(); i++)
    {
        EXPECT_TRUE(test::all_close(cpu_results.at(i), int_results.at(i)));
    }
    // sum1 will have two reshapes added around it. sum2 will be replaced
    // with a reshape
    EXPECT_EQ(count_ops_of_type<op::Reshape>(cpu_f), 3);
}

TEST(cpu_test, collapse_dims2)
{
    // Collapse dims around a dot where one of the inputs is a scalar
    auto make_function = []() -> std::shared_ptr<Function> {
        auto A = make_shared<op::Parameter>(element::f32, Shape{1, 3, 1, 1});
        auto B = make_shared<op::Parameter>(element::f32, Shape{1, 1});
        auto dot = make_shared<op::Dot>(A, B, 1);
        return make_shared<Function>(NodeVector{dot}, ParameterVector{A, B});
    };

    auto backend = runtime::Backend::create("CPU");
    auto cpu_f = make_function();
    auto int_f = make_function();

    test::Uniform<float> rng(-100.0f, 100.0f);
    vector<vector<float>> args;
    for (shared_ptr<op::Parameter> param : cpu_f->get_parameters())
    {
        vector<float> tensor_val(shape_size(param->get_shape()));
        rng.initialize(tensor_val);
        args.push_back(tensor_val);
    }
    auto int_results = execute(int_f, args, "INTERPRETER");
    auto cpu_results = execute(cpu_f, args, "CPU");

    for (size_t i = 0; i < cpu_results.size(); i++)
    {
        EXPECT_TRUE(test::all_close(cpu_results.at(i), int_results.at(i)));
    }
}

TEST(cpu_test, convert_layout)
{
    auto make_function = []() -> std::shared_ptr<Function> {
        auto W = std::make_shared<op::Parameter>(element::f32, Shape{10, 400});
        auto X = std::make_shared<op::Parameter>(element::f32, Shape{400, 10});
        auto W_reshape = std::make_shared<op::Reshape>(W, AxisVector{1, 0}, Shape{400, 10});

        auto add1 = std::make_shared<op::Add>(X, W_reshape);
        auto sub1 = std::make_shared<op::Subtract>(X, W_reshape);
        auto mul1 = std::make_shared<op::Multiply>(X, W_reshape);

        return make_shared<Function>(NodeVector{add1, sub1, mul1}, ParameterVector{W, X});
    };
    auto backend = runtime::Backend::create("CPU");
    auto cpu_f = make_function();
    auto int_f = make_function();

    test::Uniform<float> rng(-100.0f, 100.0f);
    vector<vector<float>> args;
    for (shared_ptr<op::Parameter> param : cpu_f->get_parameters())
    {
        vector<float> tensor_val(shape_size(param->get_shape()));
        rng.initialize(tensor_val);
        args.push_back(tensor_val);
    }
    auto int_results = execute(int_f, args, "INTERPRETER");
    auto cpu_results = execute(cpu_f, args, "CPU");

    size_t count = count_ops_of_type<runtime::cpu::op::ConvertLayout>(cpu_f);
    ASSERT_EQ(count, 1);
    for (size_t i = 0; i < cpu_results.size(); i++)
    {
        EXPECT_TRUE(test::all_close(cpu_results.at(i), int_results.at(i)));
    }
}

TEST(cpu_test, post_layout_reshape_convertlayout)
{
    auto make_function = []() -> std::shared_ptr<Function> {
        auto A = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3, 4});
        auto B = make_shared<op::Parameter>(element::f32, Shape{5, 2, 1, 1});
        auto conv = make_shared<op::Convolution>(A,
                                                 B,
                                                 Strides{1, 1},
                                                 Strides{1, 1},
                                                 CoordinateDiff{0, 0},
                                                 CoordinateDiff{0, 0},
                                                 Strides{1, 1});
        auto reshape = make_shared<op::Reshape>(conv, AxisVector{0, 2, 3, 1}, Shape{1, 3, 4, 5});
        return make_shared<Function>(NodeVector{reshape}, ParameterVector{A, B});
    };

    auto int_f = make_function();
    auto cpu_f = make_function();
    compare_backends(int_f, cpu_f, "INTERPRETER", "CPU");
}

TEST(cpu_test, mkldnn_layouts_eltwise)
{
    Shape input_shape{3, 11, 14, 14};
    Shape filter_shape{5, 11, 2, 2};

    auto make_function = [&]() {
        auto input = std::make_shared<op::Parameter>(element::f32, input_shape);
        auto filter = std::make_shared<op::Parameter>(element::f32, filter_shape);
        auto conv = std::make_shared<op::Convolution>(input, filter, Strides{2, 2}, Strides{1, 1});
        auto sigmoid = std::make_shared<op::Sigmoid>(conv);
        auto f = make_shared<Function>(NodeVector{sigmoid}, ParameterVector{input, filter});
        return f;
    };

    auto int_f = make_function();
    auto cpu_f = make_function();
    compare_backends(int_f, cpu_f, "INTERPRETER", "CPU");
}

TEST(cpu_test, convolution_large_padding)
{
    Shape input_shape{1, 1, 100, 100};
    Shape filter_shape{1, 1, 3, 3};

    auto make_function = [&]() {
        auto input = std::make_shared<op::Parameter>(element::f32, input_shape);
        auto filter = std::make_shared<op::Parameter>(element::f32, filter_shape);
        auto conv = std::make_shared<op::Convolution>(input,
                                                      filter,
                                                      Strides{1, 1},
                                                      Strides{9, 9},
                                                      CoordinateDiff{9, 9},
                                                      CoordinateDiff{9, 9});
        auto f = make_shared<Function>(NodeVector{conv}, ParameterVector{input, filter});
        return f;
    };

    auto int_f = make_function();
    auto cpu_f = make_function();
    compare_backends(int_f, cpu_f, "INTERPRETER", "CPU", 1e-4, 1e-4);
}
