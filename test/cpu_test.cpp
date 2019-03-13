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
#include "ngraph/runtime/cpu/cpu_backend.hpp"
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

static void compare_backends(const std::shared_ptr<Function>& f1,
                             const std::shared_ptr<Function>& f2,
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
    handle->call_with_validate({result}, {a, b, c});
    EXPECT_EQ(read_vector<float>(result),
              (test::NDArray<float, 2>({{54, 80}, {110, 144}})).get_vector());

    handle->call_with_validate({result}, {b, a, c});
    EXPECT_EQ(read_vector<float>(result),
              (test::NDArray<float, 2>({{54, 80}, {110, 144}})).get_vector());

    handle->call_with_validate({result}, {a, c, b});
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
    handle->call_with_validate({result}, {a, b});

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

#if 0
static std::shared_ptr<Function> make_function(const std::string& file_name)
{
    const string json_path = file_util::path_join(SERIALIZED_ZOO, file_name);
    const string json_string = file_util::read_file_to_string(json_path);
    stringstream ss(json_string);
    shared_ptr<Function> func = ngraph::deserialize(ss);
    return func;
}

TEST(cpu_test, memory_reuse_mxnet_densenet121)
{
    const std::string file_name("mxnet/mxnet_densenet121_inference_batch1_float32.json");
    auto cpu_f = make_function(file_name);

    test::Uniform<float> rng(-1.0f, 1.0f);
    vector<vector<float>> args;

    for (shared_ptr<op::Parameter> param : cpu_f->get_parameters())
    {
        vector<float> tensor_val(shape_size(param->get_shape()));
        rng.initialize(tensor_val);
        args.push_back(tensor_val);
    }

    // without memory reuse
    auto cpu_results = execute(cpu_f, args, "CPU");

    auto cpu_f_new = make_function(file_name);
    auto cpu_results_new = execute(cpu_f_new, args, "CPU");
    for (size_t i = 0; i < cpu_results.size(); i++)
    {
        EXPECT_TRUE(test::all_close(cpu_results.at(i), cpu_results_new.at(i), 1.0e-4f, 1.0e-4f));
    }

    // with memory reuse
    auto backend = runtime::Backend::create("CPU");
    auto parms = cpu_f->get_parameters();
    std::vector<std::shared_ptr<ngraph::runtime::Tensor>> arg_tensors(args.size());
    for (size_t i = 0; i < args.size(); i++)
    {
        auto t = backend->create_tensor(parms.at(i)->get_element_type(), parms.at(i)->get_shape());
        copy_data(t, args.at(i));
        arg_tensors.at(i) = t;
    }

    auto results = cpu_f->get_results();
    std::vector<std::shared_ptr<ngraph::runtime::Tensor>> result_tensors(results.size());

    for (size_t i = 0; i < results.size(); i++)
    {
        result_tensors.at(i) =
            backend->create_tensor(results.at(i)->get_element_type(), results.at(i)->get_shape());
    }

    ngraph::pass::PassConfig pass_config;
    pass_config.set_pass_attribute("CPUMemoryAssignment::ReuseMemory", true);

    auto cpu_f_new_reuse = make_function(file_name);

    shared_ptr<runtime::Executable> handle = backend->compile(cpu_f_new_reuse, pass_config);
    for (auto it = 0; it < 2; it++)
    {
        handle->call_with_validate(result_tensors, arg_tensors);

        std::vector<std::vector<float>> cpu_results_new_reuse;
        for (auto rt : result_tensors)
        {
            cpu_results_new_reuse.push_back(read_vector<float>(rt));
        }

        for (size_t i = 0; i < cpu_results.size(); i++)
        {
            EXPECT_TRUE(
                test::all_close(cpu_results.at(i), cpu_results_new_reuse.at(i), 1.0e-4f, 1.0e-4f));
        }
    }
}
#endif

TEST(cpu_test, memory_reuse_destructive_oi_relu)
{
    auto shape_a = Shape{2, 5};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto B = make_shared<op::Parameter>(element::f32, shape_a);
    auto C = make_shared<op::Parameter>(element::f32, shape_a);
    auto add = make_shared<op::Add>(A, B);
    auto relu = make_shared<op::Relu>(add);
    auto subtract = make_shared<op::Subtract>(C, relu);
    auto shape_rt = Shape{2, 5};
    auto f = make_shared<Function>(subtract, ParameterVector{A, B, C});

    auto backend = runtime::Backend::create("CPU");

    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 8, -8, 17, -0.5, 1, 8, -8, 17, -0.5});
    auto b = backend->create_tensor(element::f32, shape_a);
    copy_data(b, vector<float>{1, 2, 3, 4, 0.5, 1, 8, -8, 17, -0.5});
    auto c = backend->create_tensor(element::f32, shape_a);
    copy_data(c, vector<float>{2, 10, 0, 21, 0, 2, 16, 0, 34, 0});
    auto result = backend->create_tensor(element::f32, shape_rt);
    vector<float> expected{0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    shared_ptr<runtime::Executable> handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b, c});
    ASSERT_NE(handle, nullptr);
    EXPECT_EQ(read_vector<float>(result), expected);
}

TEST(cpu_test, memory_reuse_cacheable_no_destructive_oi_relu)
{
    auto shape_a = Shape{2, 5};
    auto A = make_shared<op::Parameter>(element::f32, shape_a, true);
    auto B = make_shared<op::Parameter>(element::f32, shape_a, true);
    auto C = make_shared<op::Parameter>(element::f32, shape_a);
    auto add = make_shared<op::Add>(A, B);
    auto relu = make_shared<op::Relu>(add);
    auto subtract = make_shared<op::Subtract>(C, relu);
    auto shape_rt = Shape{2, 5};
    auto f = make_shared<Function>(subtract, ParameterVector{A, B, C});

    auto backend = runtime::Backend::create("CPU");

    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 8, -8, 17, -0.5, 1, 8, -8, 17, -0.5});
    auto b = backend->create_tensor(element::f32, shape_a);
    copy_data(b, vector<float>{1, 2, 3, 4, 0.5, 1, 8, -8, 17, -0.5});
    auto c = backend->create_tensor(element::f32, shape_a);
    copy_data(c, vector<float>{2, 10, 0, 21, 0, 2, 16, 0, 34, 0});
    auto result = backend->create_tensor(element::f32, shape_rt);
    vector<float> expected{0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    shared_ptr<runtime::Executable> handle = backend->compile(f);
    ASSERT_NE(handle, nullptr);
    handle->call_with_validate({result}, {a, b, c});
    EXPECT_EQ(read_vector<float>(result), expected);

    a->set_stale(false);
    b->set_stale(false);
    handle->call_with_validate({result}, {a, b, c});
    EXPECT_EQ(read_vector<float>(result), expected);
}

TEST(cpu_test, memory_reuse_in_place_concat_after_in_place_slice)
{
    Shape shape_a{4, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto B = make_shared<op::Slice>(A, Coordinate{0, 0}, Coordinate{2, 4});
    auto D = make_shared<op::Slice>(B, Coordinate{1, 0}, Coordinate{2, 4});
    auto E = make_shared<op::Slice>(A, Coordinate{2, 0}, Coordinate{3, 4});
    auto r = make_shared<op::Concat>(NodeVector{B, D, E}, 0);
    auto f = make_shared<Function>(r, ParameterVector{A});

    auto backend = runtime::Backend::create("CPU");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto result = backend->create_tensor(element::f32, shape_a);

    shared_ptr<runtime::Executable> handle = backend->compile(f);
    handle->call_with_validate({result}, {a});

    EXPECT_EQ((vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 5, 6, 7, 8, 9, 10, 11, 12}),
              read_vector<float>(result));
}

TEST(cpu_test, memory_reuse_in_place_slice_after_in_place_concat)
{
    Shape shape{1, 1};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto add1 = make_shared<op::Add>(A, B);
    auto C = make_shared<op::Parameter>(element::f32, shape);
    auto D = make_shared<op::Parameter>(element::f32, shape);
    auto add2 = make_shared<op::Add>(C, D);
    auto subtract = make_shared<op::Subtract>(C, A);
    auto concat = make_shared<op::Concat>(NodeVector{add1, add2, subtract}, 0);
    Shape shape_r{2, 1};
    auto slice = make_shared<op::Slice>(concat, Coordinate{0, 0}, Coordinate{2, 1});
    auto f = make_shared<Function>(slice, ParameterVector{A, B, C, D});

    auto backend = runtime::Backend::create("CPU");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{2});
    auto c = backend->create_tensor(element::f32, shape);
    copy_data(c, vector<float>{3});
    auto d = backend->create_tensor(element::f32, shape);
    copy_data(d, vector<float>{4});
    auto result = backend->create_tensor(element::f32, shape_r);

    shared_ptr<runtime::Executable> handle = backend->compile(f);
    ASSERT_NE(handle, nullptr);
    handle->call_with_validate({result}, {a, b, c, d});
    EXPECT_EQ((vector<float>{3, 7}), read_vector<float>(result));
}

TEST(cpu_test, convert_inplace)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::u8, shape);
    auto B = op::Constant::create(element::u8, shape, {1, 1, 1, 1});
    auto C = op::Constant::create(element::i8, shape, {1, 1, 1, 1});
    auto f =
        make_shared<Function>(make_shared<op::Convert>(A + B, element::i8) - C, ParameterVector{A});

    auto backend = runtime::Backend::create("CPU");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::u8, shape);
    copy_data(a, vector<uint8_t>{1, 2, 3, 254});
    auto result = backend->create_tensor(element::i8, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_EQ((vector<int8_t>{1, 2, 3, -2}), read_vector<int8_t>(result));
}

TEST(cpu_test, rotated_pooling)
{
    auto make_f = [&](bool is_4d, bool avgpool) {
        auto input_shape = is_4d ? Shape{2, 4, 4, 1} : Shape{2, 4, 4, 4, 1};
        auto rotate_order = is_4d ? AxisVector{3, 0, 1, 2} : AxisVector{4, 0, 1, 2, 3};
        auto pool_shape = is_4d ? Shape{1, 2, 4, 4} : Shape{1, 2, 4, 4, 4};
        auto window_shape = is_4d ? Shape{2, 2} : Shape{2, 2, 2};
        auto input = make_shared<op::Parameter>(element::f32, input_shape); // C, H, W, N
        auto transpose = make_shared<op::Reshape>(input, rotate_order, pool_shape);
        if (avgpool)
        {
            return make_shared<Function>(make_shared<op::AvgPool>(transpose, window_shape),
                                         ParameterVector{input});
        }
        else
        {
            return make_shared<Function>(make_shared<op::MaxPool>(transpose, window_shape),
                                         ParameterVector{input});
        }
    };

    compare_backends(make_f(true, true), make_f(true, true), "INTERPRETER", "CPU");   // 4D AvgPool
    compare_backends(make_f(true, false), make_f(true, false), "INTERPRETER", "CPU"); // 4D MaxPool
    compare_backends(make_f(false, true), make_f(false, true), "INTERPRETER", "CPU"); // 5D AvgPool
    compare_backends(
        make_f(false, false), make_f(false, false), "INTERPRETER", "CPU"); // 5D MaxPool
}
