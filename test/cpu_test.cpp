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
#include <thread>

#include "gtest/gtest.h"
#include "misc.hpp"
#include "ngraph/autodiff/adjoints.hpp"
#include "ngraph/file_util.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/batch_norm.hpp"
#include "ngraph/op/erf.hpp"
#include "ngraph/op/fused/conv_fused.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/pass/constant_folding.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/visualize_tree.hpp"
#include "ngraph/runtime/cpu/cpu_backend.hpp"
#include "ngraph/runtime/cpu/cpu_builder.hpp"
#include "ngraph/runtime/cpu/mkldnn_utils.hpp"
#include "ngraph/runtime/cpu/op/convert_layout.hpp"
#include "ngraph/runtime/cpu/op/max_pool_with_indices.hpp"
#include "ngraph/serializer.hpp"
#include "ngraph/util.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
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
    ASSERT_EQ(relu->output(0).get_tensor().get_pool_offset(),
              add->output(0).get_tensor().get_pool_offset());
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
    ASSERT_NE(relu->output(0).get_tensor().get_pool_offset(),
              add->output(0).get_tensor().get_pool_offset());
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
    EXPECT_TRUE(test::all_close_f(read_vector<float>(result),
                                  (test::NDArray<float, 2>({{54, 80}, {110, 144}})).get_vector()));

    handle->call_with_validate({result}, {b, a, c});
    EXPECT_TRUE(test::all_close_f(read_vector<float>(result),
                                  (test::NDArray<float, 2>({{54, 80}, {110, 144}})).get_vector()));

    handle->call_with_validate({result}, {a, c, b});
    EXPECT_TRUE(test::all_close_f(read_vector<float>(result),
                                  (test::NDArray<float, 2>({{50, 72}, {98, 128}})).get_vector()));

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

    EXPECT_TRUE(test::all_close_f(vector<float>{expected_result}, rv));
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
    EXPECT_TRUE(test::all_close_f(read_vector<float>(result), expected));
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
    EXPECT_TRUE(test::all_close_f(read_vector<float>(result), expected));

    a->set_stale(false);
    b->set_stale(false);
    handle->call_with_validate({result}, {a, b, c});
    EXPECT_TRUE(test::all_close_f(read_vector<float>(result), expected));
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

    EXPECT_TRUE(
        test::all_close_f((vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 5, 6, 7, 8, 9, 10, 11, 12}),
                          read_vector<float>(result)));
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
    EXPECT_TRUE(test::all_close_f((vector<float>{3, 7}), read_vector<float>(result)));
}

TEST(cpu_test, memory_reuse_in_place_slice_after_in_place_reshape_from_constant)
{
    Shape shape_a{2, 1, 2, 2};
    Shape shape_r{2, 1, 2, 2};
    vector<float> a_data(shape_size(shape_a));
    iota(a_data.begin(), a_data.end(), 1);

    auto A = op::Constant::create(element::f32, shape_a, a_data);
    auto reshape = make_shared<op::Reshape>(A, AxisVector{0, 1, 2, 3}, shape_r);
    Shape shape{1, 1, 2, 2};
    auto slice = make_shared<op::Slice>(reshape, Coordinate{1, 0, 0, 0}, Coordinate{2, 1, 2, 2});
    auto neg = make_shared<op::Negative>(slice);
    auto f = make_shared<Function>(neg, ParameterVector{});

    auto backend = runtime::Backend::create("CPU");

    auto result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {});
    EXPECT_TRUE(test::all_close_f(
        vector<float>{-5., -6., -7., -8.}, read_vector<float>(result), MIN_FLOAT_TOLERANCE_BITS));
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

// for float this will be 18 bits matching
// for bfloat this will be 6 bits matching
constexpr int three_quarters_of_available_bits = (MAX_FLOAT_BITS * 3) / 4;
constexpr int tolerance = FLOAT_MANTISSA_BITS - three_quarters_of_available_bits;

bool static is_codegen_mode()
{
    static bool codegen_set = false;
    static bool codegen_mode = false;
    if (!codegen_set)
    {
        const char* ngraph_codegen = std::getenv("NGRAPH_CODEGEN");
        codegen_mode = (ngraph_codegen != nullptr) && std::string(ngraph_codegen) != "0";
        codegen_set = true;
    }
    return codegen_mode;
}

TEST(cpu_test, thread_safe_calls_convolution_2d_2items)
{
    if (is_codegen_mode())
    {
        //TODO change to skip when there is a new release of gtest
        NGRAPH_WARN << "This test is skipped for CODEGEN mode.";
        return;
    }

    set_environment("NGRAPH_CPU_CONCURRENCY", "2", 1);

    Shape shape_a{2, 1, 3, 5};
    Shape shape_b{2, 1, 2, 2};
    Shape shape_r{2, 2, 2, 4};
    auto make_graph = [shape_a, shape_b] {
        auto A = make_shared<op::Parameter>(element::f32, shape_a);
        auto B = make_shared<op::Parameter>(element::f32, shape_b);
        return make_shared<Function>(
            make_shared<op::Convolution>(A,
                                         B,
                                         Strides{1, 1},        // move_strides
                                         Strides{1, 1},        // filter_dilation
                                         CoordinateDiff{0, 0}, // below_pads
                                         CoordinateDiff{0, 0}, // above_pads
                                         Strides{1, 1}),       // data_dilation
            ParameterVector{A, B});
    };

    auto backend = runtime::Backend::create("CPU");
    auto function = make_graph();

    vector<float> expected_result{
        0.63940430f,  0.04736328f,  -1.37304688f, -0.56201172f, -0.46606445f, 0.48364258f,
        1.40625000f,  0.15795898f,  -0.55004883f, 0.73339844f,  0.10668945f,  -0.95751953f,
        -0.96679688f, -0.21215820f, 1.21826172f,  -0.91894531f, 0.12402344f,  0.76953125f,
        1.20581055f,  0.65917969f,  0.62841797f,  -0.46386719f, -0.68554688f, -0.82348633f,
        0.22509766f,  -0.60864258f, -0.45166016f, -0.05249023f, 0.99462891f,  -1.09497070f,
        -0.75244141f, 0.56250000f};

    auto handle = backend->compile(function);

    auto make_call = [&]() {
        // Create some tensors for input/output
        auto a = backend->create_tensor(element::f32, shape_a);
        copy_data(
            a, vector<float>{0.67187500f,  0.54687500f,  -0.56250000f, -0.35937500f, -0.09375000f,
                             0.54687500f,  -0.54687500f, 0.89062500f,  0.82812500f,  -0.54687500f,
                             1.00000000f,  -0.07812500f, -0.89062500f, 0.40625000f,  -0.35937500f,
                             0.54687500f,  0.60937500f,  0.59375000f,  0.09375000f,  -0.21875000f,
                             0.76562500f,  0.40625000f,  -0.73437500f, -0.95312500f, -0.50000000f,
                             -0.29687500f, 0.76562500f,  -0.26562500f, -0.50000000f, 0.53125000f});
        auto b = backend->create_tensor(element::f32, shape_b);
        copy_data(b,
                  vector<float>{0.67187500f,
                                0.54687500f,
                                -0.56250000f,
                                -0.35937500f,
                                -0.09375000f,
                                0.54687500f,
                                -0.54687500f,
                                0.89062500f});
        auto result = backend->create_tensor(element::f32, shape_r);

        handle->call_with_validate({result}, {a, b});

        EXPECT_TRUE(test::all_close_f(
            vector<float>{expected_result}, read_vector<float>(result), tolerance));
    };

    std::thread call1(make_call);
    std::thread call2(make_call);
    std::thread call3(make_call);
    call1.join();
    call2.join();
    call3.join();

    unset_environment("NGRAPH_CPU_CONCURRENCY");
}

TEST(cpu_test, constant_reshape)
{
    Shape shape_in{2, 4};
    Shape shape_out{2, 4, 1};

    const vector<float> values_in{0, 1, 2, 3, 4, 5, 6, 7};
    auto constant = make_shared<op::Constant>(element::f32, shape_in, values_in);
    auto reshape = make_shared<op::Reshape>(constant, AxisVector{0, 1}, shape_out);
    auto f = make_shared<Function>(reshape, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>(
        ngraph::runtime::cpu::GetGlobalCFDispatcherCPU());
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::Reshape>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const =
        std::dynamic_pointer_cast<op::Constant>(f->get_results().at(0)->get_argument(0));
    ASSERT_TRUE(new_const);
    const vector<float> values_out = new_const->get_vector<float>();

    EXPECT_TRUE(test::all_close_f(values_in, values_out, MIN_FLOAT_TOLERANCE_BITS));
}

TEST(cpu_test, constant_reshape_permute)
{
    Shape shape_in{2, 4};
    Shape shape_out{4, 2};

    vector<double> values_in{0, 1, 2, 3, 4, 5, 6, 7};
    auto constant = make_shared<op::Constant>(element::f64, shape_in, values_in);
    auto reshape = make_shared<op::Reshape>(constant, AxisVector{1, 0}, shape_out);
    auto f = make_shared<Function>(reshape, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>(
        ngraph::runtime::cpu::GetGlobalCFDispatcherCPU());
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::Reshape>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const =
        std::dynamic_pointer_cast<op::Constant>(f->get_results().at(0)->get_argument(0));
    ASSERT_TRUE(new_const);
    const vector<double> values_out = new_const->get_vector<double>();

    const vector<double> values_permute{0, 4, 1, 5, 2, 6, 3, 7};
    EXPECT_TRUE(test::all_close_f(values_permute, values_out, MIN_FLOAT_TOLERANCE_BITS));
}

TEST(cpu_test, constant_broadcast)
{
    Shape shape_in{2};
    Shape shape_out{2, 4};

    vector<int> values_in{0, 1};
    auto constant = make_shared<op::Constant>(element::i32, shape_in, values_in);
    auto broadcast = make_shared<op::Broadcast>(constant, shape_out, AxisSet{1});
    auto f = make_shared<Function>(broadcast, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>(
        ngraph::runtime::cpu::GetGlobalCFDispatcherCPU());
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::Broadcast>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const =
        std::dynamic_pointer_cast<op::Constant>(f->get_results().at(0)->get_argument(0));
    ASSERT_TRUE(new_const);
    auto values_out = new_const->get_vector<int>();

    vector<int> values_permute{0, 0, 0, 0, 1, 1, 1, 1};
    ASSERT_EQ(values_permute, values_out);
}

TEST(cpu_test, constant_pad_exterior)
{
    Shape shape_in{2};

    vector<int> values_in{777, 888};
    auto constant = make_shared<op::Constant>(element::i32, shape_in, values_in);
    auto pad_value = make_shared<op::Constant>(element::i32, Shape{}, vector<int>{111});

    CoordinateDiff padding_below{1};
    CoordinateDiff padding_above{2};

    auto broadcast = make_shared<op::Pad>(constant, pad_value, padding_below, padding_above);
    auto f = make_shared<Function>(broadcast, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>(
        ngraph::runtime::cpu::GetGlobalCFDispatcherCPU());
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::Pad>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const =
        std::dynamic_pointer_cast<op::Constant>(f->get_results().at(0)->get_argument(0));
    ASSERT_TRUE(new_const);
    auto values_out = new_const->get_vector<int>();

    vector<int> padded_values{111, 777, 888, 111, 111};
    ASSERT_EQ(padded_values, values_out);
}

template <typename T>
static std::vector<T> get_result_constant(std::shared_ptr<Function> f, size_t pos)
{
    auto new_const =
        std::dynamic_pointer_cast<op::Constant>(f->get_results().at(pos)->get_argument(0));
    return new_const->get_vector<T>();
}

TEST(cpu_test, constant_unary_binary)
{
    Shape shape_in{4};
    vector<int> values_a{1, 2, 3, 4};
    vector<int> values_b{1, 2, 3, 4};
    vector<int> values_c{-1, -1, -1, -1};
    vector<int> values_d{1, 4, 9, 16};
    vector<int> values_e{1, -2, -3, 4};
    auto a = make_shared<op::Constant>(element::i32, shape_in, values_a);
    auto b = make_shared<op::Constant>(element::i32, shape_in, values_b);
    auto c = make_shared<op::Constant>(element::i32, shape_in, values_c);
    auto d = make_shared<op::Constant>(element::i32, shape_in, values_d);
    auto e = make_shared<op::Constant>(element::i32, shape_in, values_e);

    auto add = a + b;
    auto sub = a - b;
    auto mul = a * b;
    auto divn = a / b;
    auto min = make_shared<op::Minimum>(c, a);
    auto max = make_shared<op::Maximum>(a, c);
    auto absn = make_shared<op::Abs>(c);
    auto neg = make_shared<op::Negative>(c);
    auto sqrt = make_shared<op::Sqrt>(d);
    auto neg_sqrt = make_shared<op::Sqrt>(c);
    auto relu = make_shared<op::Relu>(e);

    auto f = make_shared<Function>(NodeVector{add, sub, mul, divn, min, max, absn, neg, sqrt, relu},
                                   ParameterVector{});
    auto f_error = make_shared<Function>(NodeVector{neg_sqrt}, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>(
        ngraph::runtime::cpu::GetGlobalCFDispatcherCPU());
    pass_manager.run_passes(f);

    //expected values
    vector<int> add_expected{2, 4, 6, 8};
    vector<int> sub_expected{0, 0, 0, 0};
    vector<int> mul_expected{1, 4, 9, 16};
    vector<int> div_expected{1, 1, 1, 1};
    vector<int> min_expected{-1, -1, -1, -1};
    vector<int> max_expected{1, 2, 3, 4};
    vector<int> abs_neg_expected{1, 1, 1, 1};
    vector<int> sqrt_expected{1, 2, 3, 4};
    vector<int> relu_expected{1, 0, 0, 4};

    ASSERT_EQ(get_result_constant<int>(f, 0), add_expected);
    ASSERT_EQ(get_result_constant<int>(f, 1), sub_expected);
    ASSERT_EQ(get_result_constant<int>(f, 2), mul_expected);
    ASSERT_EQ(get_result_constant<int>(f, 3), div_expected);
    ASSERT_EQ(get_result_constant<int>(f, 4), min_expected);
    ASSERT_EQ(get_result_constant<int>(f, 5), max_expected);
    ASSERT_EQ(get_result_constant<int>(f, 6), abs_neg_expected);
    ASSERT_EQ(get_result_constant<int>(f, 7), abs_neg_expected);
    ASSERT_EQ(get_result_constant<int>(f, 8), sqrt_expected);
    ASSERT_EQ(get_result_constant<int>(f, 9), relu_expected);
    ASSERT_ANY_THROW(pass_manager.run_passes(f_error));
}

TEST(cpu_test, conv_test_winograd)
{
    /*  This test checks for the cpu specific graph pass handling for conv_winograd implementation. 
        On SKX with MKLDNN version >= v0.18.0, mkldnn_verbose should match the following

        mkldnn_verbose,info,Intel(R) MKL-DNN v0.18.0 (Git Hash 863ff6e7042cec7d2e29897fe9f0872e0888b0fc),Intel(R) Advanced Vector Extensions 512 (Intel(R) AVX-512) with AVX512BW, AVX512VL, and AVX512DQ extensions
        mkldnn_verbose,create,reorder,simple:any,undef,in:f32_nchw out:f32_OIhw16i16o,num:1,64x3x3x3,0.0129395
        mkldnn_verbose,exec,reorder,simple:any,undef,in:f32_nchw out:f32_OIhw16i16o,num:1,64x3x3x3,0.414062
        mkldnn_verbose,create,reorder,simple:any,undef,in:f32_nchw out:f32_nChw16c,num:1,64x3x224x224,0.0119629
        mkldnn_verbose,exec,reorder,simple:any,undef,in:f32_nchw out:f32_nChw16c,num:1,64x3x224x224,19.302
        mkldnn_verbose,create,convolution,jit_wino_4x3:avx512_core,forward_training,fsrc:nChw16c fwei:OIhw16i16o fbia:undef fdst:nChw16c,alg:convolution_winograd,mb64_ic3oc64_ih224oh224kh3sh1dh0ph1_iw224ow224kw3sw1dw0pw1,1.84106
        mkldnn_verbose,exec,convolution,jit_wino_4x3:avx512_core,forward_training,fsrc:nChw16c fwei:OIhw16i16o fbia:undef fdst:nChw16c,alg:convolution_winograd,mb64_ic3oc64_ih224oh224kh3sh1dh0ph1_iw224ow224kw3sw1dw0pw1,46.6631
        mkldnn_verbose,create,reorder,jit:uni,undef,in:f32_nChw16c out:f32_nchw,num:1,64x64x224x224,0.279053
        mkldnn_verbose,exec,reorder,jit:uni,undef,in:f32_nChw16c out:f32_nchw,num:1,64x64x224x224,100.219
    */
    auto make_function = []() -> std::shared_ptr<Function> {
        auto input = make_shared<op::Parameter>(element::f32, Shape{64, 3, 224, 224});
        auto filter = make_shared<op::Parameter>(element::f32, Shape{64, 3, 3, 3});
        auto conv = make_shared<op::Convolution>(input,
                                                 filter,
                                                 Strides{1, 1},
                                                 Strides{1, 1},
                                                 CoordinateDiff{1, 1},
                                                 CoordinateDiff{1, 1},
                                                 Strides{1, 1});
        return make_shared<Function>(conv, ParameterVector{input, filter});

    };
    auto backend = runtime::Backend::create("CPU");
    auto cpu_f = make_function();

    test::Uniform<float> rng(-100.0f, 100.0f);
    vector<vector<float>> args;
    for (shared_ptr<op::Parameter> param : cpu_f->get_parameters())
    {
        vector<float> tensor_val(shape_size(param->get_shape()));
        rng.initialize(tensor_val);
        args.push_back(tensor_val);
    }
    auto cpu_results = execute(cpu_f, args, "CPU");
}

TEST(cpu_test, conv_negative_padding)
{
    auto make_f = [&]() {
        Shape shape_a{1, 16, 2, 2};
        auto A = make_shared<op::Parameter>(element::f32, shape_a);
        Shape shape_b{32, 16, 1, 1};
        auto B = make_shared<op::Parameter>(element::f32, shape_b);
        auto conv1 = make_shared<op::Convolution>(A,
                                                  B,
                                                  Strides{1, 1},
                                                  Strides{1, 1},
                                                  CoordinateDiff{-1, -1},
                                                  CoordinateDiff{0, 0},
                                                  Strides{1, 1});
        return make_shared<Function>(conv1, ParameterVector{A, B});

    };
    compare_backends(make_f(), make_f(), "CPU", "INTERPRETER");
}

TEST(cpu_test, gauss_error_function_erf_float32)
{
    auto make_function = []() -> std::shared_ptr<Function> {
        auto A = make_shared<op::Parameter>(element::f32, Shape{1, 4, 10, 6, 10});
        auto erf = make_shared<op::Erf>(A);
        return make_shared<Function>(erf, ParameterVector{A});
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

TEST(cpu_test, gauss_error_function_erf_int32)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::i32, shape);
    auto make_function = [&]() -> std::shared_ptr<Function> {
        auto erf = make_shared<op::Erf>(A);
        return make_shared<Function>(erf, ParameterVector{A});
    };

    auto backend = runtime::Backend::create("CPU");
    auto cpu_f = make_function();

    auto input_nd_array = test::NDArray<int, 2>({{45, 2}, {7, 9}});
    auto expected_result_nd_array =
        test::NDArray<int, 2>({{static_cast<int>(std::erf(45)), static_cast<int>(std::erf(2))},
                               {static_cast<int>(std::erf(7)), static_cast<int>(std::erf(9))}});

    // Create some tensors for input/output
    shared_ptr<runtime::Tensor> a = backend->create_tensor(element::i32, shape);
    shared_ptr<runtime::Tensor> result = backend->create_tensor(element::i32, shape);

    copy_data(a, input_nd_array.get_vector());

    auto handle = backend->compile(cpu_f);
    handle->call_with_validate({result}, {a});

    auto result_values = read_vector<int>(result);
    auto expected_values = expected_result_nd_array.get_vector();
    ASSERT_EQ(result_values, expected_values);
}

TEST(cpu_test, max_pool_with_indices_2d_2channel_2image)
{
    Shape shape_a{2, 2, 5, 5};
    Shape window_shape{2, 3};
    auto window_movement_strides = Strides{1, 1};
    Shape padding_below{0, 0};
    Shape padding_above{0, 0};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto max_pool = make_shared<op::MaxPoolWithIndices>(
        A, window_shape, window_movement_strides, padding_below, padding_above);
    Shape shape_r{2, 2, 4, 3};
    auto data = make_shared<op::Result>(make_shared<op::GetOutputElement>(max_pool, 0));
    auto indices = make_shared<op::Result>(make_shared<op::GetOutputElement>(max_pool, 1));
    auto f = make_shared<Function>(ResultVector{data, indices}, ParameterVector{A});

    auto backend = runtime::Backend::create("CPU");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 4>({{{{0, 1, 0, 2, 1}, // img 0 chan 0
                                         {0, 3, 2, 0, 0},
                                         {2, 0, 0, 0, 1},
                                         {2, 0, 1, 1, 2},
                                         {0, 2, 1, 0, 0}},

                                        {{0, 0, 0, 2, 0}, // img 0 chan 1
                                         {0, 2, 3, 0, 1},
                                         {2, 0, 1, 0, 2},
                                         {3, 1, 0, 0, 0},
                                         {2, 0, 0, 0, 0}}},

                                       {{{0, 2, 1, 1, 0}, // img 1 chan 0
                                         {0, 0, 2, 0, 1},
                                         {0, 0, 1, 2, 3},
                                         {2, 0, 0, 3, 0},
                                         {0, 0, 0, 0, 0}},

                                        {{2, 1, 0, 0, 1}, // img 1 chan 1
                                         {0, 2, 0, 0, 0},
                                         {1, 1, 2, 0, 2},
                                         {1, 1, 1, 0, 1},
                                         {1, 0, 0, 0, 2}}}})
                  .get_vector());
    auto result_data = backend->create_tensor(element::f32, shape_r);
    auto result_indices = backend->create_tensor(element::i32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result_data, result_indices}, {a});
    EXPECT_TRUE(test::all_close_f((test::NDArray<float, 4>({{{{3, 3, 2}, // img 0 chan 0
                                                              {3, 3, 2},
                                                              {2, 1, 2},
                                                              {2, 2, 2}},

                                                             {{3, 3, 3}, // img 0 chan 1
                                                              {3, 3, 3},
                                                              {3, 1, 2},
                                                              {3, 1, 0}}},

                                                            {{{2, 2, 2}, // img 1 chan 0
                                                              {2, 2, 3},
                                                              {2, 3, 3},
                                                              {2, 3, 3}},

                                                             {{2, 2, 1}, // img 1 chan 1
                                                              {2, 2, 2},
                                                              {2, 2, 2},
                                                              {1, 1, 2}}}})
                                       .get_vector()),
                                  read_vector<float>(result_data),
                                  MIN_FLOAT_TOLERANCE_BITS));

    EXPECT_TRUE(test::all_close((test::NDArray<int, 4>({{{{4, 3, 1}, // img 0 chan 0
                                                          {1, 0, 0},
                                                          {0, 4, 5},
                                                          {0, 3, 2}},

                                                         {{5, 4, 3}, // img 0 chan 1
                                                          {2, 1, 0},
                                                          {3, 1, 2},
                                                          {0, 0, 0}}},

                                                        {{{1, 0, 3}, // img 1 chan 0
                                                          {2, 1, 5},
                                                          {3, 5, 2},
                                                          {0, 2, 1}},

                                                         {{0, 3, 2}, // img 1 chan 1
                                                          {1, 0, 3},
                                                          {2, 1, 0},
                                                          {0, 0, 5}}}})
                                     .get_vector()),
                                read_vector<int>(result_indices)));
}

TEST(cpu_test, max_pool_with_indices_bprop_2d_2channel_2image)
{
    Shape shape_a{2, 2, 5, 5};
    Shape window_shape{2, 3};
    auto window_movement_strides = Strides{1, 1};
    Shape padding_below{0, 0};
    Shape padding_above{0, 0};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_i{2, 2, 4, 3};
    auto indices = make_shared<op::Parameter>(element::i32, shape_i);
    auto delta = make_shared<op::Parameter>(element::f32, shape_i);

    auto max_pool_bprop = make_shared<op::MaxPoolWithIndicesBackprop>(
        A, delta, indices, window_shape, window_movement_strides, padding_below, padding_above);

    auto f = make_shared<Function>(max_pool_bprop, ParameterVector{A, delta, indices});

    auto backend = runtime::Backend::create("CPU");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 4>({{{{0, 1, 0, 2, 1}, // img 0 chan 0
                                         {0, 3, 2, 0, 0},
                                         {2, 0, 0, 0, 1},
                                         {2, 0, 1, 1, 2},
                                         {0, 2, 1, 0, 0}},

                                        {{0, 0, 0, 2, 0}, // img 0 chan 1
                                         {0, 2, 3, 0, 1},
                                         {2, 0, 1, 0, 2},
                                         {3, 1, 0, 0, 0},
                                         {2, 0, 0, 0, 0}}},

                                       {{{0, 2, 1, 1, 0}, // img 1 chan 0
                                         {0, 0, 2, 0, 1},
                                         {0, 0, 1, 2, 3},
                                         {2, 0, 0, 3, 0},
                                         {0, 0, 0, 0, 0}},

                                        {{2, 1, 0, 0, 1}, // img 1 chan 1
                                         {0, 2, 0, 0, 0},
                                         {1, 1, 2, 0, 2},
                                         {1, 1, 1, 0, 1},
                                         {1, 0, 0, 0, 2}}}})
                  .get_vector());

    auto i = backend->create_tensor(element::i32, shape_i);
    copy_data(i,
              test::NDArray<int, 4>({{{{4, 3, 1}, // img 0 chan 0
                                       {1, 0, 0},
                                       {0, 4, 5},
                                       {0, 3, 2}},

                                      {{5, 4, 3}, // img 0 chan 1
                                       {2, 1, 0},
                                       {3, 1, 2},
                                       {0, 0, 0}}},

                                     {{{1, 0, 3}, // img 1 chan 0
                                       {2, 1, 5},
                                       {3, 5, 2},
                                       {0, 2, 1}},

                                      {{0, 3, 2}, // img 1 chan 1
                                       {1, 0, 3},
                                       {2, 1, 0},
                                       {0, 0, 5}}}})
                  .get_vector());

    auto d = backend->create_tensor(element::f32, shape_i);
    copy_data(d,
              test::NDArray<float, 4>({{{{0.3f, 0.3f, 0.2f}, // img 0 chan 0
                                         {0.3f, 0.3f, 0.2f},
                                         {0.2f, 0.1f, 0.2f},
                                         {0.2f, 0.2f, 0.2f}},

                                        {{0.3f, 0.3f, 0.3f}, // img 0 chan 1
                                         {0.3f, 0.3f, 0.3f},
                                         {0.3f, 0.1f, 0.2f},
                                         {0.3f, 0.1f, 0.4f}}},

                                       {{{0.2f, 0.2f, 0.2f}, // img 1 chan 0
                                         {0.2f, 0.2f, 0.3f},
                                         {0.2f, 0.3f, 0.3f},
                                         {0.2f, 0.3f, 0.3f}},

                                        {{0.2f, 0.2f, 0.1f}, // img 1 chan 1
                                         {0.2f, 0.2f, 0.2f},
                                         {0.2f, 0.2f, 0.2f},
                                         {0.1f, 0.1f, 0.2f}}}})
                  .get_vector());

    auto result = backend->create_tensor(element::f32, shape_a);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, d, i});
    EXPECT_TRUE(test::all_close_f((test::NDArray<float, 4>({{{{0, 0, 0, 0.2, 0}, // img 0 chan 0
                                                              {0, 1.2, 0.2, 0, 0},
                                                              {0.2, 0, 0, 0, 0},
                                                              {0.2, 0, 0.1, 0, 0.4},
                                                              {0, 0.2, 0, 0, 0}},

                                                             {{0, 0, 0, 0, 0}, // img 0 chan 1
                                                              {0, 0, 1.8, 0, 0},
                                                              {0, 0, 0.1, 0, 0.2},
                                                              {0.6, 0.1, 0.4, 0, 0},
                                                              {0, 0, 0, 0, 0}}},

                                                            {{{0, 0.4, 0, 0, 0}, // img 1 chan 0
                                                              {0, 0, 0.6, 0, 0},
                                                              {0, 0, 0, 0, 0.6},
                                                              {0.4, 0, 0, 0.9, 0},
                                                              {0, 0, 0, 0, 0}},

                                                             {{0.2, 0, 0, 0, 0.1}, // img 1 chan 1
                                                              {0, 0.6, 0, 0, 0},
                                                              {0, 0, 0.8, 0, 0},
                                                              {0.1, 0.1, 0, 0, 0},
                                                              {0, 0, 0, 0, 0.2}}}})
                                       .get_vector()),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

TEST(cpu_test, max_pool_bprop_2d_2channel_2image)
{
    Shape shape_a{2, 2, 5, 5};
    Shape window_shape{2, 3};
    auto window_movement_strides = Strides{1, 1};
    Shape padding_below{0, 0};
    Shape padding_above{0, 0};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_i{2, 2, 4, 3};
    auto delta = make_shared<op::Parameter>(element::f32, shape_i);

    auto max_pool_bprop = make_shared<op::MaxPoolBackprop>(
        A, delta, window_shape, window_movement_strides, padding_below, padding_above);

    auto f = make_shared<Function>(max_pool_bprop, ParameterVector{A, delta});

    auto backend = runtime::Backend::create("CPU");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 4>({{{{0, 1, 0, 2, 1}, // img 0 chan 0
                                         {0, 3, 2, 0, 0},
                                         {2, 0, 0, 0, 1},
                                         {2, 0, 1, 1, 2},
                                         {0, 2, 1, 0, 0}},

                                        {{0, 0, 0, 2, 0}, // img 0 chan 1
                                         {0, 2, 3, 0, 1},
                                         {2, 0, 1, 0, 2},
                                         {3, 1, 0, 0, 0},
                                         {2, 0, 0, 0, 0}}},

                                       {{{0, 2, 1, 1, 0}, // img 1 chan 0
                                         {0, 0, 2, 0, 1},
                                         {0, 0, 1, 2, 3},
                                         {2, 0, 0, 3, 0},
                                         {0, 0, 0, 0, 0}},

                                        {{2, 1, 0, 0, 1}, // img 1 chan 1
                                         {0, 2, 0, 0, 0},
                                         {1, 1, 2, 0, 2},
                                         {1, 1, 1, 0, 1},
                                         {1, 0, 0, 0, 2}}}})
                  .get_vector());

    auto d = backend->create_tensor(element::f32, shape_i);
    copy_data(d,
              test::NDArray<float, 4>({{{{0.3, 0.3, 0.2}, // img 0 chan 0
                                         {0.3, 0.3, 0.2},
                                         {0.2, 0.1, 0.2},
                                         {0.2, 0.2, 0.2}},

                                        {{0.3, 0.3, 0.3}, // img 0 chan 1
                                         {0.3, 0.3, 0.3},
                                         {0.3, 0.1, 0.2},
                                         {0.3, 0.1, 0.4}}},

                                       {{{0.2, 0.2, 0.2}, // img 1 chan 0
                                         {0.2, 0.2, 0.3},
                                         {0.2, 0.3, 0.3},
                                         {0.2, 0.3, 0.3}},

                                        {{0.2, 0.2, 0.1}, // img 1 chan 1
                                         {0.2, 0.2, 0.2},
                                         {0.2, 0.2, 0.2},
                                         {0.1, 0.1, 0.2}}}})
                  .get_vector());

    auto result = backend->create_tensor(element::f32, shape_a);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, d});
    EXPECT_TRUE(test::all_close_f((test::NDArray<float, 4>({{{{0, 0, 0, 0.2, 0}, // img 0 chan 0
                                                              {0, 1.2, 0.2, 0, 0},
                                                              {0.2, 0, 0, 0, 0},
                                                              {0.2, 0, 0.1, 0, 0.4},
                                                              {0, 0.2, 0, 0, 0}},

                                                             {{0, 0, 0, 0, 0}, // img 0 chan 1
                                                              {0, 0, 1.8, 0, 0},
                                                              {0, 0, 0.1, 0, 0.2},
                                                              {0.6, 0.1, 0.4, 0, 0},
                                                              {0, 0, 0, 0, 0}}},

                                                            {{{0, 0.4, 0, 0, 0}, // img 1 chan 0
                                                              {0, 0, 0.6, 0, 0},
                                                              {0, 0, 0, 0, 0.6},
                                                              {0.4, 0, 0, 0.9, 0},
                                                              {0, 0, 0, 0, 0}},

                                                             {{0.2, 0, 0, 0, 0.1}, // img 1 chan 1
                                                              {0, 0.6, 0, 0, 0},
                                                              {0, 0, 0.8, 0, 0},
                                                              {0.1, 0.1, 0, 0, 0},
                                                              {0, 0, 0, 0, 0.2}}}})
                                       .get_vector()),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

TEST(cpu_test, avg_pool_bprop_2d_2channel_2image)
{
    Shape shape_a{2, 2, 3, 3};
    Shape window_shape{2, 2};
    auto window_movement_strides = Strides{1, 1};
    Shape padding_below{0, 0};
    Shape padding_above{0, 0};
    Shape shape_d{2, 2, 2, 2};
    auto delta = make_shared<op::Parameter>(element::f32, shape_d);

    auto avg_pool_bprop = make_shared<op::AvgPoolBackprop>(
        shape_a, delta, window_shape, window_movement_strides, padding_below, padding_above, false);

    auto f = make_shared<Function>(avg_pool_bprop, ParameterVector{delta});

    auto backend = runtime::Backend::create("CPU");

    // Create some tensors for input/output
    auto d = backend->create_tensor(element::f32, shape_d);
    copy_data(d,
              test::NDArray<float, 4>({{{{0.3, 0.3}, // img 0 chan 0
                                         {0.3, 0.3}},

                                        {{0.2, 0.2}, // img 0 chan 1
                                         {0.2, 0.2}}},

                                       {{{0.1, 0.1}, // img 1 chan 0
                                         {0.1, 0.1}},

                                        {{0.4, 0.4}, // img 1 chan 1
                                         {0.4, 0.4}}}})
                  .get_vector());

    auto result = backend->create_tensor(element::f32, shape_a);

    float denom = 2 * 2;

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {d});
    EXPECT_TRUE(test::all_close_f(
        (test::NDArray<float, 4>({{{{0.3f / denom, 0.6f / denom, 0.3f / denom}, // img 0 chan 0
                                    {0.6f / denom, 1.2f / denom, 0.6f / denom},
                                    {0.3f / denom, 0.6f / denom, 0.3f / denom}},

                                   {{0.2f / denom, 0.4f / denom, 0.2f / denom}, // img 0 chan 1
                                    {0.4f / denom, 0.8f / denom, 0.4f / denom},
                                    {0.2f / denom, 0.4f / denom, 0.2f / denom}}},

                                  {{{0.1f / denom, 0.2f / denom, 0.1f / denom}, // img 1 chan 0
                                    {0.2f / denom, 0.4f / denom, 0.2f / denom},
                                    {0.1f / denom, 0.2f / denom, 0.1f / denom}},

                                   {{0.4f / denom, 0.8f / denom, 0.4f / denom}, // img 1 chan 1
                                    {0.8f / denom, 1.6f / denom, 0.8f / denom},
                                    {0.4f / denom, 0.8f / denom, 0.4f / denom}}}})
             .get_vector()),
        read_vector<float>(result),
        MIN_FLOAT_TOLERANCE_BITS));
}
