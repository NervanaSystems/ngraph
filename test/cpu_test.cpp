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
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/visualize_tree.hpp"
#include "ngraph/runtime/cpu/op/convert_layout.hpp"
#include "ngraph/runtime/cpu/pass/cpu_assignment.hpp"
#include "ngraph/runtime/cpu/pass/cpu_fusion.hpp"
#include "ngraph/runtime/cpu/pass/cpu_layout.hpp"
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

TEST(cpu_test, unhandled_op)
{
    auto A = make_shared<op::Parameter>(element::f32, Shape{});
    auto unhandled = make_shared<UnhandledOp>(A);
    auto f = make_shared<Function>(unhandled, op::ParameterVector{A});
    auto backend = runtime::Backend::create("CPU");
    ASSERT_THROW(backend->compile(f), unsupported_op);
}

TEST(cpu_test, trivial_in_place_relu)
{
    auto A = make_shared<op::Parameter>(element::f32, Shape{16, 1});
    auto B = make_shared<op::Parameter>(element::f32, Shape{16, 1});
    auto add = A + B;
    auto relu = make_shared<op::Relu>(add);
    auto f = make_shared<Function>(relu, op::ParameterVector{A, B});
    auto backend = runtime::Backend::create("CPU");
    (backend->compile(f));
    ASSERT_EQ(relu->get_outputs().at(0).get_tensor().get_pool_offset(),
              add->get_outputs().at(0).get_tensor().get_pool_offset());
}

TEST(cpu_test, trivial_in_place_relu_fail)
{
    auto A = make_shared<op::Parameter>(element::f32, Shape{16, 1});
    auto B = make_shared<op::Parameter>(element::f32, Shape{16, 1});
    auto add = A + B;
    auto relu = make_shared<op::Relu>(add);
    auto add2 = relu + add;
    auto f = make_shared<Function>(add2, op::ParameterVector{A, B});
    auto backend = runtime::Backend::create("CPU");
    (backend->compile(f));
    ASSERT_NE(relu->get_outputs().at(0).get_tensor().get_pool_offset(),
              add->get_outputs().at(0).get_tensor().get_pool_offset());
}

#ifdef NGRAPH_TBB_ENABLE
TEST(cpu_test, abc_tbb)
{
    // Force TBB flow graph generation in the CPU backend
    // This has no effect on other backends
    bool use_tbb = (getenv("NGRAPH_CPU_USE_TBB") != nullptr);
    if (!use_tbb)
    {
        setenv("NGRAPH_CPU_USE_TBB", "1", 1);
    }

    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto C = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>((A + B) * C, op::ParameterVector{A, B, C});

    auto backend = runtime::Backend::create("CPU");

    // Create some tensors for input/output
    shared_ptr<runtime::Tensor> a = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> b = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> c = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> result = backend->create_tensor(element::f32, shape);

    copy_data(a, test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
    copy_data(b, test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());
    copy_data(c, test::NDArray<float, 2>({{9, 10}, {11, 12}}).get_vector());

    backend->call_with_validate(f, {result}, {a, b, c});
    EXPECT_EQ(read_vector<float>(result),
              (test::NDArray<float, 2>({{54, 80}, {110, 144}})).get_vector());

    backend->call_with_validate(f, {result}, {b, a, c});
    EXPECT_EQ(read_vector<float>(result),
              (test::NDArray<float, 2>({{54, 80}, {110, 144}})).get_vector());

    backend->call_with_validate(f, {result}, {a, c, b});
    EXPECT_EQ(read_vector<float>(result),
              (test::NDArray<float, 2>({{50, 72}, {98, 128}})).get_vector());

    if (!use_tbb)
    {
        unsetenv("NGRAPH_CPU_USE_TBB");
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
    auto f = make_shared<Function>(ResultVector{pool1_result}, op::ParameterVector{A, B});

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

    backend->call_with_validate(f, {result}, {a, b});

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
        return make_shared<Function>(NodeVector{squeeze}, op::ParameterVector{A, B});
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
        return make_shared<Function>(NodeVector{expand}, op::ParameterVector{A, B});
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
        return make_shared<Function>(NodeVector{squeeze}, op::ParameterVector{A, B});
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
        return make_shared<Function>(NodeVector{conv2}, op::ParameterVector{A, B1, B2});
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
        return make_shared<Function>(NodeVector{conv2}, op::ParameterVector{A, B1, B2});
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
        return make_shared<Function>(NodeVector{sqrt}, op::ParameterVector{A});
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
        return make_shared<Function>(NodeVector{reshape}, op::ParameterVector{A});
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

TEST(cpu_test, collapse_dims1)
{
    // Expand multiple dimensions. Ensure no extra conversions downstream
    auto make_function = []() -> std::shared_ptr<Function> {
        auto A = make_shared<op::Parameter>(element::f32, Shape{1, 4, 10, 6, 10});
        auto sum1 = make_shared<op::Sum>(A, AxisVector{1});    // Shape{1, 10, 6, 10}
        auto sum2 = make_shared<op::Sum>(sum1, AxisVector{0}); // Shape{10, 6, 10}
        return make_shared<Function>(NodeVector{sum2}, op::ParameterVector{A});
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
