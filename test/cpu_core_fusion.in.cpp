//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#include "ngraph/file_util.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/batch_mat_mul_transpose.hpp"
#include "ngraph/op/group_conv.hpp"
#include "ngraph/op/relu.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/softmax.hpp"
#include "ngraph/pass/batch_fusion.hpp"
#include "ngraph/pass/core_fusion.hpp"
#include "ngraph/pass/graph_rewrite.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pattern/matcher.hpp"
#include "ngraph/pattern/op/label.hpp"
#include "ngraph/pattern/op/skip.hpp"
#include "ngraph/serializer.hpp"
#include "ngraph/util.hpp"
#include "util/all_close.hpp"
#include "util/autodiff/backprop_function.hpp"
#include "util/matcher.hpp"
#include "util/random.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace ngraph;
using namespace std;

static string s_manifest = "${MANIFEST}";

#ifndef NGRAPH_JSON_DISABLE
NGRAPH_TEST(${BACKEND_NAME}, core_fusion_sigmoid_bprop_fusion)
{
    const string json_path = file_util::path_join(SERIALIZED_ZOO, "mxnet/Graph_fprop_sigmoid.json");
    const string json_string = file_util::read_file_to_string(json_path);
    stringstream ss(json_string);
    shared_ptr<Function> func = ngraph::deserialize(ss);
    auto df = autodiff::backprop_function(func);
    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    backend->compile(df);
    size_t ccg = count_ops_of_type<op::v0::SigmoidBackprop>(df);
    ASSERT_EQ(ccg, 1);
}
#endif

NGRAPH_TEST(${BACKEND_NAME}, core_fusion_zero_padded_reshaped_conv)
{
    auto X = make_shared<op::v0::Parameter>(element::f32, Shape{1, 2, 2, 1});
    auto F = make_shared<op::v0::Parameter>(element::f32, Shape{1, 1, 1, 1});

    auto pad_value =
        op::v0::Constant::create<float>(element::f32, Shape{}, std::vector<float>{0.0f});

    auto pad = make_shared<op::v0::Pad>(
        X, pad_value, CoordinateDiff{0, 1, 0, 0}, CoordinateDiff{0, 0, 1, 0});

    auto reshape = make_shared<op::v0::Reshape>(pad, AxisVector{0, 3, 1, 2}, Shape{1, 1, 3, 3});

    auto conv = make_shared<op::v0::Convolution>(reshape,
                                                 F,
                                                 Strides{1, 1},
                                                 Strides{1, 1},
                                                 CoordinateDiff{0, 0},
                                                 CoordinateDiff{0, 0},
                                                 Strides{1, 1});

    auto func = make_shared<Function>(conv, ParameterVector{X, F});

    ASSERT_EQ(count_ops_of_type<op::v0::Pad>(func), 1);

    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    backend->compile(func);

    ASSERT_EQ(count_ops_of_type<op::v0::Pad>(func), 0);
}

NGRAPH_TEST(${BACKEND_NAME}, core_fusion_zero_padded_conv)
{
    auto X = make_shared<op::v0::Parameter>(element::f32, Shape{1, 1, 2, 2});
    auto F = make_shared<op::v0::Parameter>(element::f32, Shape{1, 1, 1, 1});

    auto pad_value =
        op::v0::Constant::create<float>(element::f32, Shape{}, std::vector<float>{0.0f});

    auto pad = make_shared<op::v0::Pad>(
        X, pad_value, CoordinateDiff{0, 0, 0, 1}, CoordinateDiff{0, 0, 1, 0});

    auto conv = make_shared<op::v0::Convolution>(pad,
                                                 F,
                                                 Strides{1, 1},
                                                 Strides{1, 1},
                                                 CoordinateDiff{0, 0},
                                                 CoordinateDiff{0, 0},
                                                 Strides{1, 1});

    auto func = make_shared<Function>(conv, ParameterVector{X, F});

    ASSERT_EQ(count_ops_of_type<op::v0::Pad>(func), 1);

    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    backend->compile(func);

    ASSERT_EQ(count_ops_of_type<op::v0::Pad>(func), 0);
}

NGRAPH_TEST(${BACKEND_NAME}, core_fusion_non_zero_padded_conv)
{
    auto X = make_shared<op::v0::Parameter>(element::f32, Shape{1, 1, 2, 2});
    auto F = make_shared<op::v0::Parameter>(element::f32, Shape{1, 1, 1, 1});

    auto pad_value =
        op::v0::Constant::create<float>(element::f32, Shape{}, std::vector<float>{1.0f});

    auto pad = make_shared<op::v0::Pad>(
        X, pad_value, CoordinateDiff{0, 0, 0, 1}, CoordinateDiff{0, 0, 1, 0});

    auto conv = make_shared<op::v0::Convolution>(pad,
                                                 F,
                                                 Strides{1, 1},
                                                 Strides{1, 1},
                                                 CoordinateDiff{0, 0},
                                                 CoordinateDiff{0, 0},
                                                 Strides{1, 1});

    auto func = make_shared<Function>(conv, ParameterVector{X, F});

    ASSERT_EQ(count_ops_of_type<op::v0::Pad>(func), 1);

    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    backend->compile(func);

    ASSERT_EQ(count_ops_of_type<op::v0::Pad>(func), 1);
}

NGRAPH_TEST(${BACKEND_NAME}, core_fusion_zero_padded_conv_backprop_filters)
{
    auto X = make_shared<op::v0::Parameter>(element::f32, Shape{1, 1, 2, 2});
    auto F = make_shared<op::v0::Parameter>(element::f32, Shape{1, 1, 2, 2});

    auto pad_value =
        op::v0::Constant::create<float>(element::f32, Shape{}, std::vector<float>{0.0f});

    auto pad = make_shared<op::v0::Pad>(
        X, pad_value, CoordinateDiff{0, 0, 0, 1}, CoordinateDiff{0, 0, 1, 0});

    auto conv = make_shared<op::v0::ConvolutionBackpropFilters>(pad,
                                                                Shape{1, 1, 2, 2},
                                                                F,
                                                                Strides{1, 1},
                                                                Strides{1, 1},
                                                                CoordinateDiff{0, 0},
                                                                CoordinateDiff{0, 0},
                                                                Strides{1, 1});

    auto func = make_shared<Function>(conv, ParameterVector{X, F});

    ASSERT_EQ(count_ops_of_type<op::v0::Pad>(func), 1);

    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    backend->compile(func);

    ASSERT_EQ(count_ops_of_type<op::v0::Pad>(func), 0);
}

#ifndef NGRAPH_JSON_DISABLE
NGRAPH_TEST(${BACKEND_NAME}, core_fusion_softmax_crossentropy_fprop_1)
{
    const std::string file_name("paddlepaddle/ngraph-paddlepaddle-function3.json");
    auto cpu_f = make_function_from_file(file_name);
    auto int_f = make_function_from_file(file_name);
    test::Uniform<double> rng(-1.0, 1.0);
    vector<vector<double>> args;

    for (shared_ptr<op::v0::Parameter> param : int_f->get_parameters())
    {
        vector<double> tensor_val(shape_size(param->get_output_shape(0)));
        rng.initialize(tensor_val);
        args.push_back(tensor_val);
    }
    auto int_results = execute(int_f, args, "INTERPRETER");
    auto cpu_results = execute(cpu_f, args, "${BACKEND_NAME}");
    for (size_t i = 0; i < cpu_results.size(); i++)
    {
        EXPECT_TRUE(test::all_close(cpu_results.at(i), int_results.at(i)));
    }
    // during this optimization for numeric stability we will reduce softmax operation to
    // - summation (labels (input - max(input) - log (summation(exp ^ (input - max(input)))
    // count_of(softmax) should be equal to zero if fusion is successful
    size_t softmax = count_ops_of_type<op::v0::Softmax>(cpu_f);
    ASSERT_EQ(softmax, 0);
}

NGRAPH_TEST(${BACKEND_NAME}, core_fusion_softmax_crossentropy_fprop_2)
{
    const std::string file_name("paddlepaddle/ngraph-paddlepaddle-function1.json");
    auto cpu_f = make_function_from_file(file_name);
    auto int_f = make_function_from_file(file_name);
    test::Uniform<double> rng(-1.0, 1.0);
    vector<vector<double>> args;

    for (shared_ptr<op::v0::Parameter> param : int_f->get_parameters())
    {
        vector<double> tensor_val(shape_size(param->get_output_shape(0)));
        rng.initialize(tensor_val);
        args.push_back(tensor_val);
    }
    auto int_results = execute(int_f, args, "INTERPRETER");
    auto cpu_results = execute(cpu_f, args, "${BACKEND_NAME}");
    for (size_t i = 0; i < cpu_results.size(); i++)
    {
        EXPECT_TRUE(test::all_close(cpu_results.at(i), int_results.at(i)));
    }
    // during this optimization for numeric stability we will reduce softmax operation to
    // - summation (labels (input - max(input) - log (summation(exp ^ (input - max(input)))
    // count_of(softmax) should be equal to zero if fusion is successful
    size_t softmax = count_ops_of_type<op::v0::Softmax>(cpu_f);
    ASSERT_EQ(softmax, 0);
}

NGRAPH_TEST(${BACKEND_NAME}, core_fusion_softmax_crossentropy_bprop_with_soft_labels)
{
    const std::string file_name("paddlepaddle/ngraph-paddlepaddle-bprop0.json");
    auto cpu_f = make_function_from_file(file_name);
    auto int_f = make_function_from_file(file_name);
    test::Uniform<double> rng(-1.0, 1.0);
    vector<vector<double>> args;

    for (shared_ptr<op::v0::Parameter> param : int_f->get_parameters())
    {
        vector<double> tensor_val(shape_size(param->get_output_shape(0)));
        rng.initialize(tensor_val);
        args.push_back(tensor_val);
    }
    auto int_results = execute(int_f, args, "INTERPRETER");
    auto cpu_results = execute(cpu_f, args, "${BACKEND_NAME}");

    for (size_t i = 0; i < cpu_results.size(); i++)
    {
        EXPECT_TRUE(test::all_close(cpu_results.at(i), int_results.at(i)));
    }

    // during this optimization for numeric stability we will eliminate (softmax / softmax)
    // the number of div operator for cpu_f should be zero if the fusion is valid
    size_t divide = count_ops_of_type<op::v1::Divide>(cpu_f);
    ASSERT_EQ(divide, 0);
}

NGRAPH_TEST(${BACKEND_NAME}, core_fusion_softmax_crossentropy_bprop_with_ignore_mask)
{
    const std::string file_name("paddlepaddle/ngraph-paddlepaddle-bprop1.json");
    auto cpu_f = make_function_from_file(file_name);
    auto int_f = make_function_from_file(file_name);
    test::Uniform<double> rng(-1.0, 1.0);
    vector<vector<double>> args;

    for (shared_ptr<op::v0::Parameter> param : int_f->get_parameters())
    {
        vector<double> tensor_val(shape_size(param->get_output_shape(0)));
        rng.initialize(tensor_val);
        args.push_back(tensor_val);
    }
    auto int_results = execute(int_f, args, "INTERPRETER");
    auto cpu_results = execute(cpu_f, args, "${BACKEND_NAME}");
    for (size_t i = 0; i < cpu_results.size(); i++)
    {
        EXPECT_TRUE(test::all_close(cpu_results.at(i), int_results.at(i)));
    }

    // during this optimization for numeric stability we will eliminate (softmax / softmax)
    // the number of div operator for cpu_f should be zero if the fusion is valid
    size_t divide = count_ops_of_type<op::v1::Divide>(cpu_f);
    ASSERT_EQ(divide, 0);
}
#endif

// TODO(pthoreho): MLIR currently does not support all the op's needed for CrossEntropy+Softmax
// this results in multiple CompiledKernels and we cannot able to safely check for certain op's
// from the function created by user.
// Note: remove this guards once we have full support for CE and Softmax through MLIR
namespace
{
    void test_softmax_crossentropy(const string& backend_name,
                                   Shape input_shape,
                                   Shape label_shape,
                                   bool soft_label,
                                   int64_t ignore_index)
    {
        auto input = std::make_shared<op::v0::Parameter>(element::f64, input_shape);
        auto labels = std::make_shared<op::v0::Parameter>(element::i64, label_shape);
        auto sm_ce =
            std::make_shared<op::v0::SoftmaxCrossEntropy>(input, labels, soft_label, ignore_index);
        auto cpu_f = make_shared<Function>(sm_ce, ParameterVector{input, labels});

        test::Uniform<double> rng(-1.0, 1.0);
        vector<vector<double>> args;
        for (shared_ptr<op::v0::Parameter> param : cpu_f->get_parameters())
        {
            vector<double> tensor_val(shape_size(param->get_output_shape(0)));
            rng.initialize(tensor_val);
            args.push_back(tensor_val);
        }

        auto cpu_results = execute(cpu_f, args, backend_name);
        // if softlabels = flase, we will have one one hot encoding for labels
        if (!soft_label)
        {
            size_t onehot = count_ops_of_type<op::v0::OneHot>(cpu_f);
            ASSERT_EQ(onehot, 1);
        }
        if (ignore_index >= 0 && !soft_label)
        // check for the mask
        {
            size_t not_equal = count_ops_of_type<op::v1::NotEqual>(cpu_f);
            ASSERT_EQ(not_equal, 1);
        }
    }
}

NGRAPH_TEST(${BACKEND_NAME}, core_fusion_softmax_crossentropy)
{
    test_softmax_crossentropy("${BACKEND_NAME}", Shape{41, 37}, Shape{41, 37}, true, -1);
    test_softmax_crossentropy("${BACKEND_NAME}", Shape{41, 37}, Shape{41, 1}, false, 5);
}

namespace
{
    void test_crossentropy(const string& backend_name,
                           Shape input_shape,
                           Shape label_shape,
                           bool soft_label,
                           int64_t ignore_index)
    {
        auto input = std::make_shared<op::v0::Parameter>(element::f64, input_shape);
        auto labels = std::make_shared<op::v0::Parameter>(element::i64, label_shape);
        auto sm_ce =
            std::make_shared<op::v0::CrossEntropy>(input, labels, soft_label, ignore_index);
        auto cpu_f = make_shared<Function>(sm_ce, ParameterVector{input, labels});

        test::Uniform<double> rng(-1.0, 1.0);
        vector<vector<double>> args;
        for (shared_ptr<op::v0::Parameter> param : cpu_f->get_parameters())
        {
            vector<double> tensor_val(shape_size(param->get_output_shape(0)));
            rng.initialize(tensor_val);
            args.push_back(tensor_val);
        }

        auto cpu_results = execute(cpu_f, args, backend_name);
        // if softlabels = flase, we will have one one hot encoding for labels
        if (!soft_label)
        {
            size_t onehot = count_ops_of_type<op::v0::OneHot>(cpu_f);
            ASSERT_EQ(onehot, 1);
        }
        if (ignore_index >= 0 && !soft_label)
        // check for the mask
        {
            size_t not_equal = count_ops_of_type<op::v1::NotEqual>(cpu_f);
            ASSERT_EQ(not_equal, 1);
        }
    }
}

NGRAPH_TEST(${BACKEND_NAME}, core_fusion_crossentropy)
{
    test_crossentropy("${BACKEND_NAME}", Shape{41, 37}, Shape{41, 37}, true, -1);
    test_crossentropy("${BACKEND_NAME}", Shape{41, 37}, Shape{41, 1}, false, 5);
    test_crossentropy("${BACKEND_NAME}", Shape{10, 2, 4, 10}, Shape{10, 2, 4, 1}, false, 5);
    test_crossentropy("${BACKEND_NAME}", Shape{4, 3, 2, 4}, Shape{4, 3, 2, 4}, true, -1);
}
