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
#include "misc.hpp"
#include "ngraph/autodiff/adjoints.hpp"
#include "ngraph/env_util.hpp"
#include "ngraph/file_util.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/batch_mat_mul_transpose.hpp"
#include "ngraph/op/batch_norm.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/conv_fused.hpp"
#include "ngraph/op/dequantize.hpp"
#include "ngraph/op/experimental/generate_mask.hpp"
#include "ngraph/op/experimental/quantized_conv_bias.hpp"
#include "ngraph/op/gelu.hpp"
#include "ngraph/op/group_conv.hpp"
#include "ngraph/op/max_pool.hpp"
#include "ngraph/op/negative.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/op/quantize.hpp"
#include "ngraph/op/quantized_convolution.hpp"
#include "ngraph/op/relu.hpp"
#include "ngraph/op/result.hpp"
#include "ngraph/op/reverse_sequence.hpp"
#include "ngraph/op/sigmoid.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/op/tanh.hpp"
#include "ngraph/pass/algebraic_simplification.hpp"
#include "ngraph/pass/batch_fusion.hpp"
#include "ngraph/pass/core_fusion.hpp"
#include "ngraph/pass/graph_rewrite.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/reshape_elimination.hpp"
#include "ngraph/pattern/matcher.hpp"
#include "ngraph/pattern/op/label.hpp"
#include "ngraph/pattern/op/skip.hpp"
#include "ngraph/runtime/cpu/cpu_layout_descriptor.hpp"
#include "ngraph/runtime/cpu/cpu_tensor.hpp"
#include "ngraph/runtime/cpu/op/batch_norm_relu.hpp"
#include "ngraph/runtime/cpu/op/bounded_relu.hpp"
#include "ngraph/runtime/cpu/op/conv_add.hpp"
#include "ngraph/runtime/cpu/op/conv_relu.hpp"
#include "ngraph/runtime/cpu/op/convert_layout.hpp"
#include "ngraph/runtime/cpu/op/deconv.hpp"
#include "ngraph/runtime/cpu/op/dropout.hpp"
#include "ngraph/runtime/cpu/op/gelu_backprop.hpp"
#include "ngraph/runtime/cpu/op/group_conv_bias.hpp"
#include "ngraph/runtime/cpu/op/leaky_relu.hpp"
#include "ngraph/runtime/cpu/op/lstm.hpp"
#include "ngraph/runtime/cpu/op/matmul_bias.hpp"
#include "ngraph/runtime/cpu/op/rnn.hpp"
#include "ngraph/runtime/cpu/op/rnn_utils.hpp"
#include "ngraph/runtime/cpu/op/sigmoid_mul.hpp"
#include "ngraph/runtime/cpu/op/update_slice.hpp"
#include "ngraph/runtime/cpu/pass/cpu_fusion.hpp"
#include "ngraph/runtime/cpu/pass/cpu_mat_fusion.hpp"
#include "ngraph/runtime/cpu/pass/cpu_post_layout_optimizations.hpp"
#include "ngraph/runtime/cpu/pass/cpu_rnn_fusion.hpp"
#include "ngraph/runtime/cpu/pass/cpu_workspace_insertion.hpp"
#include "ngraph/serializer.hpp"
#include "ngraph/util.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/autodiff/backprop_function.hpp"
#include "util/autodiff/numeric_compare.hpp"
#include "util/matcher.hpp"
#include "util/random.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace ngraph;
using namespace std;

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, cpu_fusion_gemm_cpu_broadcast_row)
{
    Shape shapeA{3, 2};
    Shape shapeB{2, 3};
    Shape shapeC{2, 2};
    auto A = make_shared<op::v0::Parameter>(element::f32, shapeA);
    auto B = make_shared<op::v0::Parameter>(element::f32, shapeB);

    auto bias =
        op::v0::Constant::create<float>(element::f32, Shape{2}, std::vector<float>{2.0f, 3.0f});

    auto cg = make_shared<op::MatmulBias>(
        A, B, bias, A->get_output_shape(0), B->get_output_shape(0), true, true, AxisSet{0});

    auto f = make_shared<Function>(cg, ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    shared_ptr<runtime::Tensor> a = backend->create_tensor(element::f32, shapeA);
    shared_ptr<runtime::Tensor> b = backend->create_tensor(element::f32, shapeB);
    shared_ptr<runtime::Tensor> result = backend->create_tensor(element::f32, shapeC);

    vector<float> dataA{1.0f, 4.0f, 1.0f, 4.0f, 1.0f, 4.0f};
    vector<float> dataB{3.0f, 3.0f, 3.0f, 9.0f, 9.0f, 9.0f};
    copy_data(a, dataA);
    copy_data(b, dataB);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    vector<float> expected{11, 30, 38, 111};
    EXPECT_TRUE(test::all_close_f(read_vector<float>(result), expected, MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, cpu_fusion_gemm_cpu_broadcast_column)
{
    Shape shapeA{3, 2};
    Shape shapeB{2, 3};
    Shape shapeC{2, 2};
    auto A = make_shared<op::v0::Parameter>(element::f32, shapeA);
    auto B = make_shared<op::v0::Parameter>(element::f32, shapeB);

    auto bias =
        op::v0::Constant::create<float>(element::f32, Shape{2}, std::vector<float>{2.0f, 3.0f});

    auto cg = make_shared<op::MatmulBias>(
        A, B, bias, A->get_output_shape(0), B->get_output_shape(0), true, true, AxisSet{1});

    auto f = make_shared<Function>(cg, ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    shared_ptr<runtime::Tensor> a = backend->create_tensor(element::f32, shapeA);
    shared_ptr<runtime::Tensor> b = backend->create_tensor(element::f32, shapeB);
    shared_ptr<runtime::Tensor> result = backend->create_tensor(element::f32, shapeC);

    vector<float> dataA{1.0f, 4.0f, 1.0f, 4.0f, 1.0f, 4.0f};
    vector<float> dataB{3.0f, 3.0f, 3.0f, 9.0f, 9.0f, 9.0f};
    copy_data(a, dataA);
    copy_data(b, dataB);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    vector<float> expected{11, 29, 39, 111};
    EXPECT_TRUE(test::all_close_f(read_vector<float>(result), expected, MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, cpu_fusion_gemm_cpu_broadcast_matrix)
{
    Shape shapeA{3, 2};
    Shape shapeB{2, 3};
    Shape shapeC{2, 2};
    auto A = make_shared<op::v0::Parameter>(element::f32, shapeA);
    auto B = make_shared<op::v0::Parameter>(element::f32, shapeB);

    auto reshape_w = make_shared<op::v0::Reshape>(A, AxisVector{1, 0}, Shape{2, 3});
    auto reshape_x = make_shared<op::v0::Reshape>(B, AxisVector{1, 0}, Shape{3, 2});

    auto one = op::v0::Constant::create<float>(element::f32, Shape{}, std::vector<float>{1.0f});

    auto broadcast = make_shared<op::v0::Broadcast>(one, shapeC, AxisSet{0, 1});
    auto cg = make_shared<op::MatmulBias>(
        A, B, one, A->get_output_shape(0), B->get_output_shape(0), true, true, AxisSet{0, 1});

    auto f = make_shared<Function>(cg, ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    shared_ptr<runtime::Tensor> a = backend->create_tensor(element::f32, shapeA);
    shared_ptr<runtime::Tensor> b = backend->create_tensor(element::f32, shapeB);
    shared_ptr<runtime::Tensor> result = backend->create_tensor(element::f32, shapeC);

    vector<float> dataA{1.0f, 4.0f, 1.0f, 4.0f, 1.0f, 4.0f};
    vector<float> dataB{3.0f, 3.0f, 3.0f, 9.0f, 9.0f, 9.0f};
    copy_data(a, dataA);
    copy_data(b, dataB);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    vector<float> expected{10, 28, 37, 109};
    EXPECT_TRUE(test::all_close_f(read_vector<float>(result), expected, MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, cpu_fusion_gemm_cpu_no_bias)
{
    auto shapeA = Shape{3, 2};
    auto shapeB = Shape{2, 3};
    auto shapeC = Shape{2, 2};
    auto A = make_shared<op::v0::Parameter>(element::f32, shapeA);
    auto B = make_shared<op::v0::Parameter>(element::f32, shapeB);

    auto reshape_w = make_shared<op::v0::Reshape>(A, AxisVector{1, 0}, Shape{2, 3});
    auto reshape_x = make_shared<op::v0::Reshape>(B, AxisVector{1, 0}, Shape{3, 2});

    auto cg = make_shared<op::MatmulBias>(
        A, B, Output<Node>(), A->get_output_shape(0), B->get_output_shape(0), true, true);

    auto f = make_shared<Function>(cg, ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    shared_ptr<runtime::Tensor> a = backend->create_tensor(element::f32, shapeA);
    shared_ptr<runtime::Tensor> b = backend->create_tensor(element::f32, shapeB);
    shared_ptr<runtime::Tensor> result = backend->create_tensor(element::f32, shapeC);

    vector<float> dataA{1.0f, 4.0f, 1.0f, 4.0f, 1.0f, 4.0f};
    vector<float> dataB{3.0f, 3.0f, 3.0f, 9.0f, 9.0f, 9.0f};
    copy_data(a, dataA);
    copy_data(b, dataB);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    vector<float> expected{9, 27, 36, 108};
    EXPECT_TRUE(test::all_close_f(read_vector<float>(result), expected, MIN_FLOAT_TOLERANCE_BITS));
}

namespace
{
    struct ConvolutionBiasTestData
    {
        size_t n{0};
        size_t c{0};
        size_t filter{0};
        size_t kernel_size{0};
        size_t w{0};
        size_t h{0};
        shared_ptr<runtime::Tensor> data_val;
        shared_ptr<runtime::Tensor> weights_val;
        shared_ptr<runtime::Tensor> bias_val;
        shared_ptr<runtime::Tensor> result_val;
        shared_ptr<runtime::Tensor> delta_val;
        shared_ptr<runtime::Tensor> d_data_val;
        shared_ptr<runtime::Tensor> d_weights_val;
        shared_ptr<runtime::Tensor> d_bias_val;
        vector<float> expected_result_val;
        vector<float> expected_d_data_val;
        vector<float> expected_d_weights_val;
        vector<float> expected_d_bias_val;

        Shape data_shape;
        Shape weights_shape;
        Shape bias_shape;
        Shape result_shape;
        shared_ptr<op::v0::Parameter> data;
        shared_ptr<op::v0::Parameter> weights;
        shared_ptr<op::v0::Parameter> bias;
        shared_ptr<op::v0::Parameter> delta;

        void n1c1h3w3(runtime::Backend* backend)
        {
            n = 1;
            c = 1;
            filter = 1;
            kernel_size = 3;
            w = 3;
            h = w;

            data_shape = Shape{n, c, h, w};
            data = make_shared<op::v0::Parameter>(element::f32, data_shape);
            weights_shape = Shape{filter, c, kernel_size, kernel_size};
            weights = make_shared<op::v0::Parameter>(element::f32, weights_shape);
            bias_shape = Shape{filter};
            bias = make_shared<op::v0::Parameter>(element::f32, bias_shape);
            result_shape = Shape{n, filter, 1, 1};

            data_val = backend->create_tensor(element::f32, data_shape);
            copy_data(data_val,
                      vector<float>{-0.67765152f,
                                    0.10073948f,
                                    0.57595438f,
                                    -0.3469252f,
                                    -0.22134334f,
                                    -1.80471897f,
                                    -0.80642909f,
                                    1.22033095f,
                                    2.23235631f});
            weights_val = backend->create_tensor(element::f32, weights_shape);
            copy_data(weights_val,
                      vector<float>{0.20070229f,
                                    -0.54968649f,
                                    -0.19819015f,
                                    -0.38577855f,
                                    1.37109005f,
                                    -0.23789984f,
                                    0.14867957f,
                                    -0.49851316f,
                                    -0.84815776f});
            bias_val = backend->create_tensor(element::f32, bias_shape);
            copy_data(bias_val, vector<float>{0.07811152f});

            result_val = backend->create_tensor(element::f32, result_shape);
            copy_data(result_val, vector<float>{0});

            delta = make_shared<op::v0::Parameter>(element::f32, result_shape);
            delta_val = backend->create_tensor(element::f32, result_shape);
            copy_data(delta_val, vector<float>{-2.58936238f});

            d_data_val = backend->create_tensor(element::f32, data_shape);
            copy_data(d_data_val, vector<float>{0, 0, 0, 0, 0, 0, 0, 0, 0});

            d_weights_val = backend->create_tensor(element::f32, weights_shape);
            copy_data(d_weights_val, vector<float>{0, 0, 0, 0, 0, 0, 0, 0, 0});

            d_bias_val = backend->create_tensor(element::f32, bias_shape);
            copy_data(d_bias_val, vector<float>{0});

            expected_result_val = vector<float>{-2.58936238f};
            expected_d_data_val = vector<float>{-0.51969099f,
                                                1.42333758f,
                                                0.5131861f,
                                                0.99892044f,
                                                -3.5502491f,
                                                0.61600888f,
                                                -0.3849853f,
                                                1.29083121f,
                                                2.19618773f};
            expected_d_weights_val = vector<float>{1.7546854f,
                                                   -0.26085103f,
                                                   -1.49135458f,
                                                   0.89831507f,
                                                   0.57313812f,
                                                   4.67307138f,
                                                   2.08813715f,
                                                   -3.15987897f,
                                                   -5.7803793f};
            expected_d_bias_val = vector<float>{-2.58936238f};
        }
    };
}

NGRAPH_TEST(${BACKEND_NAME}, cpu_fusion_conv_bias_fprop_n1c1h3w3)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    ConvolutionBiasTestData conv_test;
    conv_test.n1c1h3w3(backend.get());

    auto convolution = make_shared<op::v0::Convolution>(conv_test.data, conv_test.weights);
    auto convolution_bias = make_shared<op::v0::ConvolutionBias>(convolution, conv_test.bias);

    auto f = make_shared<Function>(
        convolution_bias, ParameterVector{conv_test.data, conv_test.weights, conv_test.bias});

    auto handle = backend->compile(f);
    handle->call_with_validate({conv_test.result_val},
                               {conv_test.data_val, conv_test.weights_val, conv_test.bias_val});
    auto result_vec = read_vector<float>(conv_test.result_val);

    EXPECT_TRUE(
        test::all_close(conv_test.expected_result_val, read_vector<float>(conv_test.result_val)));
}

NGRAPH_TEST(${BACKEND_NAME}, cpu_fusion_conv_bias_bprop_n1c1h3w3)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    ConvolutionBiasTestData conv_test;
    conv_test.n1c1h3w3(backend.get());

    auto convolution = make_shared<op::v0::Convolution>(conv_test.data, conv_test.weights);
    auto convolution_bias = make_shared<op::v0::ConvolutionBias>(convolution, conv_test.bias);

    auto f = make_shared<Function>(
        convolution_bias, ParameterVector{conv_test.data, conv_test.weights, conv_test.bias});

    ngraph::autodiff::Adjoints adjoints(OutputVector{convolution_bias},
                                        OutputVector{conv_test.delta});

    auto d_data = adjoints.backprop_output(conv_test.data);
    auto d_weights = adjoints.backprop_output(conv_test.weights);
    auto d_bias = adjoints.backprop_output(conv_test.bias);

    auto df = make_shared<Function>(
        OutputVector{d_data, d_weights, d_bias},
        ParameterVector{conv_test.data, conv_test.weights, conv_test.bias, conv_test.delta});
    auto handle = backend->compile(df);
    handle->call_with_validate(

        {conv_test.d_data_val, conv_test.d_weights_val, conv_test.d_bias_val},
        {conv_test.data_val, conv_test.weights_val, conv_test.bias_val, conv_test.delta_val});

    EXPECT_TRUE(
        test::all_close(conv_test.expected_d_data_val, read_vector<float>(conv_test.d_data_val)));
    EXPECT_TRUE(test::all_close(conv_test.expected_d_weights_val,
                                read_vector<float>(conv_test.d_weights_val)));
    EXPECT_TRUE(
        test::all_close(conv_test.expected_d_bias_val, read_vector<float>(conv_test.d_bias_val)));
}

namespace
{
    static void test_batchnorm_multiply_add_relu(const string& backend_name, Shape input_shape)
    {
        auto make_bn_relu_function = [&]() {
            auto c_axis = input_shape[1];
            auto input = make_shared<op::v0::Parameter>(element::f32, input_shape);
            auto mean_shape = Shape{c_axis};
            auto mean = std::make_shared<op::v0::Parameter>(element::f32, mean_shape);
            auto var_shape = Shape{c_axis};
            auto var = std::make_shared<op::v0::Parameter>(element::f32, var_shape);
            auto gamma_shape = Shape{c_axis};
            auto gamma = make_shared<op::v0::Parameter>(element::f32, gamma_shape);
            auto beta_shape = Shape{c_axis};
            auto beta = make_shared<op::v0::Parameter>(element::f32, beta_shape);
            double eps = 0.001;
            auto bn = std::make_shared<ngraph::op::v0::BatchNormInference>(
                eps, gamma, beta, input, mean, var);

            std::vector<size_t> vec{0};
            for (size_t i = 2; i < input_shape.size(); i++)
            {
                vec.push_back(i);
            }
            auto broadcast1_input = std::make_shared<op::v0::Parameter>(element::f32, gamma_shape);
            auto broadcast1 = std::make_shared<ngraph::op::v0::Broadcast>(
                broadcast1_input, input_shape, AxisSet(vec));
            auto multiply = std::make_shared<ngraph::op::v1::Multiply>(bn, broadcast1);

            auto broadcast2_input = std::make_shared<op::v0::Parameter>(element::f32, gamma_shape);
            auto broadcast2 = std::make_shared<ngraph::op::v0::Broadcast>(
                broadcast2_input, input_shape, AxisSet(vec));

            auto add = std::make_shared<ngraph::op::v1::Add>(multiply, broadcast2);
            auto relu = std::make_shared<ngraph::op::v0::Relu>(add);
            auto f = make_shared<Function>(
                relu,
                ParameterVector{gamma, beta, input, mean, var, broadcast1_input, broadcast2_input});
            return f;
        };

        auto cpu_f = make_bn_relu_function();
        auto int_f = make_bn_relu_function();
        test::Uniform<float> rng(1.0f, 10.0f);
        vector<vector<float>> args;

        for (shared_ptr<op::v0::Parameter> param : int_f->get_parameters())
        {
            vector<float> tensor_val(shape_size(param->get_output_shape(0)));
            rng.initialize(tensor_val);
            args.push_back(tensor_val);
        }
        auto int_results = execute(int_f, args, "INTERPRETER");
        auto cpu_results = execute(cpu_f, args, backend_name);
        for (size_t i = 0; i < cpu_results.size(); i++)
        {
            EXPECT_TRUE(test::all_close(cpu_results.at(i), int_results.at(i), 1.0e-4f, 1.0e-4f));
        }

        size_t bn_relu = count_ops_of_type<op::BatchNormInferenceRelu>(cpu_f);
        ASSERT_EQ(bn_relu, 1);
    }
}

NGRAPH_TEST(${BACKEND_NAME}, cpu_fusion_batchnorm_multiply_add_relu)
{
    test_batchnorm_multiply_add_relu("${BACKEND_NAME}", Shape{1, 3, 2, 2});
    test_batchnorm_multiply_add_relu("${BACKEND_NAME}", Shape{1, 2, 2, 2, 2});
    test_batchnorm_multiply_add_relu("${BACKEND_NAME}", Shape{2, 2, 2, 4, 4});
}

NGRAPH_TEST(${BACKEND_NAME}, cpu_fusion_batchnorm_multiply_add_relu_no_fusion)
{
    auto input_shape = Shape{3, 3, 2, 2};
    auto make_bn_relu_function = [&]() {
        auto c_axis = input_shape[1];
        auto input = make_shared<op::v0::Parameter>(element::f32, input_shape);
        auto mean_shape = Shape{c_axis};
        auto mean = std::make_shared<op::v0::Parameter>(element::f32, mean_shape);
        auto var_shape = Shape{c_axis};
        auto var = std::make_shared<op::v0::Parameter>(element::f32, var_shape);
        auto gamma_shape = Shape{c_axis};
        auto gamma = make_shared<op::v0::Parameter>(element::f32, gamma_shape);
        auto beta_shape = Shape{c_axis};
        auto beta = make_shared<op::v0::Parameter>(element::f32, beta_shape);
        double eps = 0.001;
        auto bn = std::make_shared<ngraph::op::v0::BatchNormInference>(
            eps, gamma, beta, input, mean, var);

        std::vector<size_t> vec;
        for (size_t i = 1; i < input_shape.size(); i++)
        {
            vec.push_back(i);
        }
        auto broadcast1_input = std::make_shared<op::v0::Parameter>(element::f32, Shape{3});
        auto broadcast1 = std::make_shared<ngraph::op::v0::Broadcast>(
            broadcast1_input, input_shape, AxisSet(vec));
        auto multiply = std::make_shared<ngraph::op::v1::Multiply>(bn, broadcast1);

        auto broadcast2_input = std::make_shared<op::v0::Parameter>(element::f32, Shape{3});
        auto broadcast2 = std::make_shared<ngraph::op::v0::Broadcast>(
            broadcast2_input, input_shape, AxisSet(vec));

        auto add = std::make_shared<ngraph::op::v1::Add>(multiply, broadcast2);
        auto relu = std::make_shared<ngraph::op::v0::Relu>(add);
        auto f = make_shared<Function>(
            relu,
            ParameterVector{gamma, beta, input, mean, var, broadcast1_input, broadcast2_input});
        return f;
    };

    auto cpu_f = make_bn_relu_function();
    auto int_f = make_bn_relu_function();
    test::Uniform<float> rng(1.0f, 10.0f);
    vector<vector<float>> args;

    for (shared_ptr<op::v0::Parameter> param : int_f->get_parameters())
    {
        vector<float> tensor_val(shape_size(param->get_output_shape(0)));
        rng.initialize(tensor_val);
        args.push_back(tensor_val);
    }
    auto int_results = execute(int_f, args, "INTERPRETER");
    auto cpu_results = execute(cpu_f, args, "${BACKEND_NAME}");
    for (size_t i = 0; i < cpu_results.size(); i++)
    {
        EXPECT_TRUE(test::all_close(cpu_results.at(i), int_results.at(i), 1.0e-4f, 1.0e-4f));
    }

    size_t bn_relu = count_ops_of_type<op::BatchNormInferenceRelu>(cpu_f);
    ASSERT_EQ(bn_relu, 0);
}

NGRAPH_TEST(${BACKEND_NAME}, cpu_fusion_batchnorm_fprop_relu_b1c2h2w2)
{
    auto input_shape = Shape{1, 2, 2, 2};
    auto input = make_shared<op::v0::Parameter>(element::f32, input_shape);
    auto mean_shape = Shape{2};
    auto var_shape = Shape{2};
    auto gamma_shape = Shape{2};
    auto gamma = make_shared<op::v0::Parameter>(element::f32, gamma_shape);
    auto beta_shape = Shape{2};
    auto beta = make_shared<op::v0::Parameter>(element::f32, beta_shape);
    double eps = 0.001;
    auto shape_r = Shape{1, 2, 2, 2};
    auto bn = make_shared<op::v0::BatchNormTraining>(input, gamma, beta, eps);

    auto output_rt = bn->output(0);
    // Note, op::v0::Splice is used to break Relu(BatchNorm) fusion
    // otherwise we will be comparing two BatchNormRelus
    // Unfortunately, we can't use INTERPRETER for
    // verifying the results as it doesn't implement
    // BatchNorm op.
    auto slice =
        std::make_shared<op::v0::Slice>(output_rt, Coordinate{0, 0, 0, 0}, Coordinate{1, 2, 2, 2});
    auto output_relu = std::make_shared<op::v0::Relu>(slice);
    auto mean_rt = bn->output(1);
    auto variance_rt = bn->output(2);

    auto bn_relu = make_shared<op::BatchNormTrainingRelu>(eps, gamma, beta, input);
    auto output_rt_bnr = bn_relu->output(0);
    auto mean_rt_bnr = bn_relu->output(1);
    auto variance_rt_bnr = bn_relu->output(2);

    auto f = make_shared<Function>(
        OutputVector{
            output_relu, mean_rt, variance_rt, output_rt_bnr, mean_rt_bnr, variance_rt_bnr},
        ParameterVector{input, gamma, beta});
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto input_t = backend->create_tensor(element::f32, Shape{1, 2, 2, 2});

    copy_data(input_t,
              vector<float>{0.54881352f,
                            0.71518934f,
                            0.60276335f,
                            0.54488319f,
                            0.42365479f,
                            0.64589411f,
                            0.4375872f,
                            0.89177299f});
    auto gamma_t = backend->create_tensor(element::f32, gamma_shape);
    copy_data(gamma_t, vector<float>{1.0f, 1.0f});
    auto beta_t = backend->create_tensor(element::f32, beta_shape);
    copy_data(beta_t, vector<float>{0.0f, 0.0f});
    auto bn_output = backend->create_tensor(element::f32, shape_r);
    auto result_mean = backend->create_tensor(element::f32, mean_shape);
    auto result_variance = backend->create_tensor(element::f32, var_shape);

    auto bn_output_bnr = backend->create_tensor(element::f32, shape_r);
    auto result_mean_bnr = backend->create_tensor(element::f32, mean_shape);
    auto result_variance_bnr = backend->create_tensor(element::f32, var_shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({bn_output,
                                result_mean,
                                result_variance,
                                bn_output_bnr,
                                result_mean_bnr,
                                result_variance_bnr},
                               {input_t, gamma_t, beta_t});

    EXPECT_TRUE(test::all_close(read_vector<float>(bn_output), read_vector<float>(bn_output_bnr)));
    EXPECT_TRUE(
        test::all_close(read_vector<float>(result_mean), read_vector<float>(result_mean_bnr)));
    EXPECT_TRUE(test::all_close(read_vector<float>(result_variance),
                                read_vector<float>(result_variance_bnr)));
}

namespace
{
    static void test_batchnorm_fprop_relu(const string& backend_name, Shape input_shape)
    {
        auto make_bn_relu_function = [&]() {
            auto c_axis = input_shape[1];
            auto input = make_shared<op::v0::Parameter>(element::f32, input_shape);
            auto mean_shape = Shape{c_axis};
            auto var_shape = Shape{c_axis};
            auto gamma_shape = Shape{c_axis};
            auto gamma = make_shared<op::v0::Parameter>(element::f32, gamma_shape);
            auto beta_shape = Shape{c_axis};
            auto beta = make_shared<op::v0::Parameter>(element::f32, beta_shape);
            double eps = 0.001;
            auto shape_r = input_shape;
            auto bn = make_shared<op::v0::BatchNormTraining>(eps, gamma, beta, input);
            auto output_rt = bn->output(0);

            auto output_relu = std::make_shared<op::v0::Relu>(output_rt);
            auto mean_rt = bn->output(1);
            auto variance_rt = bn->output(2);

            auto f = make_shared<Function>(OutputVector{output_relu, mean_rt, variance_rt},
                                           ParameterVector{input, gamma, beta});
            return f;
        };
        auto cpu_f = make_bn_relu_function();
        auto int_f = make_bn_relu_function();
        test::Uniform<float> rng(-10.0f, 10.0f);
        vector<vector<float>> args;

        for (shared_ptr<op::v0::Parameter> param : int_f->get_parameters())
        {
            vector<float> tensor_val(shape_size(param->get_output_shape(0)));
            rng.initialize(tensor_val);
            args.push_back(tensor_val);
        }
        auto int_results = execute(int_f, args, "INTERPRETER");
        auto cpu_results = execute(cpu_f, args, backend_name);
        for (size_t i = 0; i < cpu_results.size(); i++)
        {
            EXPECT_TRUE(test::all_close(cpu_results.at(i), int_results.at(i), 1.0e-4f, 1.0e-4f));
        }
    }
}

NGRAPH_TEST(${BACKEND_NAME}, cpu_fusion_conv_relu_n2c1h2w2_2)
{
    Shape shape_a{2, 1, 6, 6};
    Shape shape_weights{1, 1, 2, 2};

    auto make_int_function = [shape_a, shape_weights]() {
        auto A = std::make_shared<op::v0::Parameter>(element::f32, shape_a);
        auto weights = std::make_shared<op::v0::Parameter>(element::f32, shape_weights);
        auto conv = std::make_shared<op::v0::Convolution>(A, weights, Strides{2, 2}, Strides{1, 1});
        auto relu = std::make_shared<op::v0::Relu>(conv);
        auto f = make_shared<Function>(OutputVector{relu}, ParameterVector{A, weights});
        return f;
    };

    auto int_f = make_int_function();

    auto make_cpu_function = [shape_a, shape_weights]() {
        auto A = std::make_shared<op::v0::Parameter>(element::f32, shape_a);
        auto weights = std::make_shared<op::v0::Parameter>(element::f32, shape_weights);
        auto conv = std::make_shared<op::v0::Convolution>(A, weights, Strides{2, 2}, Strides{1, 1});
        auto conv_relu = std::make_shared<op::ConvolutionRelu>(conv);
        auto f = make_shared<Function>(OutputVector{conv_relu}, ParameterVector{A, weights});
        return f;
    };

    auto cpu_f = make_cpu_function();

    vector<vector<float>> args{
        {1.25f,  2.25f, 5.25f, 6.25f,  -1.25f, -1.25f, 3.25f, -4.25f, 7.25f,  8.25f,  -1.25f,
         -1.25f, 1.25f, 2.25f, -3.25f, 2.25f,  4.25f,  4.25f, 1.25f,  2.25f,  -4.25f, 2.25f,
         4.25f,  4.25f, 0.f,   0.f,    -1.f,   0.f,    2.f,   2.f,    0.f,    0.f,    0.f,
         0.f,    2.f,   2.f,   1.25f,  2.25f,  5.25f,  6.25f, 1.25f,  1.25f,  3.25f,  4.25f,
         -7.25f, 8.25f, 1.25f, -1.25f, -1.25f, 2.25f,  3.25f, 2.25f,  -4.25f, -4.25f, -1.25f,
         -2.25f, 4.25f, 2.25f, 4.25f,  4.25f,  0.f,    0.f,   1.f,    0.f,    -2.f,   2.f,
         0.f,    0.f,   0.f,   0.f,    -2.f,   -2.f},
        {2., 2., 2., 2.}};

    auto int_results = execute(int_f, args, "INTERPRETER");
    auto cpu_results = execute(cpu_f, args, "${BACKEND_NAME}");
    EXPECT_TRUE(test::all_close(cpu_results.at(0), int_results.at(0)));
}

NGRAPH_TEST(${BACKEND_NAME}, cpu_fusion_conv_bias_relu_n2c1h2w2_2)
{
    Shape shape_a{2, 1, 6, 6};
    Shape shape_weights{1, 1, 2, 2};
    Shape shape_bias{1};

    auto make_int_function = [shape_a, shape_weights, shape_bias]() {
        auto A = std::make_shared<op::v0::Parameter>(element::f32, shape_a);
        auto weights = std::make_shared<op::v0::Parameter>(element::f32, shape_weights);
        auto conv = std::make_shared<op::v0::Convolution>(A, weights, Strides{2, 2}, Strides{1, 1});
        auto bias = std::make_shared<op::v0::Parameter>(element::f32, shape_bias);
        auto conv_bias = conv + std::make_shared<op::v0::Broadcast>(
                                    bias, conv->get_output_shape(0), AxisSet{0, 2, 3});
        auto relu = std::make_shared<op::v0::Relu>(conv_bias);
        auto f = make_shared<Function>(OutputVector{relu}, ParameterVector{A, weights, bias});
        return f;
    };

    auto int_f = make_int_function();

    auto make_cpu_function = [shape_a, shape_weights, shape_bias]() {
        auto A = std::make_shared<op::v0::Parameter>(element::f32, shape_a);
        auto weights = std::make_shared<op::v0::Parameter>(element::f32, shape_weights);
        auto bias = std::make_shared<op::v0::Parameter>(element::f32, shape_bias);
        auto conv = std::make_shared<op::v0::Convolution>(A, weights, Strides{2, 2}, Strides{1, 1});
        auto conv_bias_relu = std::make_shared<op::v0::ConvolutionBias>(conv, bias, true);
        auto f =
            make_shared<Function>(OutputVector{conv_bias_relu}, ParameterVector{A, weights, bias});
        return f;
    };

    auto cpu_f = make_cpu_function();

    vector<vector<float>> args{
        {1.25f,  2.25f, 5.25f, 6.25f,  -1.25f, -1.25f, 3.25f, -4.25f, 7.25f,  8.25f,  -1.25f,
         -1.25f, 1.25f, 2.25f, -3.25f, 2.25f,  4.25f,  4.25f, 1.25f,  2.25f,  -4.25f, 2.25f,
         4.25f,  4.25f, 0.f,   0.f,    -1.f,   0.f,    2.f,   2.f,    0.f,    0.f,    0.f,
         0.f,    2.f,   2.f,   1.25f,  2.25f,  5.25f,  6.25f, 1.25f,  1.25f,  3.25f,  4.25f,
         -7.25f, 8.25f, 1.25f, -1.25f, -1.25f, 2.25f,  3.25f, 2.25f,  -4.25f, -4.25f, -1.25f,
         -2.25f, 4.25f, 2.25f, 4.25f,  4.25f,  0.f,    0.f,   1.f,    0.f,    -2.f,   2.f,
         0.f,    0.f,   0.f,   0.f,    -2.f,   -2.f},
        {2., 2., 2., 2.},
        {0.1f}};

    auto int_results = execute(int_f, args, "INTERPRETER");
    auto cpu_results = execute(cpu_f, args, "${BACKEND_NAME}");
    EXPECT_TRUE(test::all_close(cpu_results.at(0), int_results.at(0)));
}

NGRAPH_TEST(${BACKEND_NAME}, cpu_fusion_conv_horizontal_fusion)
{
    Shape shape_a{2, 1, 6, 6};
    Shape shape_weights{1, 1, 2, 2};
    Shape shape_bias{1};

    auto make_function = [shape_a, shape_weights, shape_bias]() {
        auto A = std::make_shared<op::v0::Parameter>(element::f32, shape_a);
        auto weights1 = std::make_shared<op::v0::Parameter>(element::f32, shape_weights);
        auto conv1 =
            std::make_shared<op::v0::Convolution>(A, weights1, Strides{2, 2}, Strides{1, 1});
        auto bias1 = std::make_shared<op::v0::Parameter>(element::f32, shape_bias);
        auto conv_bias1 = conv1 + std::make_shared<op::v0::Broadcast>(
                                      bias1, conv1->get_output_shape(0), AxisSet{0, 2, 3});
        auto relu1 = std::make_shared<op::v0::Relu>(conv_bias1);

        auto weights2 = std::make_shared<op::v0::Parameter>(element::f32, shape_weights);
        auto conv2 =
            std::make_shared<op::v0::Convolution>(A, weights2, Strides{2, 2}, Strides{1, 1});
        auto bias2 = std::make_shared<op::v0::Parameter>(element::f32, shape_bias);
        auto conv_bias2 = conv2 + std::make_shared<op::v0::Broadcast>(
                                      bias2, conv2->get_output_shape(0), AxisSet{0, 2, 3});
        auto relu2 = std::make_shared<op::v0::Relu>(conv_bias2);

        auto concat = std::make_shared<op::v0::Concat>(OutputVector{relu1, relu2}, 1);
        auto f = make_shared<Function>(OutputVector{concat},
                                       ParameterVector{A, weights1, bias1, weights2, bias2});
        return f;
    };
    auto int_f = make_function();
    auto cpu_f = make_function();

    vector<vector<float>> args{
        {1.25f,  2.25f, 5.25f, 6.25f,  -1.25f, -1.25f, 3.25f, -4.25f, 7.25f,  8.25f,  -1.25f,
         -1.25f, 1.25f, 2.25f, -3.25f, 2.25f,  4.25f,  4.25f, 1.25f,  2.25f,  -4.25f, 2.25f,
         4.25f,  4.25f, 0.f,   0.f,    -1.f,   0.f,    2.f,   2.f,    0.f,    0.f,    0.f,
         0.f,    2.f,   2.f,   1.25f,  2.25f,  5.25f,  6.25f, 1.25f,  1.25f,  3.25f,  4.25f,
         -7.25f, 8.25f, 1.25f, -1.25f, -1.25f, 2.25f,  3.25f, 2.25f,  -4.25f, -4.25f, -1.25f,
         -2.25f, 4.25f, 2.25f, 4.25f,  4.25f,  0.f,    0.f,   1.f,    0.f,    -2.f,   2.f,
         0.f,    0.f,   0.f,   0.f,    -2.f,   -2.f},
        {2., 2., 2., 2.},
        {0.1f},
        {3., 3., 3., 3.},
        {0.2f}};

    auto int_results = execute(int_f, args, "INTERPRETER");
    auto cpu_results = execute(cpu_f, args, "${BACKEND_NAME}");
    EXPECT_TRUE(test::all_close(cpu_results.at(0), int_results.at(0)));

    size_t cpu_ck = count_ops_of_type<op::v0::CompiledKernel>(cpu_f);
    if (!cpu_ck)
    {
        size_t cpu_cb = count_ops_of_type<op::v0::ConvolutionBias>(cpu_f);
        ASSERT_EQ(cpu_cb, 1);
    }
}

namespace
{
    // ConvolutionBiasAdd relies on an in-place fused DNNL kernel.
    // Need to ensure that it is fused only when in-place buffer allocation is feasible
    shared_ptr<Function> gen_conv_bias_add(bool param_input, bool result_output)
    {
        auto A = make_shared<op::v0::Parameter>(element::f32, Shape{2, 1, 2, 2});
        auto weights = make_shared<op::v0::Parameter>(element::f32, Shape{1, 1, 1, 1});
        auto bias = make_shared<op::v0::Parameter>(element::f32, Shape{1});
        auto conv = make_shared<op::v0::Convolution>(A, weights, Strides{1, 1}, Strides{1, 1});
        auto bias_broadcast =
            make_shared<op::v0::Broadcast>(bias, conv->get_output_shape(0), AxisSet{0, 2, 3});
        auto convbias = conv + bias_broadcast;
        auto B = make_shared<op::v0::Parameter>(element::f32, Shape{2, 1, 2, 2});
        auto abs_B = make_shared<op::v0::Abs>(B);
        auto add = param_input ? make_shared<op::v1::Add>(convbias, B)
                               : make_shared<op::v1::Add>(convbias, abs_B);
        auto abs = make_shared<op::v0::Abs>(add);

        return result_output ? make_shared<Function>(add, ParameterVector{A, weights, bias, B})
                             : make_shared<Function>(abs, ParameterVector{A, weights, bias, B});
    }
}

NGRAPH_TEST(${BACKEND_NAME}, cpu_fusion_conv_bias_add)
{
    auto int_f = gen_conv_bias_add(false, false);
    auto cpu_f = gen_conv_bias_add(false, false);

    vector<vector<float>> args{{1.25f, 2.25f, 5.25f, 6.25f, -1.25f, -1.25f, 3.25f, -4.25f},
                               {-1.25f},
                               {2.25f},
                               {1.25f, 2.25f, -3.25f, 2.25f, 4.25f, 4.25f, 1.25f, 2.25f}};

    auto int_results = execute(int_f, args, "INTERPRETER");
    auto cpu_results = execute(cpu_f, args, "${BACKEND_NAME}");
    EXPECT_TRUE(test::all_close(cpu_results.at(0), int_results.at(0)));
}

namespace
{
    // ConvolutionAdd relies on an in-place fused DNNL kernel.
    // Need to ensure that it is fused only when in-place buffer allocation is feasible
    shared_ptr<Function> gen_conv_add(bool param_input, bool result_output)
    {
        auto A = make_shared<op::v0::Parameter>(element::f32, Shape{2, 1, 2, 2});
        auto weights = make_shared<op::v0::Parameter>(element::f32, Shape{1, 1, 1, 1});
        auto conv = make_shared<op::v0::Convolution>(A, weights, Strides{1, 1}, Strides{1, 1});
        auto B = make_shared<op::v0::Parameter>(element::f32, Shape{2, 1, 2, 2});
        auto abs_B = make_shared<op::v0::Abs>(B);
        auto add =
            param_input ? make_shared<op::v1::Add>(conv, B) : make_shared<op::v1::Add>(conv, abs_B);
        auto abs = make_shared<op::v0::Abs>(add);

        return result_output ? make_shared<Function>(add, ParameterVector{A, weights, B})
                             : make_shared<Function>(abs, ParameterVector{A, weights, B});
    }
}

NGRAPH_TEST(${BACKEND_NAME}, cpu_fusion_conv_add)
{
    auto int_f = gen_conv_add(false, false);
    auto cpu_f = gen_conv_add(false, false);

    vector<vector<float>> args{{1.25f, 2.25f, 5.25f, 6.25f, -1.25f, -1.25f, 3.25f, -4.25f},
                               {-1.25f},
                               {1.25f, 2.25f, -3.25f, 2.25f, 4.25f, 4.25f, 1.25f, 2.25f}};

    auto int_results = execute(int_f, args, "INTERPRETER");
    auto cpu_results = execute(cpu_f, args, "${BACKEND_NAME}");
    EXPECT_TRUE(test::all_close(cpu_results.at(0), int_results.at(0)));

    int_f = gen_conv_add(false, true);
    cpu_f = gen_conv_add(false, true);

    int_results = execute(int_f, args, "INTERPRETER");
    cpu_results = execute(cpu_f, args, "${BACKEND_NAME}");
    EXPECT_TRUE(test::all_close(cpu_results.at(0), int_results.at(0)));
}

namespace
{
    shared_ptr<Function> gen_groupconv_batchnorm(const bool add_goe,
                                                 const bool with_relu,
                                                 const Shape shape_in,
                                                 const Shape shape_weights,
                                                 const Shape shape_out,
                                                 const size_t groups)
    {
        auto input = make_shared<op::v0::Parameter>(element::f32, shape_in);
        auto weights = make_shared<op::v0::Parameter>(element::f32, shape_weights);

        unsigned long OC = shape_out.at(1);
        Shape shape_bn{OC};
        auto group_conv = make_shared<op::v0::GroupConvolution>(input,
                                                                weights,
                                                                Strides{1, 1},
                                                                Strides{1, 1},
                                                                CoordinateDiff{0, 0},
                                                                CoordinateDiff{0, 0},
                                                                Strides{1, 1},
                                                                groups);

        double eps = 0.001;
        auto gamma = std::make_shared<op::v0::Parameter>(element::f32, shape_bn);
        auto beta = std::make_shared<op::v0::Parameter>(element::f32, shape_bn);
        auto mean = std::make_shared<op::v0::Parameter>(element::f32, shape_bn);
        auto var = std::make_shared<op::v0::Parameter>(element::f32, shape_bn);

        auto goe_bn = group_conv->output(0);

        // Adding a goe will stop fusion since the patterns wont expect to see this op
        auto bn =
            add_goe
                ? std::make_shared<op::v0::BatchNormInference>(goe_bn, gamma, beta, mean, var, eps)
                : std::make_shared<op::v0::BatchNormInference>(
                      group_conv, gamma, beta, mean, var, eps);
        if (with_relu)
        {
            auto prelu = std::make_shared<op::v0::Relu>(bn);
            auto f = make_shared<Function>(OutputVector{prelu},
                                           ParameterVector{input, weights, gamma, beta, mean, var});
            return f;
        }
        else
        {
            auto f = make_shared<Function>(OutputVector{bn},
                                           ParameterVector{input, weights, gamma, beta, mean, var});
            return f;
        }
    }

    void fuse_groupconv_batchnorm_helper(Shape shape_in,
                                         Shape shape_weights,
                                         Shape shape_r,
                                         size_t groups)
    {
        auto func_fuse =
            gen_groupconv_batchnorm(false, false, shape_in, shape_weights, shape_r, groups);
        auto func_fuse2 =
            gen_groupconv_batchnorm(false, true, shape_in, shape_weights, shape_r, groups);

        {
            pass::Manager pass_manager;
            pass_manager.register_pass<runtime::cpu::pass::CPUFusion>();
            pass_manager.run_passes(func_fuse);
            ASSERT_EQ(count_ops_of_type<op::GroupConvolutionBias>(func_fuse), 1);
        }

        {
            // test groupconv + batchnorm + relu fusion
            pass::Manager pass_manager;
            pass_manager.register_pass<runtime::cpu::pass::CPUFusion>();
            pass_manager.run_passes(func_fuse2);
            ASSERT_EQ(count_ops_of_type<op::GroupConvolutionBias>(func_fuse2), 1);
            ASSERT_EQ(count_ops_of_type<op::v0::Relu>(func_fuse2), 0);
        }
    }

    void groupconv_batchnorm_test_val_helper(const string& backend_name,
                                             const bool with_relu,
                                             Shape shape_in,
                                             Shape shape_weights,
                                             Shape shape_r,
                                             size_t groups)
    {
        shared_ptr<Function> fuse_func =
            gen_groupconv_batchnorm(false, with_relu, shape_in, shape_weights, shape_r, groups);
        shared_ptr<Function> nofuse_func =
            gen_groupconv_batchnorm(true, with_relu, shape_in, shape_weights, shape_r, groups);

        test::Uniform<float> rng(1.0f, 100.0f);
        vector<vector<float>> args;
        for (shared_ptr<op::v0::Parameter> param : fuse_func->get_parameters())
        {
            vector<float> tensor_val(shape_size(param->get_output_shape(0)));
            rng.initialize(tensor_val);
            args.push_back(tensor_val);
        }

        auto fuse_results = execute(fuse_func, args, "${BACKEND_NAME}");
        auto nofuse_results = execute(nofuse_func, args, "${BACKEND_NAME}");

        EXPECT_TRUE(test::all_close(fuse_results.at(0), nofuse_results.at(0)));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, cpu_fusion_fuse_groupconv_batchnorm1)
{
    Shape shape_in{1, 20, 5, 5};
    Shape shape_weights{8, 10, 3, 3};
    Shape shape_r{1, 8, 3, 3};
    fuse_groupconv_batchnorm_helper(shape_in, shape_weights, shape_r, 2);
    groupconv_batchnorm_test_val_helper(
        "${BACKEND_NAME}", false, shape_in, shape_weights, shape_r, 2);
    groupconv_batchnorm_test_val_helper(
        "${BACKEND_NAME}", true, shape_in, shape_weights, shape_r, 2);
}

NGRAPH_TEST(${BACKEND_NAME}, cpu_fusion_fuse_groupconv_batchnorm2)
{
    Shape shape_in{1, 20, 5, 5};
    Shape shape_weights{5, 4, 3, 3};
    Shape shape_r{1, 5, 3, 3};
    fuse_groupconv_batchnorm_helper(shape_in, shape_weights, shape_r, 5);
    groupconv_batchnorm_test_val_helper(
        "${BACKEND_NAME}", false, shape_in, shape_weights, shape_r, 5);
    groupconv_batchnorm_test_val_helper(
        "${BACKEND_NAME}", true, shape_in, shape_weights, shape_r, 5);
}

NGRAPH_TEST(${BACKEND_NAME}, cpu_fusion_fuse_groupconv_batchnorm3)
{
    Shape shape_in{1, 20, 5, 5};
    Shape shape_weights{20, 1, 3, 3};
    Shape shape_r{1, 20, 3, 3};
    fuse_groupconv_batchnorm_helper(shape_in, shape_weights, shape_r, 20);
    groupconv_batchnorm_test_val_helper(
        "${BACKEND_NAME}", false, shape_in, shape_weights, shape_r, 20);
    groupconv_batchnorm_test_val_helper(
        "${BACKEND_NAME}", true, shape_in, shape_weights, shape_r, 20);
}

NGRAPH_TEST(${BACKEND_NAME}, cpu_fusion_fuse_groupconv_batchnorm4)
{
    Shape shape_in{1, 20, 4, 4};
    Shape shape_weights{5, 20, 1, 1};
    Shape shape_r{1, 5, 4, 4};
    fuse_groupconv_batchnorm_helper(shape_in, shape_weights, shape_r, 1);
    groupconv_batchnorm_test_val_helper(
        "${BACKEND_NAME}", false, shape_in, shape_weights, shape_r, 1);
    groupconv_batchnorm_test_val_helper(
        "${BACKEND_NAME}", true, shape_in, shape_weights, shape_r, 1);
}

namespace
{
    std::vector<shared_ptr<runtime::Tensor>>
        rnn_matrix_fusion_eval(const string& backend_name,
                               const size_t time_steps,
                               const Shape& data_shape,
                               const Shape& weights_shape,
                               const Shape& bias_shape,
                               const vector<float>& data_val,
                               const vector<float>& weights_val,
                               const vector<float>& bias_val,
                               const bool enable_pass)
    {
        auto data = make_shared<op::v0::Parameter>(element::f32, data_shape);
        auto weights = make_shared<op::v0::Parameter>(element::f32, weights_shape);
        auto bias = make_shared<op::v0::Parameter>(element::f32, bias_shape);

        // results from each time step
        OutputVector results;
        for (size_t t = 0; t < time_steps; ++t)
        {
            auto data_slice = make_shared<op::v0::Slice>(
                data, Coordinate{0, t, 0}, Coordinate{data_shape[0], t + 1, data_shape[2]});
            auto data_reshape = make_shared<op::v0::Reshape>(
                data_slice, AxisVector{0, 1, 2}, Shape{data_shape[0], data_shape[2]});
            auto weights_reshape = make_shared<op::v0::Reshape>(
                weights, AxisVector{1, 0}, Shape{weights_shape[1], weights_shape[0]});
            auto dot = make_shared<op::v0::Dot>(data_reshape, weights_reshape);
            auto bias_broadcast =
                make_shared<op::v0::Broadcast>(bias, dot->get_output_shape(0), AxisSet{0});
            auto add = make_shared<op::v1::Add>(dot, bias_broadcast);
            results.push_back(add);
        }
        auto func = make_shared<Function>(results, ParameterVector{data, weights, bias});
        if (enable_pass)
        {
            pass::Manager pass_manager;
            pass_manager.register_pass<runtime::cpu::pass::CPURnnMatFusion>();
            pass_manager.register_pass<runtime::cpu::pass::CPUFusion>(
                pass::FusionType::REGULAR_FUSIONS);
            pass_manager.run_passes(func);
            // check all of our dot/add are converted to a single MatmulBias op.
            size_t count = count_ops_of_type<op::MatmulBias>(func);
            EXPECT_EQ(count, 1);
        }

        auto backend = runtime::Backend::create("${BACKEND_NAME}");

        shared_ptr<runtime::Tensor> data_tensor =
            backend->create_tensor(element::f32, data->get_output_shape(0));
        shared_ptr<runtime::Tensor> weights_tensor =
            backend->create_tensor(element::f32, weights->get_output_shape(0));
        shared_ptr<runtime::Tensor> bias_tensor =
            backend->create_tensor(element::f32, bias->get_output_shape(0));

        std::vector<shared_ptr<runtime::Tensor>> result_tensors;
        for (auto r : results)
        {
            result_tensors.push_back(backend->create_tensor(element::f32, r.get_shape()));
        }

        copy_data(data_tensor, data_val);
        copy_data(weights_tensor, weights_val);
        copy_data(bias_tensor, bias_val);
        auto handle = backend->compile(func);
        handle->call_with_validate(result_tensors, {data_tensor, weights_tensor, bias_tensor});
        return result_tensors;
    }
}

NGRAPH_TEST(${BACKEND_NAME}, cpu_fusion_rnn_matrix_fusion_eval_pass)
{
    const size_t time_steps = 4;
    Shape data_shape{3, time_steps, 5};
    Shape weights_shape{6, data_shape[2]};
    Shape bias_shape{6};

    test::Uniform<float> rng{0, 1, 0};
    vector<float> data_val(shape_size(data_shape));
    vector<float> weights_val(shape_size(weights_shape));
    vector<float> bias_val(shape_size(bias_shape));
    rng.initialize(data_val);
    rng.initialize(weights_val);
    rng.initialize(bias_val);

    std::vector<shared_ptr<runtime::Tensor>> result_expected =
        rnn_matrix_fusion_eval("${BACKEND_NAME}",
                               time_steps,
                               data_shape,
                               weights_shape,
                               bias_shape,
                               data_val,
                               weights_val,
                               bias_val,
                               false);
    std::vector<shared_ptr<runtime::Tensor>> result_fused =
        rnn_matrix_fusion_eval("${BACKEND_NAME}",
                               time_steps,
                               data_shape,
                               weights_shape,
                               bias_shape,
                               data_val,
                               weights_val,
                               bias_val,
                               true);
    for (size_t i = 0; i < result_expected.size(); ++i)
    {
        EXPECT_TRUE(test::all_close<float>(result_expected[i], result_fused[i]));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, cpu_fusion_backwards_maxpool_with_indices_n4_c1_hw4_2x2_max)
{
    Shape shape_a{1, 4, 4, 4};
    Shape maxpool_shape{1, 4, 3, 3};
    auto A = std::make_shared<op::v0::Parameter>(element::f32, shape_a);
    Shape window_shape{2, 2};
    auto window_movement_strides = Strides{1, 1};
    auto maxpool = std::make_shared<op::v0::MaxPool>(A, window_shape, window_movement_strides);
    auto f = std::make_shared<Function>(maxpool, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    shared_ptr<runtime::Tensor> ep = backend->create_tensor(element::f32, maxpool_shape);
    vector<float> dataEp(shape_size(maxpool_shape), 4);

    shared_ptr<runtime::Tensor> input = backend->create_tensor(element::f32, shape_a);
    shared_ptr<runtime::Tensor> output = backend->create_tensor(element::f32, shape_a);

    vector<float> dataInput{11.f, 31.f, 40.f, 47.f, 13.f, 61.f, 48.f, 59.f, 17.f, 39.f, 64.f,
                            62.f, 45.f, 55.f, 36.f, 19.f, 65.f, 33.f, 49.f, 30.f, 56.f, 41.f,
                            53.f, 58.f, 22.f, 35.f, 52.f, 50.f, 63.f, 54.f, 12.f, 26.f, 44.f,
                            21.f, 69.f, 24.f, 46.f, 25.f, 51.f, 29.f, 72.f, 15.f, 73.f, 10.f,
                            16.f, 37.f, 70.f, 32.f, 28.f, 66.f, 57.f, 27.f, 60.f, 42.f, 43.f,
                            71.f, 18.f, 38.f, 67.f, 68.f, 14.f, 20.f, 34.f, 23.f};

    vector<float> expected{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 12.0f, 0.0f, 4.0f, 0.0f, 0.0f,  16.0f,
                           0.0f, 0.0f, 4.0f, 0.0f, 0.0f, 4.0f,  0.0f, 0.0f, 0.0f, 4.0f,  0.0f,
                           8.0f, 8.0f, 0.0f, 0.0f, 4.0f, 0.0f,  4.0f, 4.0f, 0.0f, 0.0f,  0.0f,
                           0.0f, 8.0f, 0.0f, 4.0f, 0.0f, 0.0f,  0.0f, 8.0f, 0.0f, 16.0f, 0.0f,
                           0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 8.0f,  0.0f, 0.0f, 4.0f, 0.0f,  0.0f,
                           8.0f, 0.0f, 4.0f, 8.0f, 4.0f, 0.0f,  0.0f, 0.0f, 0.0f};

    copy_data(ep, dataEp);
    copy_data(input, dataInput);

    auto C = std::make_shared<op::v0::Parameter>(element::f32, maxpool_shape);
    auto df = autodiff::backprop_function(f);

    {
        OutputVector nv_cwi;
        pass::Manager pass_manager;
        pass_manager.register_pass<runtime::cpu::pass::CPUWorkspaceInsertion>(nv_cwi);
        pass_manager.run_passes(df);
    }

    auto handle = backend->compile(df);
    handle->call_with_validate({output}, {input, ep});
    EXPECT_TRUE(test::all_close_f(read_vector<float>(output), expected, MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, cpu_fusion_conv_batch_norm_folding)
{
    Shape shape_input{1, 8, 3, 3};
    Shape shape_weights{2, 8, 1, 1};
    Shape shape_norm{2};

    auto make_function = [shape_input, shape_weights, shape_norm]() {
        auto input = std::make_shared<op::v0::Parameter>(element::f32, shape_input);
        auto weights = std::make_shared<op::v0::Parameter>(element::f32, shape_weights);
        double eps = 0.001;
        auto gamma = std::make_shared<op::v0::Parameter>(element::f32, shape_norm);
        auto beta = std::make_shared<op::v0::Parameter>(element::f32, shape_norm);
        auto mean = std::make_shared<op::v0::Parameter>(element::f32, shape_norm);
        auto var = std::make_shared<op::v0::Parameter>(element::f32, shape_norm);
        auto conv =
            std::make_shared<op::v0::Convolution>(input, weights, Strides{1, 1}, Strides{1, 1});
        auto bn = std::make_shared<op::v0::BatchNormInference>(conv, gamma, beta, mean, var, eps);
        auto f = make_shared<Function>(OutputVector{bn},
                                       ParameterVector{input, weights, gamma, beta, mean, var});
        return f;
    };

    auto int_f = make_function();
    auto cpu_f = make_function();

    vector<vector<float>> args{
        {1.25f,  2.25f, 5.25f, 6.25f,  -1.25f, -1.25f, 3.25f, -4.25f, 7.25f,  8.25f,  -1.25f,
         -1.25f, 1.25f, 2.25f, -3.25f, 2.25f,  4.25f,  4.25f, 1.25f,  2.25f,  -4.25f, 2.25f,
         4.25f,  4.25f, 0.f,   0.f,    -1.f,   0.f,    2.f,   2.f,    0.f,    0.f,    0.f,
         0.f,    2.f,   2.f,   1.25f,  2.25f,  5.25f,  6.25f, 1.25f,  1.25f,  3.25f,  4.25f,
         -7.25f, 8.25f, 1.25f, -1.25f, -1.25f, 2.25f,  3.25f, 2.25f,  -4.25f, -4.25f, -1.25f,
         -2.25f, 4.25f, 2.25f, 4.25f,  4.25f,  0.f,    0.f,   1.f,    0.f,    -2.f,   2.f,
         0.f,    0.f,   0.f,   0.f,    -2.f,   -2.f},
        {1.25f,
         2.25f,
         5.25f,
         6.25f,
         -1.25f,
         -1.25f,
         3.25f,
         -4.25f,
         7.25f,
         8.25f,
         -1.25f,
         0.f,
         0.f,
         0.f,
         0.f,
         -2.f},
        {-0.9384f, 0.01875f},
        {11.0f, 1.3f},
        {0.12f, 0.31f},
        {0.01f, 0.11f},
    };

    auto int_results = execute(int_f, args, "INTERPRETER");
    auto cpu_results = execute(cpu_f, args, "${BACKEND_NAME}");
    EXPECT_TRUE(test::all_close(cpu_results.at(0), int_results.at(0)));
}

NGRAPH_TEST(${BACKEND_NAME}, cpu_fusion_convbias_batch_norm_folding)
{
    Shape shape_input{2, 8, 5, 5};
    Shape shape_weights{2, 8, 2, 2};
    Shape shape_norm{2};

    auto make_function = [shape_input, shape_weights, shape_norm]() {
        auto input = std::make_shared<op::v0::Parameter>(element::f32, shape_input);
        auto weights = std::make_shared<op::v0::Parameter>(element::f32, shape_weights);
        auto bias = std::make_shared<op::v0::Parameter>(element::f32, Shape{2});
        double eps = 1.01;
        auto gamma = std::make_shared<op::v0::Parameter>(element::f32, shape_norm);
        auto beta = std::make_shared<op::v0::Parameter>(element::f32, shape_norm);
        auto mean = std::make_shared<op::v0::Parameter>(element::f32, shape_norm);
        auto var = std::make_shared<op::v0::Parameter>(element::f32, shape_norm);
        auto conv =
            std::make_shared<op::v0::Convolution>(input, weights, Strides{1, 1}, Strides{1, 1});
        auto convbias = conv + std::make_shared<op::v0::Broadcast>(
                                   bias, conv->get_output_shape(0), AxisSet{0, 2, 3});
        auto bn =
            std::make_shared<op::v0::BatchNormInference>(convbias, gamma, beta, mean, var, eps);
        auto f = make_shared<Function>(
            OutputVector{bn}, ParameterVector{input, weights, bias, gamma, beta, mean, var});
        return f;
    };

    auto int_f = make_function();
    auto cpu_f = make_function();

    test::Uniform<float> rng(1.0f, 100.0f);
    vector<vector<float>> args;
    for (shared_ptr<op::v0::Parameter> param : cpu_f->get_parameters())
    {
        vector<float> tensor_val(shape_size(param->get_output_shape(0)));
        rng.initialize(tensor_val);
        args.push_back(tensor_val);
    }

    auto int_results = execute(int_f, args, "INTERPRETER");
    auto cpu_results = execute(cpu_f, args, "${BACKEND_NAME}");
    EXPECT_TRUE(test::all_close(cpu_results.at(0), int_results.at(0)));
}

NGRAPH_TEST(${BACKEND_NAME}, cpu_fusion_conv_affine_folding)
{
    Shape shape_input{1, 8, 3, 3};
    Shape shape_weights{2, 8, 1, 1};
    Shape shape_norm{2};

    auto make_function = [shape_input, shape_weights, shape_norm]() {
        auto input = std::make_shared<op::v0::Parameter>(element::f32, shape_input);
        auto weights = std::make_shared<op::v0::Parameter>(element::f32, shape_weights);

        auto a = std::make_shared<op::v0::Parameter>(element::f32, shape_norm);
        auto b = std::make_shared<op::v0::Parameter>(element::f32, shape_norm);
        auto conv =
            std::make_shared<op::v0::Convolution>(input, weights, Strides{1, 1}, Strides{1, 1});
        auto out = std::make_shared<op::v1::Add>(
            std::make_shared<op::v1::Multiply>(conv,
                                               std::make_shared<op::v0::Broadcast>(
                                                   a, conv->get_output_shape(0), AxisSet{0, 2, 3})),
            std::make_shared<op::v0::Broadcast>(b, conv->get_output_shape(0), AxisSet{0, 2, 3}));
        auto f = make_shared<Function>(OutputVector{out}, ParameterVector{input, weights, a, b});
        return f;
    };

    auto int_f = make_function();
    auto cpu_f = make_function();

    vector<vector<float>> args{
        {1.25f,  2.25f, 5.25f, 6.25f,  -1.25f, -1.25f, 3.25f, -4.25f, 7.25f,  8.25f,  -1.25f,
         -1.25f, 1.25f, 2.25f, -3.25f, 2.25f,  4.25f,  4.25f, 1.25f,  2.25f,  -4.25f, 2.25f,
         4.25f,  4.25f, 0.f,   0.f,    -1.f,   0.f,    2.f,   2.f,    0.f,    0.f,    0.f,
         0.f,    2.f,   2.f,   1.25f,  2.25f,  5.25f,  6.25f, 1.25f,  1.25f,  3.25f,  4.25f,
         -7.25f, 8.25f, 1.25f, -1.25f, -1.25f, 2.25f,  3.25f, 2.25f,  -4.25f, -4.25f, -1.25f,
         -2.25f, 4.25f, 2.25f, 4.25f,  4.25f,  0.f,    0.f,   1.f,    0.f,    -2.f,   2.f,
         0.f,    0.f,   0.f,   0.f,    -2.f,   -2.f},
        {1.25f,
         2.25f,
         5.25f,
         6.25f,
         -1.25f,
         -1.25f,
         3.25f,
         -4.25f,
         7.25f,
         8.25f,
         -1.25f,
         0.f,
         0.f,
         0.f,
         0.f,
         -2.f},
        {-0.9384f, 0.01875f},
        {11.0f, 1.3f},
    };

    auto int_results = execute(int_f, args, "INTERPRETER");
    auto cpu_results = execute(cpu_f, args, "${BACKEND_NAME}");
    EXPECT_TRUE(test::all_close(cpu_results.at(0), int_results.at(0)));
}

NGRAPH_TEST(${BACKEND_NAME}, cpu_fusion_convbias_affine_folding1)
{
    Shape shape_input{1, 6, 3, 3};
    Shape shape_weights{3, 6, 1, 1};
    Shape shape_norm{3};

    auto make_function = [shape_input, shape_weights, shape_norm]() {
        auto input = std::make_shared<op::v0::Parameter>(element::f32, shape_input);
        auto weights = std::make_shared<op::v0::Parameter>(element::f32, shape_weights);
        auto bias = std::make_shared<op::v0::Parameter>(element::f32, Shape{3});

        auto a = std::make_shared<op::v0::Parameter>(element::f32, shape_norm);
        auto b = std::make_shared<op::v0::Parameter>(element::f32, shape_norm);
        auto conv =
            std::make_shared<op::v0::Convolution>(input, weights, Strides{1, 1}, Strides{1, 1});
        auto convbias = conv + std::make_shared<op::v0::Broadcast>(
                                   bias, conv->get_output_shape(0), AxisSet{0, 2, 3});
        auto out = std::make_shared<op::v1::Add>(
            std::make_shared<op::v1::Multiply>(convbias,
                                               std::make_shared<op::v0::Broadcast>(
                                                   a, conv->get_output_shape(0), AxisSet{0, 2, 3})),
            std::make_shared<op::v0::Broadcast>(b, conv->get_output_shape(0), AxisSet{0, 2, 3}));
        auto f =
            make_shared<Function>(OutputVector{out}, ParameterVector{input, weights, bias, a, b});
        return f;
    };

    pass::Manager pass_manager;
    pass_manager.register_pass<runtime::cpu::pass::CPUFusion>();
    auto func = make_function();
    pass_manager.run_passes(func);
    ASSERT_EQ(count_ops_of_type<op::v0::ConvolutionBiasAdd>(func), 1);

    auto int_f = make_function();
    auto cpu_f = make_function();

    test::Uniform<float> rng(20.0f, 300.0f);
    vector<vector<float>> args;
    for (shared_ptr<op::v0::Parameter> param : cpu_f->get_parameters())
    {
        vector<float> tensor_val(shape_size(param->get_output_shape(0)));
        rng.initialize(tensor_val);
        args.push_back(tensor_val);
    }

    auto int_results = execute(int_f, args, "INTERPRETER");
    auto cpu_results = execute(cpu_f, args, "${BACKEND_NAME}");
    EXPECT_TRUE(test::all_close(cpu_results.at(0), int_results.at(0)));
}

NGRAPH_TEST(${BACKEND_NAME}, cpu_fusion_convbias_affine_folding2)
{
    Shape shape_input{1, 6, 3, 3};
    Shape shape_weights{3, 6, 1, 1};
    Shape shape_norm{1};

    auto make_function = [shape_input, shape_weights, shape_norm]() {
        auto input = std::make_shared<op::v0::Parameter>(element::f32, shape_input);
        auto weights = std::make_shared<op::v0::Parameter>(element::f32, shape_weights);
        auto bias = std::make_shared<op::v0::Parameter>(element::f32, Shape{3});

        auto a = std::make_shared<op::v0::Parameter>(element::f32, shape_norm);
        auto b = std::make_shared<op::v0::Parameter>(element::f32, shape_norm);
        auto conv =
            std::make_shared<op::v0::Convolution>(input, weights, Strides{1, 1}, Strides{1, 1});
        auto convbias = conv + std::make_shared<op::v0::Broadcast>(
                                   bias, conv->get_output_shape(0), AxisSet{0, 2, 3});
        auto out = std::make_shared<op::v1::Add>(
            std::make_shared<op::v1::Multiply>(convbias,
                                               std::make_shared<op::v0::Broadcast>(
                                                   a, conv->get_output_shape(0), AxisSet{1, 2, 3})),
            std::make_shared<op::v0::Broadcast>(b, conv->get_output_shape(0), AxisSet{1, 2, 3}));
        auto f =
            make_shared<Function>(OutputVector{out}, ParameterVector{input, weights, bias, a, b});
        return f;
    };

    pass::Manager pass_manager;
    pass_manager.register_pass<runtime::cpu::pass::CPUFusion>();
    auto func = make_function();
    pass_manager.run_passes(func);
    ASSERT_EQ(count_ops_of_type<op::v0::ConvolutionBiasAdd>(func), 1);

    auto int_f = make_function();
    auto cpu_f = make_function();

    test::Uniform<float> rng(20.0f, 300.0f);
    vector<vector<float>> args;
    for (shared_ptr<op::v0::Parameter> param : cpu_f->get_parameters())
    {
        vector<float> tensor_val(shape_size(param->get_output_shape(0)));
        rng.initialize(tensor_val);
        args.push_back(tensor_val);
    }

    auto int_results = execute(int_f, args, "INTERPRETER");
    auto cpu_results = execute(cpu_f, args, "${BACKEND_NAME}");
    EXPECT_TRUE(test::all_close(cpu_results.at(0), int_results.at(0)));
}

NGRAPH_TEST(${BACKEND_NAME}, batch_fusion_group_convolution)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    test::Uniform<float> rng(2.0f, 10.0f);

    const size_t GROUPS = 2;
    Shape shape_a{1, 32, 2, 2};
    auto A = make_shared<op::v0::Parameter>(element::f32, shape_a);
    Shape shape_b{2, 16, 1, 1};
    auto B = make_shared<op::v0::Parameter>(element::f32, shape_b);
    Shape shape_r{1, 2, 2, 2};
    auto group_conv = make_shared<op::v0::GroupConvolution>(A,
                                                            B,
                                                            Strides{1, 1},
                                                            Strides{1, 1},
                                                            CoordinateDiff{0, 0},
                                                            CoordinateDiff{0, 0},
                                                            Strides{1, 1},
                                                            GROUPS);

    Shape shape_c{1, 16, 2, 2};
    auto C = make_shared<op::v0::Parameter>(element::f32, shape_c);
    Shape shape_d{1, 16, 1, 1};
    auto D = make_shared<op::v0::Parameter>(element::f32, shape_d);
    auto conv_lower = make_shared<op::v0::Convolution>(C,
                                                       D,
                                                       Strides{1, 1},
                                                       Strides{1, 1},
                                                       CoordinateDiff{0, 0},
                                                       CoordinateDiff{0, 0},
                                                       Strides{1, 1});

    auto E = make_shared<op::v0::Parameter>(element::f32, shape_c);
    auto F = make_shared<op::v0::Parameter>(element::f32, shape_d);
    auto conv_upper = make_shared<op::v0::Convolution>(E,
                                                       F,
                                                       Strides{1, 1},
                                                       Strides{1, 1},
                                                       CoordinateDiff{0, 0},
                                                       CoordinateDiff{0, 0},
                                                       Strides{1, 1});

    auto f = make_shared<Function>(OutputVector{group_conv, conv_lower, conv_upper},
                                   ParameterVector{A, B, C, D, E, F});

    auto a_ = rng.initialize(backend->create_tensor(element::f32, shape_a));
    auto b_ = rng.initialize(backend->create_tensor(element::f32, shape_b));

    vector<float> rv(shape_size(shape_r), 0);
    auto group_result = std::dynamic_pointer_cast<ngraph::runtime::cpu::CPUTensor>(
        backend->create_tensor(element::f32, shape_r, rv.data()));

    auto av = read_vector<float>(a_);
    auto bv = read_vector<float>(b_);
    auto c_ = backend->create_tensor(element::f32, shape_c, av.data()); // lower data
    auto d_ = backend->create_tensor(element::f32, shape_d, bv.data()); // upper data

    auto e_ =
        backend->create_tensor(element::f32, shape_c, av.data() + av.size() / 2); // lower weights
    auto f_ =
        backend->create_tensor(element::f32, shape_d, bv.data() + bv.size() / 2); // upper weights

    Shape shape_ur{1, 1, 2, 2};
    // allocate a contigious storage for both lower and upper halves.
    vector<float> erv(shape_size(shape_r), 0);
    auto lower_result = std::dynamic_pointer_cast<ngraph::runtime::cpu::CPUTensor>(
        backend->create_tensor(element::f32, shape_ur, erv.data()));
    auto upper_result = std::dynamic_pointer_cast<ngraph::runtime::cpu::CPUTensor>(
        backend->create_tensor(element::f32, shape_ur, erv.data() + erv.size() / 2));
    auto handle = backend->compile(f);
    handle->call_with_validate({group_result, lower_result, upper_result},
                               {a_, b_, c_, d_, e_, f_});
    EXPECT_TRUE(test::all_close_f(rv, erv));
}

NGRAPH_TEST(${BACKEND_NAME}, cpu_fusion_rnn_fprop_1_lstm_cell)
{
    auto src_layer = make_shared<op::v0::Parameter>(element::f32, Shape{10, 100});
    auto src_iter = make_shared<op::v0::Parameter>(element::f32, Shape{10, 100});
    auto src_iter_c = make_shared<op::v0::Parameter>(element::f32, Shape{10, 100});
    auto weights_layer = make_shared<op::v0::Parameter>(element::f32, Shape{100, 400});
    auto weights_iter = make_shared<op::v0::Parameter>(element::f32, Shape{100, 400});
    auto biases = make_shared<op::v0::Parameter>(element::f32, Shape{400});
    const int number_of_timesteps = 1;
    const int number_of_gates_per_cell = 4;
    const int src_seq_length = 1;
    const int num_rnn_cell_states = 2;
    const int rnn_direction = 1;
    const int num_of_rnn_fused_layer = 1;
    ngraph::runtime::cpu::rnn_utils::rnntype rnn_type =
        ngraph::runtime::cpu::rnn_utils::rnntype::vanilla_lstm;

    auto rnn_node = make_shared<op::Rnn>(src_layer,
                                         src_iter,
                                         src_iter_c,
                                         weights_layer,
                                         weights_iter,
                                         biases,
                                         number_of_timesteps,
                                         number_of_gates_per_cell,
                                         src_seq_length,
                                         num_rnn_cell_states,
                                         rnn_direction,
                                         num_of_rnn_fused_layer,
                                         rnn_type);

    auto rnn_ht_output = rnn_node->output(1);
    auto rnn_ct_output = rnn_node->output(2);

    auto func = make_shared<Function>(
        OutputVector{rnn_ht_output, rnn_ct_output},
        ParameterVector{src_layer, src_iter, src_iter_c, weights_layer, weights_iter, biases});
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    shared_ptr<runtime::Tensor> src_layer_t =
        backend->create_tensor(element::f32, src_layer->get_output_shape(0));
    shared_ptr<runtime::Tensor> src_iter_t =
        backend->create_tensor(element::f32, src_iter->get_output_shape(0));
    shared_ptr<runtime::Tensor> src_iter_c_t =
        backend->create_tensor(element::f32, src_iter_c->get_output_shape(0));
    shared_ptr<runtime::Tensor> weights_layer_t =
        backend->create_tensor(element::f32, weights_layer->get_output_shape(0));
    shared_ptr<runtime::Tensor> weights_iter_t =
        backend->create_tensor(element::f32, weights_iter->get_output_shape(0));
    shared_ptr<runtime::Tensor> biases_t =
        backend->create_tensor(element::f32, biases->get_output_shape(0));
    shared_ptr<runtime::Tensor> result_ht = backend->create_tensor(element::f32, {10, 100});
    shared_ptr<runtime::Tensor> result_ct = backend->create_tensor(element::f32, Shape{10, 100});

    copy_data(src_layer_t, vector<float>(1000, 1));
    copy_data(src_iter_t, vector<float>(1000, 1));
    copy_data(src_iter_c_t, vector<float>(1000, 1));
    copy_data(weights_layer_t, vector<float>(400 * 100, 1));
    copy_data(weights_iter_t, vector<float>(400 * 100, 1));
    copy_data(biases_t, vector<float>(400, 1));

    auto handle = backend->compile(func);
    handle->call_with_validate(
        {result_ht, result_ct},
        {src_layer_t, src_iter_t, src_iter_c_t, weights_layer_t, weights_iter_t, biases_t});
    vector<float> expected_ht(10 * 100, 0.964028f);
    vector<float> expected_ct(10 * 100, 2.0f);

    EXPECT_TRUE(test::all_close(expected_ht, read_vector<float>(result_ht)));
    EXPECT_TRUE(test::all_close(expected_ct, read_vector<float>(result_ct)));
}

namespace
{
    void sigmoid_multiply_fusion_forward_compute(runtime::Backend* backend,
                                                 const ParameterVector& input_params,
                                                 const vector<vector<float>>& input_data,
                                                 const vector<Shape>& input_shapes,
                                                 const Shape& result_shape,
                                                 shared_ptr<Node> input_0_node,
                                                 shared_ptr<Node> input_1_node,
                                                 const vector<float>& expected)
    {
        shared_ptr<runtime::Tensor> result_tensor =
            backend->create_tensor(element::f32, result_shape);

        vector<shared_ptr<runtime::Tensor>> input_tensors;
        for (size_t i = 0; i < input_params.size(); ++i)
        {
            input_tensors.push_back(backend->create_tensor(element::f32, input_shapes[i]));
            copy_data(input_tensors[i], input_data[i]);
        }

        auto mul_node = input_0_node * input_1_node;
        auto func = make_shared<Function>(mul_node, input_params);
        auto handle = backend->compile(func);
        handle->call_with_validate({result_tensor}, input_tensors);
        EXPECT_TRUE(test::all_close(read_vector<float>(result_tensor), expected));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, cpu_fusion_sigmoid_multiply_fusion_forward)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    Shape data_shape{1, 1, 2, 2};
    Shape const_shape{1};

    vector<float> input_0_data{1.f, 2.f, 3.f, 4.f};
    vector<float> input_1_data{1.2f, 2.3f, 3.5f, 4.7f};
    vector<float> const_data{1.2f};
    {
        auto input_0_param = make_shared<op::v0::Parameter>(element::f32, data_shape);
        auto input_1_param = make_shared<op::v0::Parameter>(element::f32, data_shape);
        auto input_2_param = make_shared<op::v0::Parameter>(element::f32, data_shape);
        auto sigmoid_0 = make_shared<op::v0::Sigmoid>(input_0_param);
        auto sigmoid_1 = make_shared<op::v1::Add>(input_1_param, input_2_param);
        vector<float> expected{1.60833f, 3.78743f, 6.19173f, 8.54352f};
        ParameterVector input_params{input_0_param, input_1_param, input_2_param};
        vector<vector<float>> input_data{input_0_data, input_0_data, input_1_data};
        vector<Shape> input_shapes{data_shape, data_shape, data_shape};
        sigmoid_multiply_fusion_forward_compute(backend.get(),
                                                input_params,
                                                input_data,
                                                input_shapes,
                                                data_shape,
                                                sigmoid_0,
                                                sigmoid_1,
                                                expected);
    }
    {
        auto input_0_param = make_shared<op::v0::Parameter>(element::f32, data_shape);
        auto input_1_param = make_shared<op::v0::Parameter>(element::f32, const_shape);
        auto sigmoid_0 =
            make_shared<op::v0::Broadcast>(input_1_param, data_shape, AxisSet{1, 2, 3});
        auto sigmoid_1 = make_shared<op::v0::Sigmoid>(input_0_param);
        vector<float> expected{0.87727f, 1.05696f, 1.14309f, 1.17842f};
        ParameterVector input_params{input_0_param, input_1_param};
        vector<vector<float>> input_data{input_0_data, const_data};
        vector<Shape> input_shapes{data_shape, const_shape};
        sigmoid_multiply_fusion_forward_compute(backend.get(),
                                                input_params,
                                                input_data,
                                                input_shapes,
                                                data_shape,
                                                sigmoid_0,
                                                sigmoid_1,
                                                expected);
    }
    {
        auto input_0_param = make_shared<op::v0::Parameter>(element::f32, data_shape);
        auto input_1_param = make_shared<op::v0::Parameter>(element::f32, const_shape);
        auto sigmoid_0 = make_shared<op::v0::Sigmoid>(input_0_param);
        auto sigmoid_1 =
            make_shared<op::v0::Broadcast>(input_1_param, data_shape, AxisSet{1, 2, 3});
        vector<float> expected{0.87727f, 1.05696f, 1.14309f, 1.17842f};
        ParameterVector input_params{input_0_param, input_1_param};
        vector<vector<float>> input_data{input_0_data, const_data};
        vector<Shape> input_shapes{data_shape, const_shape};
        sigmoid_multiply_fusion_forward_compute(backend.get(),
                                                input_params,
                                                input_data,
                                                input_shapes,
                                                data_shape,
                                                sigmoid_0,
                                                sigmoid_1,
                                                expected);
    }
    {
        auto input_0_param = make_shared<op::v0::Parameter>(element::f32, data_shape);
        auto input_1_param = make_shared<op::v0::Parameter>(element::f32, data_shape);
        auto sigmoid_0 = make_shared<op::v0::Sigmoid>(input_0_param);
        auto sigmoid_1 = make_shared<op::v0::Sigmoid>(input_1_param);
        vector<float> expected{0.561837f, 0.800536f, 0.924652f, 0.973163f};
        ParameterVector input_params{input_0_param, input_1_param};
        vector<vector<float>> input_data{input_0_data, input_1_data};
        vector<Shape> input_shapes{data_shape, data_shape};
        sigmoid_multiply_fusion_forward_compute(backend.get(),
                                                input_params,
                                                input_data,
                                                input_shapes,
                                                data_shape,
                                                sigmoid_0,
                                                sigmoid_1,
                                                expected);
    }
    {
        auto input_0_param = make_shared<op::v0::Parameter>(element::f32, data_shape);
        auto input_1_param = make_shared<op::v0::Parameter>(element::f32, data_shape);
        auto sigmoid_0 = make_shared<op::v0::Sigmoid>(input_0_param);
        auto sigmoid_1 = make_shared<op::v0::Tanh>(input_1_param);
        vector<float> expected{0.60945f, 0.863266f, 0.950838f, 0.981851f};
        ParameterVector input_params{input_0_param, input_1_param};
        vector<vector<float>> input_data{input_0_data, input_1_data};
        vector<Shape> input_shapes{data_shape, data_shape};
        sigmoid_multiply_fusion_forward_compute(backend.get(),
                                                input_params,
                                                input_data,
                                                input_shapes,
                                                data_shape,
                                                sigmoid_0,
                                                sigmoid_1,
                                                expected);
    }
    {
        auto input_0_param = make_shared<op::v0::Parameter>(element::f32, data_shape);
        auto input_1_param = make_shared<op::v0::Parameter>(element::f32, data_shape);
        auto sigmoid_0 = make_shared<op::v0::Tanh>(input_0_param);
        auto sigmoid_1 = make_shared<op::v0::Sigmoid>(input_1_param);
        vector<float> expected{0.585304f, 0.876182f, 0.965887f, 0.990322f};
        ParameterVector input_params{input_0_param, input_1_param};
        vector<vector<float>> input_data{input_0_data, input_1_data};
        vector<Shape> input_shapes{data_shape, data_shape};
        sigmoid_multiply_fusion_forward_compute(backend.get(),
                                                input_params,
                                                input_data,
                                                input_shapes,
                                                data_shape,
                                                sigmoid_0,
                                                sigmoid_1,
                                                expected);
    }
    {
        auto input_0_param = make_shared<op::v0::Parameter>(element::f32, data_shape);
        auto input_1_param = make_shared<op::v0::Parameter>(element::f32, data_shape);
        auto sigmoid_0 = make_shared<op::v0::Tanh>(input_0_param);
        auto sigmoid_1 = make_shared<op::v0::Tanh>(input_1_param);
        vector<float> expected{0.634907f, 0.94484f, 0.993242f, 0.999164f};
        ParameterVector input_params{input_0_param, input_1_param};
        vector<vector<float>> input_data{input_0_data, input_1_data};
        vector<Shape> input_shapes{data_shape, data_shape};
        sigmoid_multiply_fusion_forward_compute(backend.get(),
                                                input_params,
                                                input_data,
                                                input_shapes,
                                                data_shape,
                                                sigmoid_0,
                                                sigmoid_1,
                                                expected);
    }
}

namespace
{
    void sigmoid_multiply_fusion_backward_compute(runtime::Backend* backend,
                                                  const ParameterVector& input_params,
                                                  const vector<vector<float>>& input_data,
                                                  const vector<Shape>& input_shapes,
                                                  const vector<float> delta_data,
                                                  const Shape& delta_shape,
                                                  const Shape& d_input_0_shape,
                                                  const Shape& d_input_1_shape,
                                                  shared_ptr<Node> input_0_node,
                                                  shared_ptr<Node> input_1_node,
                                                  shared_ptr<Node> input_0_adjoint,
                                                  shared_ptr<Node> input_1_adjoint,
                                                  const vector<float>& expected_0,
                                                  const vector<float>& expected_1)
    {
        vector<shared_ptr<runtime::Tensor>> input_tensors;
        for (size_t i = 0; i < input_params.size(); ++i)
        {
            input_tensors.push_back(backend->create_tensor(element::f32, input_shapes[i]));
            copy_data(input_tensors[i], input_data[i]);
        }

        auto delta_param = make_shared<op::v0::Parameter>(element::f32, delta_shape);
        shared_ptr<runtime::Tensor> delta_tensor =
            backend->create_tensor(element::f32, delta_shape);
        copy_data(delta_tensor, delta_data);

        ParameterVector back_params(input_params);
        back_params.push_back(delta_param);
        input_tensors.push_back(delta_tensor);

        shared_ptr<runtime::Tensor> d_input_0_tensor =
            backend->create_tensor(element::f32, d_input_0_shape);
        shared_ptr<runtime::Tensor> d_input_1_tensor =
            backend->create_tensor(element::f32, d_input_1_shape);

        using FunctionType = op::SigmoidMultiply::FunctionType;
        auto input_0_type = op::SigmoidMultiply::identify_node_type(input_0_node);
        auto input_1_type = op::SigmoidMultiply::identify_node_type(input_1_node);
        // for Identity functions, we use the node itself, otherwise use its input
        // where we will apply the function of input node
        auto input_0_alt =
            (input_0_type == FunctionType::Identity) ? input_0_node : input_0_node->get_argument(0);
        auto input_1_alt =
            (input_1_type == FunctionType::Identity) ? input_1_node : input_1_node->get_argument(0);
        auto sigmoid_mul =
            make_shared<op::SigmoidMultiply>(input_0_alt, input_1_alt, input_0_type, input_1_type);

        ngraph::autodiff::Adjoints adjoints(OutputVector{sigmoid_mul}, OutputVector{delta_param});
        auto d_input_0 = adjoints.backprop_output(input_0_adjoint);
        auto d_input_1 = adjoints.backprop_output(input_1_adjoint);
        auto df = make_shared<Function>(OutputVector{d_input_0, d_input_1}, back_params);
        auto handle = backend->compile(df);
        handle->call_with_validate({d_input_0_tensor, d_input_1_tensor}, input_tensors);
        EXPECT_TRUE(test::all_close(read_vector<float>(d_input_0_tensor), expected_0));
        EXPECT_TRUE(test::all_close(read_vector<float>(d_input_1_tensor), expected_1));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, cpu_fusion_sigmoid_multiply_fusion_backward)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    Shape data_shape{1, 1, 2, 2};
    Shape const_shape{1};

    vector<float> input_0_data{1.f, 2.f, 3.f, 4.f};
    vector<float> input_1_data{1.2f, 2.2f, 3.2f, 4.2f};
    vector<float> const_data{1.2f};
    vector<float> delta_data(shape_size(data_shape), 20.0f);

    {
        auto input_0_param = make_shared<op::v0::Parameter>(element::f32, data_shape);
        auto input_1_param = make_shared<op::v0::Parameter>(element::f32, data_shape);
        auto input_2_param = make_shared<op::v0::Parameter>(element::f32, data_shape);
        auto sigmoid_0 = make_shared<op::v0::Sigmoid>(input_0_param);
        auto sigmoid_1 =
            make_shared<op::v1::Add>(input_1_param, input_2_param, op::AutoBroadcastType::NONE);
        vector<float> expected_0{8.65093f, 8.81946f, 5.60191f, 2.89668f};
        vector<float> expected_1{14.6212f, 17.6159f, 19.0515f, 19.6403f};
        ParameterVector input_params{input_0_param, input_1_param, input_2_param};
        vector<vector<float>> input_data{input_0_data, input_0_data, input_1_data};
        vector<Shape> input_shapes{data_shape, data_shape, data_shape};
        sigmoid_multiply_fusion_backward_compute(backend.get(),
                                                 input_params,
                                                 input_data,
                                                 input_shapes,
                                                 delta_data,
                                                 data_shape,
                                                 data_shape,
                                                 data_shape,
                                                 sigmoid_0,
                                                 sigmoid_1,
                                                 input_0_param,
                                                 sigmoid_1,
                                                 expected_0,
                                                 expected_1);
    }
    {
        auto input_0_param = make_shared<op::v0::Parameter>(element::f32, data_shape);
        auto input_1_param = make_shared<op::v0::Parameter>(element::f32, const_shape);
        auto sigmoid_0 =
            make_shared<op::v0::Broadcast>(input_1_param, data_shape, AxisSet{1, 2, 3});
        auto sigmoid_1 = make_shared<op::v0::Tanh>(input_0_param);
        vector<float> expected_0{15.2319f, 19.2806f, 19.9011f, 19.9866f};
        vector<float> expected_1{10.0794f, 1.69562f, 0.236785f, 0.0321828f};
        ParameterVector input_params{input_0_param, input_1_param};
        vector<vector<float>> input_data{input_0_data, const_data};
        vector<Shape> input_shapes{data_shape, const_shape};
        sigmoid_multiply_fusion_backward_compute(backend.get(),
                                                 input_params,
                                                 input_data,
                                                 input_shapes,
                                                 delta_data,
                                                 data_shape,
                                                 data_shape,
                                                 data_shape,
                                                 sigmoid_0,
                                                 sigmoid_1,
                                                 sigmoid_0,
                                                 input_0_param,
                                                 expected_0,
                                                 expected_1);
    }
    {
        auto input_0_param = make_shared<op::v0::Parameter>(element::f32, data_shape);
        auto input_1_param = make_shared<op::v0::Parameter>(element::f32, const_shape);
        auto sigmoid_0 = make_shared<op::v0::Tanh>(input_0_param);
        auto sigmoid_1 =
            make_shared<op::v0::Broadcast>(input_1_param, data_shape, AxisSet{1, 2, 3});
        vector<float> expected_0{10.0794f, 1.69562f, 0.236785f, 0.0321828f};
        vector<float> expected_1{15.2319f, 19.2806f, 19.9011f, 19.9866f};
        ParameterVector input_params{input_0_param, input_1_param};
        vector<vector<float>> input_data{input_0_data, const_data};
        vector<Shape> input_shapes{data_shape, const_shape};
        sigmoid_multiply_fusion_backward_compute(backend.get(),
                                                 input_params,
                                                 input_data,
                                                 input_shapes,
                                                 delta_data,
                                                 data_shape,
                                                 data_shape,
                                                 data_shape,
                                                 sigmoid_0,
                                                 sigmoid_1,
                                                 input_0_param,
                                                 sigmoid_1,
                                                 expected_0,
                                                 expected_1);
    }
    {
        auto input_0_param = make_shared<op::v0::Parameter>(element::f32, data_shape);
        auto input_1_param = make_shared<op::v0::Parameter>(element::f32, data_shape);
        auto sigmoid_0 = make_shared<op::v0::Sigmoid>(input_0_param);
        auto sigmoid_1 = make_shared<op::v0::Sigmoid>(input_1_param);
        vector<float> expected_0{3.02202f, 1.89041f, 0.868146f, 0.348035f};
        vector<float> expected_1{2.60102f, 1.58192f, 0.716941f, 0.285879f};
        ParameterVector input_params{input_0_param, input_1_param};
        vector<vector<float>> input_data{input_0_data, input_1_data};
        vector<Shape> input_shapes{data_shape, data_shape};
        sigmoid_multiply_fusion_backward_compute(backend.get(),
                                                 input_params,
                                                 input_data,
                                                 input_shapes,
                                                 delta_data,
                                                 data_shape,
                                                 data_shape,
                                                 data_shape,
                                                 sigmoid_0,
                                                 sigmoid_1,
                                                 input_0_param,
                                                 input_1_param,
                                                 expected_0,
                                                 expected_1);
    }
    {
        auto input_0_param = make_shared<op::v0::Parameter>(element::f32, data_shape);
        auto input_1_param = make_shared<op::v0::Parameter>(element::f32, data_shape);
        auto sigmoid_0 = make_shared<op::v0::Sigmoid>(input_0_param);
        auto sigmoid_1 = make_shared<op::v0::Tanh>(input_1_param);
        vector<float> expected_0{3.27813f, 2.04894f, 0.900536f, 0.353095f};
        vector<float> expected_1{4.45975f, 0.84425f, 0.126201f, 0.0176579f};
        ParameterVector input_params{input_0_param, input_1_param};
        vector<vector<float>> input_data{input_0_data, input_1_data};
        vector<Shape> input_shapes{data_shape, data_shape};
        sigmoid_multiply_fusion_backward_compute(backend.get(),
                                                 input_params,
                                                 input_data,
                                                 input_shapes,
                                                 delta_data,
                                                 data_shape,
                                                 data_shape,
                                                 data_shape,
                                                 sigmoid_0,
                                                 sigmoid_1,
                                                 input_0_param,
                                                 input_1_param,
                                                 expected_0,
                                                 expected_1);
    }
    {
        auto input_0_param = make_shared<op::v0::Parameter>(element::f32, data_shape);
        auto input_1_param = make_shared<op::v0::Parameter>(element::f32, data_shape);
        auto sigmoid_0 = make_shared<op::v0::Tanh>(input_0_param);
        auto sigmoid_1 = make_shared<op::v0::Sigmoid>(input_1_param);
        vector<float> expected_0{6.45521f, 1.27207f, 0.189593f, 0.0264228f};
        vector<float> expected_1{2.70967f, 1.7314f, 0.748913f, 0.29092f};
        ParameterVector input_params{input_0_param, input_1_param};
        vector<vector<float>> input_data{input_0_data, input_1_data};
        vector<Shape> input_shapes{data_shape, data_shape};
        sigmoid_multiply_fusion_backward_compute(backend.get(),
                                                 input_params,
                                                 input_data,
                                                 input_shapes,
                                                 delta_data,
                                                 data_shape,
                                                 data_shape,
                                                 data_shape,
                                                 sigmoid_0,
                                                 sigmoid_1,
                                                 input_0_param,
                                                 input_1_param,
                                                 expected_0,
                                                 expected_1);
    }
    {
        auto input_0_param = make_shared<op::v0::Parameter>(element::f32, data_shape);
        auto input_1_param = make_shared<op::v0::Parameter>(element::f32, data_shape);
        auto sigmoid_0 = make_shared<op::v0::Tanh>(input_0_param);
        auto sigmoid_1 = make_shared<op::v0::Tanh>(input_1_param);
        vector<float> expected_0{7.00227f, 1.37874f, 0.196666f, 0.026807f};
        vector<float> expected_1{4.64603f, 0.924027f, 0.131829f, 0.0179692f};
        ParameterVector input_params{input_0_param, input_1_param};
        vector<vector<float>> input_data{input_0_data, input_1_data};
        vector<Shape> input_shapes{data_shape, data_shape};
        sigmoid_multiply_fusion_backward_compute(backend.get(),
                                                 input_params,
                                                 input_data,
                                                 input_shapes,
                                                 delta_data,
                                                 data_shape,
                                                 data_shape,
                                                 data_shape,
                                                 sigmoid_0,
                                                 sigmoid_1,
                                                 input_0_param,
                                                 input_1_param,
                                                 expected_0,
                                                 expected_1);
    }
}

namespace
{
    static void
        check_bounded_relu(const string& backend_name, Shape param_shape, float constant_val)
    {
        auto make_function = [](Shape input_shape, float alpha_val) {
            auto relu_input = std::make_shared<op::v0::Parameter>(element::f32, input_shape);
            auto relu = std::make_shared<op::v0::Relu>(relu_input);
            auto alpha = op::v0::Constant::create<float>(
                element::f32, input_shape, std::vector<float>(1.0f, alpha_val));
            auto min = std::make_shared<op::v1::Minimum>(relu, alpha);
            auto f = make_shared<Function>(OutputVector{min}, ParameterVector{relu_input});
            return f;
        };

        auto cpu_f = make_function(param_shape, constant_val);
        auto int_f = make_function(param_shape, constant_val);
        test::Uniform<float> rng(-10.0f, 10.0f);
        vector<vector<float>> args;

        for (shared_ptr<op::v0::Parameter> param : int_f->get_parameters())
        {
            vector<float> tensor_val(shape_size(param->get_output_shape(0)));
            rng.initialize(tensor_val);
            args.push_back(tensor_val);
        }
        auto int_results = execute(int_f, args, "INTERPRETER");
        auto cpu_results = execute(cpu_f, args, backend_name);

        EXPECT_EQ(1, count_ops_of_type<op::BoundedRelu>(cpu_f));
        EXPECT_TRUE(test::all_close(cpu_results.at(0), int_results.at(0), 1.0e-4f, 1.0e-4f));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, cpu_fusion_fuse_bounded_relu_inter_vs_cpu)
{
    check_bounded_relu("${BACKEND_NAME}", Shape{4, 3, 2, 2}, 6.0f);
    check_bounded_relu("${BACKEND_NAME}", Shape{4, 3}, 4.0f);
    check_bounded_relu("${BACKEND_NAME}", Shape{4, 3, 2}, 2.0f);
}

NGRAPH_TEST(${BACKEND_NAME}, cpu_fusion_fuse_dropout)
{
    auto make_function = [](Shape input_shape,
                            const uint32_t seed_val,
                            double one_minus_prob,
                            bool fuse,
                            bool use_seed) {
        auto input = std::make_shared<op::v0::Parameter>(element::f32, input_shape);
        auto value = op::v0::Constant::create(element::f32, input_shape, {one_minus_prob});
        auto const1 = op::v0::Constant::create(input->get_element_type(), Shape{}, {1});

        auto gen_mask = std::make_shared<op::v0::GenerateMask>(const1,
                                                               input->get_output_shape(0),
                                                               input->get_element_type(),
                                                               seed_val,
                                                               one_minus_prob,
                                                               use_seed);

        auto mult = std::make_shared<op::v1::Multiply>(gen_mask, input);

        auto goe = mult->output(0);

        auto pdivide = fuse ? std::make_shared<op::v1::Divide>(mult, value)
                            : std::make_shared<op::v1::Divide>(goe, value);

        auto f = make_shared<Function>(OutputVector{pdivide, gen_mask}, ParameterVector{input});

        return f;
    };

    uint32_t seed = rand();
    auto fuse_func = make_function(Shape{2, 2, 256, 256}, seed, 0.9, true, true);
    auto fuse_func2 = make_function(Shape{2, 2, 256, 256}, seed, 0.9, true, true);
    auto nofuse_func = make_function(Shape{2, 2, 256, 256}, 1, 0.9, false, false);
    {
        pass::Manager pass_manager;
        pass_manager.register_pass<runtime::cpu::pass::CPUFusion>();
        pass_manager.run_passes(fuse_func);
        pass_manager.run_passes(nofuse_func);
        ASSERT_EQ(count_ops_of_type<op::Dropout>(fuse_func), 1);
        ASSERT_EQ(count_ops_of_type<op::v0::GenerateMask>(fuse_func), 0);
        ASSERT_EQ(count_ops_of_type<op::Dropout>(nofuse_func), 0);
    }

    auto fuse_func3 = make_function(Shape{2, 2, 256, 256}, seed, 0.9, true, false);
    auto fuse_func4 = make_function(Shape{2, 2, 256, 256}, seed, 0.9, true, false);
    {
        test::Uniform<float> rng(1.0f, 100.0f);
        vector<vector<float>> args;
        for (shared_ptr<op::v0::Parameter> param : fuse_func->get_parameters())
        {
            auto name = param->get_name();
            vector<float> tensor_val(shape_size(param->get_output_shape(0)));
            rng.initialize(tensor_val);
            args.push_back(tensor_val);
        }

        auto fuse_results = execute(fuse_func, args, "${BACKEND_NAME}");
        auto fuse_results2 = execute(fuse_func2, args, "${BACKEND_NAME}");
        EXPECT_TRUE(test::all_close(fuse_results.at(0), fuse_results2.at(0)));
        EXPECT_TRUE(test::all_close(fuse_results.at(1), fuse_results2.at(1)));

        auto fuse_results3 = execute(fuse_func3, args, "${BACKEND_NAME}");
        auto fuse_results4 = execute(fuse_func4, args, "${BACKEND_NAME}");
        EXPECT_FALSE(test::all_close(fuse_results3.at(0), fuse_results4.at(0)));
        EXPECT_FALSE(test::all_close(fuse_results3.at(1), fuse_results4.at(1)));

        // Note: Since the RNG used in Dropout kernel is different than RNG used in GenerateMask
        // kernel, we can't compare fuse_results and nofuse_results
    }
}

NGRAPH_TEST(${BACKEND_NAME}, cpu_fusion_fuse_leaky_relu)
{
    auto make_function = [](Shape input_shape, vector<float> alpha_val) {
        auto input = std::make_shared<op::v0::Parameter>(element::f32, input_shape);
        auto alpha = op::v0::Constant::create<float>(element::f32, input_shape, alpha_val);
        auto out = std::make_shared<op::v1::Maximum>(
            input, std::make_shared<op::v1::Multiply>(input, alpha));
        auto f = make_shared<Function>(OutputVector{out}, ParameterVector{input});
        return f;
    };

    auto no_fuse1 = make_function(Shape{1, 2, 3}, std::vector<float>(6, -1.0f));
    auto no_fuse2 = make_function(Shape{1, 3}, std::vector<float>{1.4f, 1.2f, 1.4f});

    pass::Manager pass_manager;
    pass_manager.register_pass<runtime::cpu::pass::CPUFusion>();
    pass_manager.run_passes(no_fuse1);
    pass_manager.run_passes(no_fuse2);
    EXPECT_EQ(0, count_ops_of_type<op::CPULeakyRelu>(no_fuse1));
    EXPECT_EQ(0, count_ops_of_type<op::CPULeakyRelu>(no_fuse2));

    // non-dnnl kernel
    auto cpu_f1 = make_function(Shape{1, 2, 3}, std::vector<float>(6, 0.1f));
    // dnnl kernel
    auto cpu_f2 = make_function(Shape{2, 3}, std::vector<float>(6, 0.1f));

    vector<vector<float>> args;
    args.push_back(std::vector<float>{-1, -2, 0, 1, 2, 3});
    std::vector<float> expected_result{-0.1f, -0.2f, 0.0f, 1.0f, 2.0f, 3.0f};

    auto cpu1_results = execute(cpu_f1, args, "${BACKEND_NAME}");
    EXPECT_EQ(1, count_ops_of_type<op::CPULeakyRelu>(cpu_f1));
    EXPECT_TRUE(test::all_close(cpu1_results.at(0), expected_result));

    auto cpu2_results = execute(cpu_f2, args, "${BACKEND_NAME}");
    EXPECT_EQ(1, count_ops_of_type<op::CPULeakyRelu>(cpu_f2));
    EXPECT_TRUE(test::all_close(cpu2_results.at(0), expected_result));
}

NGRAPH_TEST(${BACKEND_NAME}, cpu_fusion_fuse_update_slice)
{
    auto make_function = [](bool fuse = true) {
        auto input = std::make_shared<op::v0::Parameter>(element::f32, Shape{4, 32, 16});
        Shape lower_bounds{1, 0, 0};
        Shape upper_bounds{2, 32, 16};
        auto slice = std::make_shared<op::v0::Slice>(
            input, fuse ? lower_bounds : Shape{3, 0, 0}, fuse ? upper_bounds : Shape{4, 32, 16});
        auto update = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 32, 16});
        auto add = std::make_shared<op::v1::Add>(slice, update);
        auto out = std::make_shared<op::v0::ReplaceSlice>(input, add, lower_bounds, upper_bounds);
        auto f = make_shared<Function>(OutputVector{out}, ParameterVector{input, update});
        return f;
    };

    auto fuse = make_function(true);
    auto no_fuse = make_function(false);

    pass::Manager pass_manager;
    pass_manager.register_pass<runtime::cpu::pass::CPUFusion>();
    pass_manager.run_passes(fuse);
    pass_manager.run_passes(no_fuse);
    EXPECT_EQ(1, count_ops_of_type<op::UpdateSlice>(fuse));
    EXPECT_EQ(0, count_ops_of_type<op::UpdateSlice>(no_fuse));

    auto int_f = make_function();
    auto cpu_f = make_function();

    test::Uniform<float> rng(0.0f, 1.0f);
    vector<vector<float>> args;
    for (shared_ptr<op::v0::Parameter> param : int_f->get_parameters())
    {
        vector<float> tensor_val(shape_size(param->get_output_shape(0)));
        rng.initialize(tensor_val);
        args.push_back(tensor_val);
    }
    auto int_results = execute(int_f, args, "INTERPRETER");
    auto cpu_results = execute(cpu_f, args, "${BACKEND_NAME}");
    for (size_t i = 0; i < cpu_results.size(); i++)
    {
        EXPECT_TRUE(test::all_close(cpu_results.at(i), int_results.at(i)));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, cpu_fusion_fuse_update_slice_inplace)
{
    auto make_function = [](bool fuse = true) {
        auto input = std::make_shared<op::v0::Parameter>(element::f32, Shape{4, 32, 16});
        auto abs = std::make_shared<op::v0::Abs>(input);
        Shape lower_bounds{1, 0, 0};
        Shape upper_bounds{2, 32, 16};
        auto slice = std::make_shared<op::v0::Slice>(abs, lower_bounds, upper_bounds);
        auto update = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 32, 16});
        auto add = std::make_shared<op::v1::Add>(slice, update);
        auto rs = std::make_shared<op::v0::ReplaceSlice>(abs, add, lower_bounds, upper_bounds);
        auto out = std::make_shared<op::v0::Abs>(rs);
        if (fuse)
        {
            return make_shared<Function>(OutputVector{out}, ParameterVector{input, update});
        }
        else
        {
            return make_shared<Function>(OutputVector{out, add}, ParameterVector{input, update});
        }
    };

    auto fuse = make_function(true);
    auto no_fuse = make_function(false);

    pass::Manager pass_manager;
    pass_manager.register_pass<runtime::cpu::pass::CPUFusion>();
    pass_manager.run_passes(fuse);
    pass_manager.run_passes(no_fuse);
    EXPECT_EQ(1, count_ops_of_type<op::UpdateSlice>(fuse));
    EXPECT_EQ(0, count_ops_of_type<op::UpdateSlice>(no_fuse));

    auto int_f = make_function();
    auto cpu_f = make_function();

    test::Uniform<float> rng(0.0f, 1.0f);
    vector<vector<float>> args;
    for (shared_ptr<op::v0::Parameter> param : int_f->get_parameters())
    {
        vector<float> tensor_val(shape_size(param->get_output_shape(0)));
        rng.initialize(tensor_val);
        args.push_back(tensor_val);
    }
    auto int_results = execute(int_f, args, "INTERPRETER");
    auto cpu_results = execute(cpu_f, args, "${BACKEND_NAME}");
    for (size_t i = 0; i < cpu_results.size(); i++)
    {
        EXPECT_TRUE(test::all_close(cpu_results.at(i), int_results.at(i)));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, cpu_fusion_fuse_update_slice_strided)
{
    auto make_function = [](bool fuse = true) {
        auto input = std::make_shared<op::v0::Parameter>(element::f32, Shape{4, 32, 16});
        Shape lower_bounds{1, 0, 0};
        Shape upper_bounds{2, 32, 16};
        Strides strides{1, 2, 2};
        auto slice = std::make_shared<op::v0::Slice>(input,
                                                     fuse ? lower_bounds : Shape{3, 0, 0},
                                                     fuse ? upper_bounds : Shape{4, 32, 16},
                                                     strides);
        auto update = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 16, 8});
        auto add = std::make_shared<op::v1::Add>(slice, update);
        auto out =
            std::make_shared<op::v0::ReplaceSlice>(input, add, lower_bounds, upper_bounds, strides);
        auto f = make_shared<Function>(OutputVector{out}, ParameterVector{input, update});
        return f;
    };

    auto fuse = make_function(true);
    auto no_fuse = make_function(false);

    pass::Manager pass_manager;
    pass_manager.register_pass<runtime::cpu::pass::CPUFusion>();
    pass_manager.run_passes(fuse);
    pass_manager.run_passes(no_fuse);
    EXPECT_EQ(1, count_ops_of_type<op::UpdateSlice>(fuse));
    EXPECT_EQ(0, count_ops_of_type<op::UpdateSlice>(no_fuse));

    auto int_f = make_function();
    auto cpu_f = make_function();

    test::Uniform<float> rng(0.0f, 1.0f);
    vector<vector<float>> args;
    for (shared_ptr<op::v0::Parameter> param : int_f->get_parameters())
    {
        vector<float> tensor_val(shape_size(param->get_output_shape(0)));
        rng.initialize(tensor_val);
        args.push_back(tensor_val);
    }
    auto int_results = execute(int_f, args, "INTERPRETER");
    auto cpu_results = execute(cpu_f, args, "${BACKEND_NAME}");
    for (size_t i = 0; i < cpu_results.size(); i++)
    {
        EXPECT_TRUE(test::all_close(cpu_results.at(i), int_results.at(i)));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, cpu_fusion_fuse_update_slice_strided_inplace)
{
    auto make_function = [](bool fuse = true) {
        auto input = std::make_shared<op::v0::Parameter>(element::f32, Shape{4, 32, 16});
        auto abs = std::make_shared<op::v0::Abs>(input);
        Shape lower_bounds{1, 0, 0};
        Shape upper_bounds{2, 32, 16};
        Strides strides{1, 4, 2};
        auto slice = std::make_shared<op::v0::Slice>(abs, lower_bounds, upper_bounds, strides);
        auto update = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 8, 8});
        auto add = std::make_shared<op::v1::Add>(slice, update);
        auto rs =
            std::make_shared<op::v0::ReplaceSlice>(abs, add, lower_bounds, upper_bounds, strides);
        auto out = std::make_shared<op::v0::Abs>(rs);
        if (fuse)
        {
            return make_shared<Function>(OutputVector{out}, ParameterVector{input, update});
        }
        else
        {
            return make_shared<Function>(OutputVector{out, add}, ParameterVector{input, update});
        }
    };

    auto fuse = make_function(true);
    auto no_fuse = make_function(false);

    pass::Manager pass_manager;
    pass_manager.register_pass<runtime::cpu::pass::CPUFusion>();
    pass_manager.run_passes(fuse);
    pass_manager.run_passes(no_fuse);
    EXPECT_EQ(1, count_ops_of_type<op::UpdateSlice>(fuse));
    EXPECT_EQ(0, count_ops_of_type<op::UpdateSlice>(no_fuse));

    auto int_f = make_function();
    auto cpu_f = make_function();

    test::Uniform<float> rng(0.0f, 1.0f);
    vector<vector<float>> args;
    for (shared_ptr<op::v0::Parameter> param : int_f->get_parameters())
    {
        vector<float> tensor_val(shape_size(param->get_output_shape(0)));
        rng.initialize(tensor_val);
        args.push_back(tensor_val);
    }
    auto int_results = execute(int_f, args, "INTERPRETER");
    auto cpu_results = execute(cpu_f, args, "${BACKEND_NAME}");
    for (size_t i = 0; i < cpu_results.size(); i++)
    {
        EXPECT_TRUE(test::all_close(cpu_results.at(i), int_results.at(i)));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, cpu_fusion_dot_batch_forward)
{
    const Shape shape_a{2, 3, 2};
    const Shape shape_b{2, 3};

    auto generate_func = [&shape_a, &shape_b]() -> shared_ptr<Function> {
        auto a = make_shared<op::v0::Parameter>(element::f32, shape_a);
        auto b = make_shared<op::v0::Parameter>(element::f32, shape_b);
        auto dot = make_shared<op::v0::Dot>(a, b);
        return make_shared<Function>(dot, ParameterVector{a, b});
    };
    shared_ptr<Function> cpu_func = generate_func();
    shared_ptr<Function> int_func = generate_func();

    test::Uniform<float> rng(0.0f, 1.0f);
    vector<vector<float>> args;
    for (shared_ptr<op::v0::Parameter> param : int_func->get_parameters())
    {
        vector<float> tensor_val(shape_size(param->get_output_shape(0)));
        rng.initialize(tensor_val);
        args.push_back(tensor_val);
    }

    auto int_results = execute(int_func, args, "INTERPRETER");
    auto cpu_results = execute(cpu_func, args, "${BACKEND_NAME}");
    for (size_t i = 0; i < cpu_results.size(); i++)
    {
        EXPECT_TRUE(test::all_close(cpu_results.at(i), int_results.at(i), 1.0e-4f, 1.0e-4f));
    }
}

namespace
{
    static std::shared_ptr<Function>
        create_rnn_input_linear_transformation_function(size_t num_timesteps,
                                                        bool data_is_4d = false)
    {
        auto W = std::make_shared<op::v0::Parameter>(element::f32, Shape{400, 50});
        auto bias = std::make_shared<op::v0::Parameter>(element::f32, Shape{400});
        ParameterVector params{W, bias};
        auto create_graph = [&]() -> std::shared_ptr<Node> {
            auto data_param =
                (data_is_4d) ? std::make_shared<op::v0::Parameter>(element::f32, Shape{2, 5, 1, 50})
                             : std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 1, 50});
            params.push_back(data_param);
            auto reshape_axis_order = data_is_4d ? AxisVector{0, 1, 2, 3} : AxisVector{0, 1, 2};
            auto data_param_reshape =
                std::make_shared<op::v0::Reshape>(data_param, reshape_axis_order, Shape{10, 50});
            auto W_reshape = std::make_shared<op::v0::Reshape>(W, AxisVector{1, 0}, Shape{50, 400});
            auto dot = std::make_shared<op::v0::Dot>(data_param_reshape, W_reshape);
            auto bias_broadcast =
                make_shared<op::v0::Broadcast>(bias, dot->get_output_shape(0), AxisSet{0});
            auto add_bias = std::make_shared<op::v1::Add>(dot, bias_broadcast);
            return move(add_bias);
        };

        OutputVector graph_nodes;
        for (size_t i = 0; i < num_timesteps; i++)
        {
            graph_nodes.push_back(create_graph());
        }
        auto concat = std::make_shared<op::v0::Concat>(graph_nodes, 0);
        return make_shared<Function>(OutputVector{concat}, params);
    }
}

NGRAPH_TEST(${BACKEND_NAME}, cpu_fusion_rnn_input_fusion_inter_vs_cpu)
{
    shared_ptr<Function> cpu_func = create_rnn_input_linear_transformation_function(10);
    shared_ptr<Function> int_func = create_rnn_input_linear_transformation_function(10);

    test::Uniform<float> rng(-10.0f, 10.0f);
    vector<vector<float>> args;
    for (shared_ptr<op::v0::Parameter> param : int_func->get_parameters())
    {
        vector<float> tensor_val(shape_size(param->get_output_shape(0)));
        rng.initialize(tensor_val);
        args.push_back(tensor_val);
    }

    auto int_results = execute(int_func, args, "INTERPRETER");
    auto cpu_results = execute(cpu_func, args, "${BACKEND_NAME}");
    for (size_t i = 0; i < cpu_results.size(); i++)
    {
        EXPECT_TRUE(test::all_close(cpu_results.at(i), int_results.at(i), 1.0e-4f, 1.0e-4f));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, cpu_quant_fusion_qconv_relu)
{
    auto make_function = []() {
        Shape shape_input{1, 2, 2, 2};
        Shape shape_weights{1, 2, 1, 1};
        auto input = std::make_shared<op::v0::Parameter>(element::f32, shape_input);
        auto weights = std::make_shared<op::v0::Parameter>(element::f32, shape_weights);
        auto input_scale = op::v0::Constant::create(element::f32, Shape{}, {2.0f});
        auto weights_scale = op::v0::Constant::create(element::f32, Shape{}, {2.0f});
        auto output_scale = op::v0::Constant::create(element::f32, Shape{}, {4.0f});
        auto int8_zero = op::v0::Constant::create(element::i8, Shape{}, {0});
        auto uint8_zero = op::v0::Constant::create(element::u8, Shape{}, {0});

        op::v0::Quantize::RoundMode round_mode =
            op::v0::Quantize::RoundMode::ROUND_NEAREST_TOWARD_EVEN;
        auto q_input = std::make_shared<op::v0::Quantize>(
            input, input_scale, uint8_zero, element::u8, AxisSet{}, round_mode);
        auto q_weights = std::make_shared<op::v0::Quantize>(
            weights, weights_scale, int8_zero, element::i8, AxisSet{}, round_mode);
        auto conv = std::make_shared<op::v0::QuantizedConvolution>(q_input,
                                                                   q_weights,
                                                                   Strides{1, 1},
                                                                   Strides{1, 1},
                                                                   CoordinateDiff{0, 0},
                                                                   CoordinateDiff{0, 0},
                                                                   Strides{1, 1},
                                                                   input_scale,
                                                                   uint8_zero,
                                                                   weights_scale,
                                                                   int8_zero,
                                                                   output_scale,
                                                                   int8_zero,
                                                                   element::i8,
                                                                   AxisSet{},
                                                                   AxisSet{},
                                                                   AxisSet{});
        auto dq = std::make_shared<op::v0::Dequantize>(
            conv, output_scale, int8_zero, element::f32, AxisSet{});
        auto relu = std::make_shared<op::v0::Relu>(dq);
        auto q = std::make_shared<op::v0::Quantize>(
            relu, output_scale, uint8_zero, element::u8, AxisSet{}, round_mode);
        auto q_f = std::make_shared<op::v0::Dequantize>(
            q, output_scale, uint8_zero, element::f32, AxisSet{});
        return make_shared<Function>(OutputVector{q_f}, ParameterVector{input, weights});
    };

    auto cpu_f1 = make_function();
    auto cpu_f2 = make_function();

    test::Uniform<float> rng(2.0f, 2.0f);
    vector<vector<float>> args;
    for (shared_ptr<op::v0::Parameter> param : cpu_f1->get_parameters())
    {
        vector<float> tensor_val(shape_size(param->get_output_shape(0)));
        rng.initialize(tensor_val);
        args.push_back(tensor_val);
    }

    set_environment("NGRAPH_PASS_ENABLES", "CPUQuantFusion:0", 1);
    auto cpu1_results = execute(cpu_f1, args, "${BACKEND_NAME}");
    set_environment("NGRAPH_PASS_ENABLES", "CPUQuantFusion:1", 1);
    auto cpu2_results = execute(cpu_f2, args, "${BACKEND_NAME}");
    // Expected output - [2, 2, ...]
    EXPECT_TRUE(test::all_close(cpu1_results.at(0), cpu2_results.at(0)));
}

NGRAPH_TEST(${BACKEND_NAME}, cpu_quant_fusion_qconvb_relu)
{
    auto make_function = []() {
        Shape shape_input{1, 2, 2, 2};
        Shape shape_weights{1, 2, 1, 1};
        auto input = std::make_shared<op::v0::Parameter>(element::f32, shape_input);
        auto weights = std::make_shared<op::v0::Parameter>(element::f32, shape_weights);
        auto bias = std::make_shared<op::v0::Parameter>(element::f32, Shape{shape_weights[0]});
        auto input_scale = op::v0::Constant::create(element::f32, Shape{}, {2.0f});
        auto weights_scale = op::v0::Constant::create(element::f32, Shape{}, {2.0f});
        auto output_scale = op::v0::Constant::create(element::f32, Shape{}, {4.0f});
        auto int8_zero = op::v0::Constant::create(element::i8, Shape{}, {0});
        auto int32_zero = op::v0::Constant::create(element::i32, Shape{}, {0});
        auto uint8_zero = op::v0::Constant::create(element::u8, Shape{}, {0});

        op::v0::Quantize::RoundMode round_mode =
            op::v0::Quantize::RoundMode::ROUND_NEAREST_TOWARD_EVEN;
        auto q_input = std::make_shared<op::v0::Quantize>(
            input, input_scale, uint8_zero, element::u8, AxisSet{}, round_mode);
        auto q_weights = std::make_shared<op::v0::Quantize>(
            weights, weights_scale, int8_zero, element::i8, AxisSet{}, round_mode);
        auto q_bias = std::make_shared<op::v0::Quantize>(
            bias, input_scale * weights_scale, int32_zero, element::i32, AxisSet{}, round_mode);
        auto requant_scale = (input_scale * weights_scale) / output_scale;
        auto conv = std::make_shared<op::v0::QuantizedConvolutionBias>(q_input,
                                                                       q_weights,
                                                                       bias,
                                                                       Strides{1, 1},
                                                                       Strides{1, 1},
                                                                       CoordinateDiff{0, 0},
                                                                       CoordinateDiff{0, 0},
                                                                       Strides{1, 1},
                                                                       requant_scale);
        auto dq = std::make_shared<op::v0::Dequantize>(
            conv, output_scale, int8_zero, element::f32, AxisSet{});
        auto relu = std::make_shared<op::v0::Relu>(dq);
        auto q = std::make_shared<op::v0::Quantize>(
            relu, output_scale, uint8_zero, element::u8, AxisSet{}, round_mode);
        auto q_f = std::make_shared<op::v0::Dequantize>(
            q, output_scale, uint8_zero, element::f32, AxisSet{});
        return make_shared<Function>(OutputVector{q_f}, ParameterVector{input, weights, bias});
    };

    auto cpu_f1 = make_function();
    auto cpu_f2 = make_function();

    test::Uniform<float> rng(2.0f, 2.0f);
    vector<vector<float>> args;
    for (shared_ptr<op::v0::Parameter> param : cpu_f1->get_parameters())
    {
        vector<float> tensor_val(shape_size(param->get_output_shape(0)));
        rng.initialize(tensor_val);
        args.push_back(tensor_val);
    }
    set_environment("NGRAPH_PASS_ENABLES", "CPUQuantFusion:0", 1);
    auto cpu1_results = execute(cpu_f1, args, "${BACKEND_NAME}");
    set_environment("NGRAPH_PASS_ENABLES", "CPUQuantFusion:1", 1);
    auto cpu2_results = execute(cpu_f2, args, "${BACKEND_NAME}");
    EXPECT_TRUE(test::all_close(cpu1_results.at(0), cpu2_results.at(0)));
}

NGRAPH_TEST(${BACKEND_NAME}, cpu_quant_fusion_qavg_pool)
{
    auto make_function = []() {
        Shape shape_input{1, 2, 4, 4};
        auto input = std::make_shared<op::v0::Parameter>(element::f32, shape_input);
        auto input_scale = op::v0::Constant::create(element::f32, Shape{}, {2.0f});
        auto weights_scale = op::v0::Constant::create(element::f32, Shape{}, {2.0f});
        auto int8_zero = op::v0::Constant::create(element::i8, Shape{}, {0});
        auto uint8_zero = op::v0::Constant::create(element::u8, Shape{}, {0});

        op::v0::Quantize::RoundMode round_mode =
            op::v0::Quantize::RoundMode::ROUND_NEAREST_TOWARD_EVEN;
        auto q_input = std::make_shared<op::v0::Quantize>(
            input, input_scale, uint8_zero, element::u8, AxisSet{}, round_mode);
        auto dq = std::make_shared<op::v0::Dequantize>(
            q_input, input_scale, uint8_zero, element::f32, AxisSet{});
        auto avg_pool = std::make_shared<op::v0::AvgPool>(dq, Shape{2, 2});
        return make_shared<Function>(OutputVector{avg_pool}, ParameterVector{input});
    };

    auto cpu_f1 = make_function();
    auto cpu_f2 = make_function();

    test::Uniform<float> rng(4.0f, 4.0f);
    vector<vector<float>> args;
    for (shared_ptr<op::v0::Parameter> param : cpu_f1->get_parameters())
    {
        vector<float> tensor_val(shape_size(param->get_output_shape(0)));
        rng.initialize(tensor_val);
        args.push_back(tensor_val);
    }

    set_environment("NGRAPH_PASS_ENABLES", "CPUQuantFusion:0", 1);
    auto cpu1_results = execute(cpu_f1, args, "${BACKEND_NAME}");
    set_environment("NGRAPH_PASS_ENABLES", "CPUQuantFusion:1", 1);
    auto cpu2_results = execute(cpu_f2, args, "${BACKEND_NAME}");
    EXPECT_TRUE(test::all_close(cpu1_results.at(0), cpu2_results.at(0)));
}

NGRAPH_TEST(${BACKEND_NAME}, cpu_quant_fusion_qmax_pool)
{
    auto make_function = []() {
        Shape shape_input{1, 2, 4, 4};
        auto input = std::make_shared<op::v0::Parameter>(element::f32, shape_input);
        auto input_scale = op::v0::Constant::create(element::f32, Shape{}, {2.0f});
        auto weights_scale = op::v0::Constant::create(element::f32, Shape{}, {2.0f});
        auto int8_zero = op::v0::Constant::create(element::i8, Shape{}, {0});
        auto uint8_zero = op::v0::Constant::create(element::u8, Shape{}, {0});

        op::v0::Quantize::RoundMode round_mode =
            op::v0::Quantize::RoundMode::ROUND_NEAREST_TOWARD_EVEN;
        auto q_input = std::make_shared<op::v0::Quantize>(
            input, input_scale, uint8_zero, element::u8, AxisSet{}, round_mode);
        auto dq = std::make_shared<op::v0::Dequantize>(
            q_input, input_scale, uint8_zero, element::f32, AxisSet{});
        auto maxpool = std::make_shared<op::v0::MaxPool>(dq, Shape{2, 2});
        return make_shared<Function>(OutputVector{maxpool}, ParameterVector{input});
    };

    auto cpu_f1 = make_function();
    auto cpu_f2 = make_function();

    test::Uniform<float> rng(1.0f, 10.0f);
    vector<vector<float>> args;
    for (shared_ptr<op::v0::Parameter> param : cpu_f1->get_parameters())
    {
        vector<float> tensor_val(shape_size(param->get_output_shape(0)));
        rng.initialize(tensor_val);
        args.push_back(tensor_val);
    }

    set_environment("NGRAPH_PASS_ENABLES", "CPUQuantFusion:0", 1);
    auto cpu1_results = execute(cpu_f1, args, "${BACKEND_NAME}");
    set_environment("NGRAPH_PASS_ENABLES", "CPUQuantFusion:1", 1);
    auto cpu2_results = execute(cpu_f2, args, "${BACKEND_NAME}");
    EXPECT_TRUE(test::all_close(cpu1_results.at(0), cpu2_results.at(0)));
}

NGRAPH_TEST(${BACKEND_NAME}, cpu_quant_fusion_qconcat)
{
    auto make_function = []() {
        auto get_input_slice = [](std::shared_ptr<op::v0::Parameter>& input) {
            auto input_scale = op::v0::Constant::create(element::f32, Shape{}, {2.0f});
            auto int8_zero = op::v0::Constant::create(element::i8, Shape{}, {0});
            auto uint8_zero = op::v0::Constant::create(element::u8, Shape{}, {0});

            op::v0::Quantize::RoundMode round_mode =
                op::v0::Quantize::RoundMode::ROUND_NEAREST_TOWARD_EVEN;
            auto q_input = std::make_shared<op::v0::Quantize>(
                input, input_scale, uint8_zero, element::u8, AxisSet{}, round_mode);
            auto dq = std::make_shared<op::v0::Dequantize>(
                q_input, input_scale, uint8_zero, element::f32, AxisSet{});
            return dq;
        };

        OutputVector concat_inputs;
        OutputVector concats;
        ParameterVector inputs;
        Shape shape_input{1, 2, 4, 4};
        inputs.push_back(std::make_shared<op::v0::Parameter>(element::f32, shape_input));
        concat_inputs.push_back(get_input_slice(inputs.back()));
        // Concat2  -- Concat7
        for (size_t i = 0; i < 6; i++)
        {
            inputs.push_back(std::make_shared<op::v0::Parameter>(element::f32, shape_input));
            concat_inputs.push_back(get_input_slice(inputs.back()));
            concats.push_back(std::make_shared<op::v0::Concat>(concat_inputs, 0));
        }
        return make_shared<Function>(concats, inputs);
    };

    auto cpu_f1 = make_function();
    auto cpu_f2 = make_function();

    test::Uniform<float> rng(2.0f, 2.0f);
    vector<vector<float>> args;
    for (shared_ptr<op::v0::Parameter> param : cpu_f1->get_parameters())
    {
        vector<float> tensor_val(shape_size(param->get_output_shape(0)));
        rng.initialize(tensor_val);
        args.push_back(tensor_val);
    }

    set_environment("NGRAPH_PASS_ENABLES", "CPUQuantFusion:0", 1);
    auto cpu1_results = execute(cpu_f1, args, "${BACKEND_NAME}");
    set_environment("NGRAPH_PASS_ENABLES", "CPUQuantFusion:1", 1);
    auto cpu2_results = execute(cpu_f2, args, "${BACKEND_NAME}");
    // Expect Concat2 -- Concat6 to be fused and not Concat7
    ASSERT_EQ(count_ops_of_type<op::v0::Concat>(cpu_f2), 6);
    EXPECT_TRUE(test::all_close(cpu1_results.at(0), cpu2_results.at(0)));
}

NGRAPH_TEST(${BACKEND_NAME}, cpu_quant_fusion_dq_q)
{
    auto make_function = [](bool match_scales = true, bool match_et = true) {
        Shape shape_input{1, 2, 2};
        auto input = std::make_shared<op::v0::Parameter>(element::i8, shape_input);
        auto dq_scale = op::v0::Constant::create(element::f32, Shape{}, {2.0f});
        auto int8_zero = op::v0::Constant::create(element::i8, Shape{}, {0});
        auto dq = std::make_shared<op::v0::Dequantize>(
            input, dq_scale, int8_zero, element::f32, AxisSet{});
        float q_scalev = 2.0f;
        if (!match_scales)
        {
            q_scalev = 1.0f;
        }
        auto q_scale = op::v0::Constant::create(element::f32, Shape{}, {q_scalev});
        op::v0::Quantize::RoundMode round_mode =
            op::v0::Quantize::RoundMode::ROUND_NEAREST_TOWARD_EVEN;
        if (match_et)
        {
            auto q = std::make_shared<op::v0::Quantize>(
                dq, q_scale, int8_zero, element::i8, AxisSet{}, round_mode);
            return make_shared<Function>(OutputVector{q}, ParameterVector{input});
        }
        else
        {
            auto uint8_zero = op::v0::Constant::create(element::u8, Shape{}, {0});
            auto q = std::make_shared<op::v0::Quantize>(
                dq, q_scale, uint8_zero, element::u8, AxisSet{}, round_mode);
            return make_shared<Function>(OutputVector{q}, ParameterVector{input});
        }
    };

    auto cpu_f1 = make_function();
    auto cpu_f2 = make_function();

    vector<vector<int8_t>> args;
    args.push_back({-1, 2, 3, 4});

    set_environment("NGRAPH_PASS_ENABLES", "CPUQuantFusion:0", 1);
    auto cpu1_results = execute(cpu_f1, args, "${BACKEND_NAME}");
    set_environment("NGRAPH_PASS_ENABLES", "CPUQuantFusion:1", 1);
    auto cpu2_results = execute(cpu_f2, args, "${BACKEND_NAME}");
    EXPECT_TRUE(test::all_close(cpu1_results.at(0), cpu2_results.at(0)));

    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    auto fuse = make_function(true, true);
    auto no_fuse1 = make_function(false, true);
    auto no_fuse2 = make_function(true, false);
    backend->compile(fuse);
    backend->compile(no_fuse1);
    backend->compile(no_fuse2);
    ASSERT_EQ(count_ops_of_type<op::v0::Quantize>(fuse), 0);
    ASSERT_EQ(count_ops_of_type<op::v0::Quantize>(no_fuse1), 1);
    ASSERT_EQ(count_ops_of_type<op::v0::Quantize>(no_fuse2), 1);
}

NGRAPH_TEST(${BACKEND_NAME}, cpu_quant_fusion_qconvbsa)
{
    auto make_function = []() {
        Shape shape_input{1, 2, 2, 2};
        Shape shape_weights{1, 2, 1, 1};
        Shape shape_summand{1, 1, 2, 2};
        auto input = std::make_shared<op::v0::Parameter>(element::f32, shape_input);
        auto weights = std::make_shared<op::v0::Parameter>(element::f32, shape_weights);
        auto bias = std::make_shared<op::v0::Parameter>(element::f32, Shape{shape_weights[0]});
        auto summand = std::make_shared<op::v0::Parameter>(element::f32, shape_summand);

        auto input_scale = op::v0::Constant::create(element::f32, Shape{}, {2.0f});
        auto weights_scale = op::v0::Constant::create(element::f32, Shape{}, {2.0f});
        auto output_scale = op::v0::Constant::create(element::f32, Shape{}, {4.0f});
        auto summand_scale = op::v0::Constant::create(element::f32, Shape{}, {2.0f});

        auto int8_zero = op::v0::Constant::create(element::i8, Shape{}, {0});
        auto int32_zero = op::v0::Constant::create(element::i32, Shape{}, {0});
        auto uint8_zero = op::v0::Constant::create(element::u8, Shape{}, {0});

        op::v0::Quantize::RoundMode round_mode =
            op::v0::Quantize::RoundMode::ROUND_NEAREST_TOWARD_EVEN;
        auto q_input = std::make_shared<op::v0::Quantize>(
            input, input_scale, uint8_zero, element::u8, AxisSet{}, round_mode);
        auto q_weights = std::make_shared<op::v0::Quantize>(
            weights, weights_scale, int8_zero, element::i8, AxisSet{}, round_mode);
        auto q_bias = std::make_shared<op::v0::Quantize>(
            bias, input_scale * weights_scale, int32_zero, element::i32, AxisSet{}, round_mode);
        auto q_summand = std::make_shared<op::v0::Quantize>(
            summand, summand_scale, int8_zero, element::i8, AxisSet{}, round_mode);

        // Left Graph
        auto requant_scale = (input_scale * weights_scale) / output_scale;
        auto conv = std::make_shared<op::v0::QuantizedConvolutionBias>(q_input,
                                                                       q_weights,
                                                                       bias,
                                                                       Strides{1, 1},
                                                                       Strides{1, 1},
                                                                       CoordinateDiff{0, 0},
                                                                       CoordinateDiff{0, 0},
                                                                       Strides{1, 1},
                                                                       requant_scale);
        auto dq_l = std::make_shared<op::v0::Dequantize>(
            conv, output_scale, int8_zero, element::f32, AxisSet{});
        auto r_l = std::make_shared<op::v0::Reshape>(dq_l, AxisVector{0, 1, 2, 3}, Shape{1, 2, 2});
        auto b_l = std::make_shared<op::v0::Broadcast>(r_l, Shape{1, 1, 2, 2}, AxisSet{0});

        // Right Graph
        auto dq_r = std::make_shared<op::v0::Dequantize>(
            q_summand, summand_scale, int8_zero, element::f32, AxisSet{});
        auto r_r = std::make_shared<op::v0::Reshape>(dq_r, AxisVector{0, 1, 2, 3}, Shape{1, 2, 2});
        auto b_r = std::make_shared<op::v0::Broadcast>(r_r, Shape{1, 1, 2, 2}, AxisSet{0});
        auto add = b_l + b_r;
        auto relu = std::make_shared<op::v0::Relu>(add);
        return make_shared<Function>(OutputVector{relu},
                                     ParameterVector{input, weights, bias, summand});
    };

    auto cpu_f1 = make_function();
    auto cpu_f2 = make_function();

    test::Uniform<float> rng(4.0f, 4.0f);
    vector<vector<float>> args;
    for (shared_ptr<op::v0::Parameter> param : cpu_f1->get_parameters())
    {
        vector<float> tensor_val(shape_size(param->get_output_shape(0)));
        rng.initialize(tensor_val);
        args.push_back(tensor_val);
    }

    // Disable CPUQuantFusion
    set_environment("NGRAPH_PASS_ENABLES", "CPUQuantFusion:0", 1);
    auto cpu1_results = execute(cpu_f1, args, "${BACKEND_NAME}");
    // Enable CPUQuantFusion
    set_environment("NGRAPH_PASS_ENABLES", "CPUQuantFusion:1", 1);
    auto cpu2_results = execute(cpu_f2, args, "${BACKEND_NAME}");
    EXPECT_TRUE(test::all_close(cpu1_results.at(0), cpu2_results.at(0)));
}

NGRAPH_TEST(${BACKEND_NAME}, cpu_quant_fusion_qconvba)
{
    auto make_function = []() {
        Shape shape_input{1, 2, 2, 2};
        Shape shape_weights{1, 2, 1, 1};
        Shape shape_summand{1, 1, 2, 2};
        auto input = std::make_shared<op::v0::Parameter>(element::f32, shape_input);
        auto weights = std::make_shared<op::v0::Parameter>(element::f32, shape_weights);
        auto bias = std::make_shared<op::v0::Parameter>(element::f32, Shape{shape_weights[0]});
        auto summand = std::make_shared<op::v0::Parameter>(element::f32, shape_summand);

        auto input_scale = op::v0::Constant::create(element::f32, Shape{}, {2.0f});
        auto weights_scale = op::v0::Constant::create(element::f32, Shape{}, {2.0f});
        auto output_scale = op::v0::Constant::create(element::f32, Shape{}, {4.0f});
        auto summand_scale = op::v0::Constant::create(element::f32, Shape{}, {4.0f});

        auto int8_zero = op::v0::Constant::create(element::i8, Shape{}, {0});
        auto int32_zero = op::v0::Constant::create(element::i32, Shape{}, {0});
        auto uint8_zero = op::v0::Constant::create(element::u8, Shape{}, {0});

        op::v0::Quantize::RoundMode round_mode =
            op::v0::Quantize::RoundMode::ROUND_NEAREST_TOWARD_EVEN;
        auto q_input = std::make_shared<op::v0::Quantize>(
            input, input_scale, uint8_zero, element::u8, AxisSet{}, round_mode);
        auto q_weights = std::make_shared<op::v0::Quantize>(
            weights, weights_scale, int8_zero, element::i8, AxisSet{}, round_mode);
        auto q_bias = std::make_shared<op::v0::Quantize>(
            bias, input_scale * weights_scale, int32_zero, element::i32, AxisSet{}, round_mode);
        auto q_summand = std::make_shared<op::v0::Quantize>(
            summand, summand_scale, uint8_zero, element::u8, AxisSet{}, round_mode);

        // Left Graph
        auto requant_scale = (input_scale * weights_scale) / output_scale;
        auto conv = std::make_shared<op::v0::QuantizedConvolutionBias>(q_input,
                                                                       q_weights,
                                                                       bias,
                                                                       Strides{1, 1},
                                                                       Strides{1, 1},
                                                                       CoordinateDiff{0, 0},
                                                                       CoordinateDiff{0, 0},
                                                                       Strides{1, 1},
                                                                       requant_scale);
        auto dq_l = std::make_shared<op::v0::Dequantize>(
            conv, output_scale, int8_zero, element::f32, AxisSet{});
        auto r_l = std::make_shared<op::v0::Reshape>(dq_l, AxisVector{0, 1, 2, 3}, Shape{1, 2, 2});
        auto b_l = std::make_shared<op::v0::Broadcast>(r_l, Shape{1, 1, 2, 2}, AxisSet{0});

        // Right Graph
        auto dq_r = std::make_shared<op::v0::Dequantize>(
            q_summand, summand_scale, uint8_zero, element::f32, AxisSet{});
        auto r_r = std::make_shared<op::v0::Reshape>(dq_r, AxisVector{0, 1, 2, 3}, Shape{1, 2, 2});
        auto b_r = std::make_shared<op::v0::Broadcast>(r_r, Shape{1, 1, 2, 2}, AxisSet{0});
        auto add = b_l + b_r;
        auto relu = std::make_shared<op::v0::Relu>(add);
        return make_shared<Function>(OutputVector{relu},
                                     ParameterVector{input, weights, bias, summand});
    };

    auto cpu_f1 = make_function();
    auto cpu_f2 = make_function();

    test::Uniform<float> rng(2.0f, 2.0f);
    vector<vector<float>> args;
    for (shared_ptr<op::v0::Parameter> param : cpu_f1->get_parameters())
    {
        vector<float> tensor_val(shape_size(param->get_output_shape(0)));
        rng.initialize(tensor_val);
        args.push_back(tensor_val);
    }

    // Disable CPUQuantFusion
    set_environment("NGRAPH_PASS_ENABLES", "CPUQuantFusion:0", 1);
    auto cpu1_results = execute(cpu_f1, args, "${BACKEND_NAME}");
    // Enable CPUQuantFusion
    set_environment("NGRAPH_PASS_ENABLES", "CPUQuantFusion:1", 1);
    auto cpu2_results = execute(cpu_f2, args, "${BACKEND_NAME}");
    EXPECT_TRUE(test::all_close(cpu1_results.at(0), cpu2_results.at(0)));
}

NGRAPH_TEST(${BACKEND_NAME}, cpu_quant_fusion_qconvba_q)
{
    auto make_function = []() {
        Shape shape_input{1, 2, 2, 2};
        Shape shape_weights{1, 2, 1, 1};
        Shape shape_summand{1, 1, 2, 2};
        auto input_l = std::make_shared<op::v0::Parameter>(element::f32, shape_input);
        auto weights_l = std::make_shared<op::v0::Parameter>(element::f32, shape_weights);
        auto bias_l = std::make_shared<op::v0::Parameter>(element::f32, Shape{shape_weights[0]});
        auto input_r = std::make_shared<op::v0::Parameter>(element::f32, shape_input);
        auto weights_r = std::make_shared<op::v0::Parameter>(element::f32, shape_weights);
        auto bias_r = std::make_shared<op::v0::Parameter>(element::f32, Shape{shape_weights[0]});

        auto input_scale_l = op::v0::Constant::create(element::f32, Shape{}, {2.0f});
        auto weights_scale_l = op::v0::Constant::create(element::f32, Shape{}, {2.0f});
        auto output_scale_l = op::v0::Constant::create(element::f32, Shape{}, {4.0f});
        auto input_scale_r = op::v0::Constant::create(element::f32, Shape{}, {5.0f});
        auto weights_scale_r = op::v0::Constant::create(element::f32, Shape{}, {5.0f});
        auto output_scale_r = op::v0::Constant::create(element::f32, Shape{}, {20.0f});

        auto int8_zero = op::v0::Constant::create(element::i8, Shape{}, {0});
        auto int32_zero = op::v0::Constant::create(element::i32, Shape{}, {0});
        auto uint8_zero = op::v0::Constant::create(element::u8, Shape{}, {0});

        op::v0::Quantize::RoundMode round_mode =
            op::v0::Quantize::RoundMode::ROUND_NEAREST_TOWARD_EVEN;
        auto q_input_l = std::make_shared<op::v0::Quantize>(
            input_l, input_scale_l, uint8_zero, element::u8, AxisSet{}, round_mode);
        auto q_weights_l = std::make_shared<op::v0::Quantize>(
            weights_l, weights_scale_l, int8_zero, element::i8, AxisSet{}, round_mode);
        auto q_bias_l = std::make_shared<op::v0::Quantize>(bias_l,
                                                           input_scale_l * weights_scale_l,
                                                           int32_zero,
                                                           element::i32,
                                                           AxisSet{},
                                                           round_mode);
        auto q_input_r = std::make_shared<op::v0::Quantize>(
            input_r, input_scale_r, uint8_zero, element::u8, AxisSet{}, round_mode);
        auto q_weights_r = std::make_shared<op::v0::Quantize>(
            weights_r, weights_scale_r, int8_zero, element::i8, AxisSet{}, round_mode);
        auto q_bias_r = std::make_shared<op::v0::Quantize>(bias_r,
                                                           input_scale_r * weights_scale_r,
                                                           int32_zero,
                                                           element::i32,
                                                           AxisSet{},
                                                           round_mode);

        // Left Graph
        auto requant_scale_l = (input_scale_l * weights_scale_l) / output_scale_l;
        auto conv_l = std::make_shared<op::v0::QuantizedConvolutionBias>(q_input_l,
                                                                         q_weights_l,
                                                                         q_bias_l,
                                                                         Strides{1, 1},
                                                                         Strides{1, 1},
                                                                         CoordinateDiff{0, 0},
                                                                         CoordinateDiff{0, 0},
                                                                         Strides{1, 1},
                                                                         requant_scale_l);
        auto dq_l = std::make_shared<op::v0::Dequantize>(
            conv_l, output_scale_l, int8_zero, element::f32, AxisSet{});
        auto r_l = std::make_shared<op::v0::Reshape>(dq_l, AxisVector{0, 1, 2, 3}, Shape{1, 2, 2});
        auto b_l = std::make_shared<op::v0::Broadcast>(r_l, Shape{1, 1, 2, 2}, AxisSet{0});

        // Right Graph
        auto requant_scale_r = (input_scale_r * weights_scale_r) / output_scale_r;
        auto conv_r = std::make_shared<op::v0::QuantizedConvolutionBias>(q_input_r,
                                                                         q_weights_r,
                                                                         q_bias_r,
                                                                         Strides{1, 1},
                                                                         Strides{1, 1},
                                                                         CoordinateDiff{0, 0},
                                                                         CoordinateDiff{0, 0},
                                                                         Strides{1, 1},
                                                                         requant_scale_r);
        auto dq_r = std::make_shared<op::v0::Dequantize>(
            conv_r, output_scale_r, int8_zero, element::f32, AxisSet{});
        auto r_r = std::make_shared<op::v0::Reshape>(dq_r, AxisVector{0, 1, 2, 3}, Shape{1, 2, 2});
        auto b_r = std::make_shared<op::v0::Broadcast>(r_r, Shape{1, 1, 2, 2}, AxisSet{0});
        auto add = b_l + b_r;
        auto relu = std::make_shared<op::v0::Relu>(add);
        auto q = std::make_shared<op::v0::Quantize>(
            relu, output_scale_r, uint8_zero, element::u8, AxisSet{}, round_mode);
        auto dq = std::make_shared<op::v0::Dequantize>(
            q, output_scale_r, uint8_zero, element::f32, AxisSet{});
        return make_shared<Function>(
            OutputVector{dq},
            ParameterVector{input_l, weights_l, bias_l, input_r, weights_r, bias_r});
    };

    auto cpu_f1 = make_function();
    auto cpu_f2 = make_function();

    test::Uniform<float> rng(2.0f, 2.0f);
    vector<vector<float>> args;
    for (shared_ptr<op::v0::Parameter> param : cpu_f1->get_parameters())
    {
        vector<float> tensor_val(shape_size(param->get_output_shape(0)));
        rng.initialize(tensor_val);
        args.push_back(tensor_val);
    }

    // Disable CPUQuantFusion
    set_environment("NGRAPH_PASS_ENABLES", "CPUQuantFusion:0", 1);
    auto cpu1_results = execute(cpu_f1, args, "${BACKEND_NAME}");
    // Enable CPUQuantFusion
    set_environment("NGRAPH_PASS_ENABLES", "CPUQuantFusion:1", 1);
    auto cpu2_results = execute(cpu_f2, args, "${BACKEND_NAME}");
    EXPECT_TRUE(test::all_close(cpu1_results.at(0), cpu2_results.at(0)));

    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    auto fuse = make_function();
    backend->compile(fuse);
    ASSERT_EQ(count_ops_of_type<op::v0::Quantize>(fuse), 6);
}

#ifndef NGRAPH_JSON_DISABLE
// Tests that rely on deserializing json files

NGRAPH_TEST(${BACKEND_NAME}, batch_fusion_fuse_batch_dot_backward)
{
    const std::string file_name("mxnet/batch_dot_3.json");
    auto cpu_f = make_function_from_file(file_name);
    auto int_f = make_function_from_file(file_name);

    pass::Manager pass_manager;
    pass_manager.register_pass<ngraph::pass::BatchFusion>();
    pass_manager.run_passes(cpu_f);

    auto int_df = autodiff::backprop_function(int_f);
    auto cpu_df = autodiff::backprop_function(cpu_f);

    test::Uniform<float> rng(-1.0f, 1.0f);
    vector<vector<float>> args;
    for (shared_ptr<op::v0::Parameter> param : cpu_df->get_parameters())
    {
        vector<float> tensor_val(shape_size(param->get_output_shape(0)));
        rng.initialize(tensor_val);
        args.push_back(tensor_val);
    }

    auto int_results = execute(int_df, args, "INTERPRETER");
    auto cpu_results = execute(cpu_df, args, "${BACKEND_NAME}");

    for (size_t i = 0; i < cpu_results.size(); i++)
    {
        EXPECT_TRUE(test::all_close(cpu_results.at(i), int_results.at(i), 1.0e-4f, 1.0e-4f));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, cpu_fusion_fuse_rnn_across_layer_2layer_3timestep)
{
    const std::string file_name("mxnet/2layer_3timestep_ic100oc100.json");
    auto cpu_f = make_function_from_file(file_name);
    auto int_f = make_function_from_file(file_name);
    test::Uniform<float> rng(-1.0f, 1.0f);
    vector<vector<float>> args;

    for (shared_ptr<op::v0::Parameter> param : int_f->get_parameters())
    {
        vector<float> tensor_val(shape_size(param->get_output_shape(0)));
        rng.initialize(tensor_val);
        args.push_back(tensor_val);
    }
    auto int_results = execute(int_f, args, "INTERPRETER");
    auto cpu_results = execute(cpu_f, args, "${BACKEND_NAME}");

    EXPECT_EQ(1, count_ops_of_type<op::Rnn>(cpu_f));
    for (size_t i = 0; i < cpu_results.size(); i++)
    {
        EXPECT_TRUE(test::all_close(cpu_results.at(i), int_results.at(i), 1.0e-4f, 1.0e-4f));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, cpu_fusion_bi_rnn_interpreter_vs_cpu)
{
    const std::string file_name("mxnet/lstm_bi_directional.json");
    auto cpu_f = make_function_from_file(file_name);
    auto int_f = make_function_from_file(file_name);
    test::Uniform<float> rng(0.0f, 1.0f);
    vector<vector<float>> args;

    for (shared_ptr<op::v0::Parameter> param : int_f->get_parameters())
    {
        vector<float> tensor_val(shape_size(param->get_output_shape(0)));
        rng.initialize(tensor_val);
        args.push_back(tensor_val);
    }
    auto int_results = execute(int_f, args, "INTERPRETER");
    auto cpu_results = execute(cpu_f, args, "${BACKEND_NAME}");
    for (size_t i = 0; i < int_results.size(); i++)
    {
        EXPECT_TRUE(test::all_close(cpu_results.at(i), int_results.at(i), 1.0e-4f, 1.0e-4f));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, cpu_fusion_rnn_fusion_1lstm_cell)
{
    const std::string file_name("mxnet/1_lstm_cell_forward.json");
    auto cpu_f = make_function_from_file(file_name);
    auto int_f = make_function_from_file(file_name);
    test::Uniform<float> rng(-1.0f, 1.0f);
    vector<vector<float>> args;

    for (shared_ptr<op::v0::Parameter> param : int_f->get_parameters())
    {
        vector<float> tensor_val(shape_size(param->get_output_shape(0)));
        rng.initialize(tensor_val);
        args.push_back(tensor_val);
    }
    auto int_results = execute(int_f, args, "INTERPRETER");
    auto cpu_results = execute(cpu_f, args, "${BACKEND_NAME}");
    for (size_t i = 0; i < cpu_results.size(); i++)
    {
        EXPECT_TRUE(test::all_close(cpu_results.at(i), int_results.at(i), 1.0e-4f, 1.0e-4f));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, cpu_fusion_rnn_fusion_1rnn_layer_3lstm_cell)
{
    const std::string file_name("mxnet/1rnn_layer_3lstm_cell.json");
    auto cpu_f = make_function_from_file(file_name);
    auto int_f = make_function_from_file(file_name);
    test::Uniform<float> rng(-1.0f, 1.0f);
    vector<vector<float>> args;

    for (shared_ptr<op::v0::Parameter> param : int_f->get_parameters())
    {
        vector<float> tensor_val(shape_size(param->get_output_shape(0)));
        rng.initialize(tensor_val);
        args.push_back(tensor_val);
    }
    auto int_results = execute(int_f, args, "INTERPRETER");
    auto cpu_results = execute(cpu_f, args, "${BACKEND_NAME}");
    for (size_t i = 0; i < cpu_results.size(); i++)
    {
        EXPECT_TRUE(test::all_close(cpu_results.at(i), int_results.at(i), 1.0e-4f, 1.0e-4f));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, cpu_fusion_lstm_cell)
{
    auto make_function = []() {
        const size_t batch_size = 3;
        const size_t input_size = 4;
        const size_t hidden_size = 4;
        const size_t gates_count = 4;

        const auto X = make_shared<op::v0::Parameter>(element::f32, Shape{batch_size, input_size});
        const auto W = make_shared<op::v0::Parameter>(element::f32,
                                                      Shape{gates_count * hidden_size, input_size});
        const auto R = make_shared<op::v0::Parameter>(
            element::f32, Shape{gates_count * hidden_size, hidden_size});
        const auto H_t =
            make_shared<op::v0::Parameter>(element::f32, Shape{batch_size, hidden_size});
        const auto C_t =
            make_shared<op::v0::Parameter>(element::f32, Shape{batch_size, hidden_size});

        const auto lstm_cell = make_shared<op::v0::LSTMCell>(X, H_t, C_t, W, R, hidden_size);
        auto ht = lstm_cell->output(0);
        auto ct = lstm_cell->output(1);

        auto lstm_function = make_shared<Function>(OutputVector{ht, ct},
                                                   ParameterVector{
                                                       X,
                                                       H_t,
                                                       C_t,
                                                       W,
                                                       R,
                                                   });
        return lstm_function;
    };
    auto lstm_function_cpu = make_function();
    auto lstm_function_inter = make_function();
    test::Uniform<float> rng(-1.0f, 1.0f);
    vector<vector<float>> args;

    for (shared_ptr<op::v0::Parameter> param : lstm_function_cpu->get_parameters())
    {
        vector<float> tensor_val(shape_size(param->get_output_shape(0)));
        rng.initialize(tensor_val);
        args.push_back(tensor_val);
    }

    auto int_results = execute(lstm_function_inter, args, "INTERPRETER");
    auto cpu_results = execute(lstm_function_cpu, args, "${BACKEND_NAME}");
    size_t lstm_op_count = count_ops_of_type<op::v0::LSTMCell>(lstm_function_cpu);

    EXPECT_EQ(lstm_op_count, 0);
    for (size_t i = 0; i < cpu_results.size(); i++)
    {
        EXPECT_TRUE(test::all_close(cpu_results.at(i), int_results.at(i), 1.0e-4f, 1.0e-4f));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, cpu_fusion_rnn_fusion_2rnn_layer_3lstm_cell)
{
    const std::string file_name("mxnet/2rnn_layer_3lstm_cell.json");
    auto cpu_f = make_function_from_file(file_name);
    auto int_f = make_function_from_file(file_name);
    test::Uniform<float> rng(-1.0f, 1.0f);
    vector<vector<float>> args;

    for (shared_ptr<op::v0::Parameter> param : int_f->get_parameters())
    {
        vector<float> tensor_val(shape_size(param->get_output_shape(0)));
        rng.initialize(tensor_val);
        args.push_back(tensor_val);
    }
    auto int_results = execute(int_f, args, "INTERPRETER");
    auto cpu_results = execute(cpu_f, args, "${BACKEND_NAME}");
    for (size_t i = 0; i < cpu_results.size(); i++)
    {
        EXPECT_TRUE(test::all_close(cpu_results.at(i), int_results.at(i), 1.0e-4f, 1.0e-4f));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, cpu_fusion_vanilla_rnn_cpu_vs_inter)
{
    const std::string file_name("tensorflow/rnn/vanilla_rnn_3_time_step.json");
    auto cpu_f = make_function_from_file(file_name);
    auto int_f = make_function_from_file(file_name);
    test::Uniform<float> rng(-1.0f, 1.0f);
    vector<vector<float>> args;

    for (shared_ptr<op::v0::Parameter> param : int_f->get_parameters())
    {
        vector<float> tensor_val(shape_size(param->get_output_shape(0)));
        rng.initialize(tensor_val);
        args.push_back(tensor_val);
    }
    auto int_results = execute(int_f, args, "INTERPRETER");
    auto cpu_results = execute(cpu_f, args, "${BACKEND_NAME}");
    for (size_t i = 0; i < cpu_results.size(); i++)
    {
        EXPECT_TRUE(test::all_close(cpu_results.at(i), int_results.at(i), 1.0e-4f, 1.0e-4f));
    }
    auto lstm_ops = get_ops_of_type<op::Rnn>(cpu_f);
    EXPECT_EQ(lstm_ops.size(), 3);
}

NGRAPH_TEST(${BACKEND_NAME}, cpu_fusion_validate_fuse_gru_inputs)
{
    const std::string file_name("mxnet/gru_debug.json");
    auto cpu_func = make_function_from_file(file_name);
    auto int_func = make_function_from_file(file_name);

    test::Uniform<float> rng(-10.0f, 10.0f);
    vector<vector<float>> args;
    for (shared_ptr<op::v0::Parameter> param : int_func->get_parameters())
    {
        vector<float> tensor_val(shape_size(param->get_output_shape(0)));
        rng.initialize(tensor_val);
        args.push_back(tensor_val);
    }

    auto int_results = execute(int_func, args, "INTERPRETER");
    auto cpu_results = execute(cpu_func, args, "${BACKEND_NAME}");
    for (size_t i = 0; i < cpu_results.size(); i++)
    {
        EXPECT_TRUE(test::all_close(cpu_results.at(i), int_results.at(i), 1.0e-4f, 1.0e-4f));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, cpu_fusion_mlir_matmul_bias)
{
    Shape shape{};
    Shape shape_w{2, 4};
    Shape shape_x{4, 1};
    Shape shape_b{1};
    auto A = make_shared<op::v0::Parameter>(element::f32, shape_w);
    auto B = make_shared<op::v0::Parameter>(element::f32, shape_x);
    auto C = make_shared<op::v0::Parameter>(element::f32, shape_b);

    auto dot = make_shared<op::v0::Dot>(A, B);
    auto broadcast = make_shared<op::v0::Broadcast>(C, dot->get_output_shape(0), AxisSet{0});
    auto add = dot + broadcast;

    auto int_func = make_shared<Function>(OutputVector{add}, ParameterVector{A, B, C});
    auto cpu_func = make_shared<Function>(OutputVector{add}, ParameterVector{A, B, C});

    test::Uniform<float> rng(-10.0f, 10.0f);
    vector<vector<float>> args;
    for (shared_ptr<op::v0::Parameter> param : int_func->get_parameters())
    {
        vector<float> tensor_val(shape_size(param->get_output_shape(0)));
        rng.initialize(tensor_val);
        args.push_back(tensor_val);
    }

    auto int_results = execute(cpu_func, args, "INTERPRETER");
    auto cpu_results = execute(cpu_func, args, "${BACKEND_NAME}");
    for (size_t i = 0; i < cpu_results.size(); i++)
    {
        EXPECT_TRUE(test::all_close(cpu_results.at(i), int_results.at(i), 1.0e-4f, 1.0e-4f));
    }
}

#endif
