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

#include <fstream>
#include <sstream>

#include "gtest/gtest.h"
#include "ngraph/frontend/onnx_import/onnx.hpp"
#include "ngraph/ngraph.hpp"
#include "util/all_close_f.hpp"
#include "util/ndarray.hpp"
#include "util/test_tools.hpp"

using namespace ngraph;

TEST(onnx, model_add_abc)
{
    auto model{ngraph::onnx_import::load_onnx_model(
        ngraph::file_util::path_join(SERIALIZED_ZOO, "onnx/add_abc.onnx"))};
    auto backend{ngraph::runtime::Backend::create("INTERPRETER")};

    ngraph::Shape shape{1};
    auto a{backend->create_tensor(ngraph::element::f32, shape)};
    copy_data(a, std::vector<float>{1});
    auto b{backend->create_tensor(ngraph::element::f32, shape)};
    copy_data(b, std::vector<float>{2});
    auto c{backend->create_tensor(ngraph::element::f32, shape)};
    copy_data(c, std::vector<float>{3});

    auto r{backend->create_tensor(ngraph::element::f32, shape)};

    backend->call(model.front(), {r}, {a, b, c});
    EXPECT_TRUE(test::all_close_f((std::vector<float>{6}), read_vector<float>(r)));
}

TEST(onnx, model_add_abc_initializers)
{
    auto model{ngraph::onnx_import::load_onnx_model(
        ngraph::file_util::path_join(SERIALIZED_ZOO, "onnx/add_abc_initializers.onnx"))};
    auto backend{ngraph::runtime::Backend::create("INTERPRETER")};

    ngraph::Shape shape{2, 2};

    auto c{backend->create_tensor(ngraph::element::f32, shape)};
    copy_data(c, std::vector<float>{1, 2, 3, 4});

    auto r{backend->create_tensor(ngraph::element::f32, shape)};

    backend->call(model.front(), {r}, {c});
    EXPECT_TRUE(test::all_close_f((std::vector<float>{3, 6, 9, 12}), read_vector<float>(r)));
}

TEST(onnx, model_split_equal_parts_default)
{
    auto model{ngraph::onnx_import::load_onnx_model(
        ngraph::file_util::path_join(SERIALIZED_ZOO, "onnx/split_equal_parts_default.onnx"))};

    auto args = std::vector<std::vector<float>>{{1, 2, 3, 4, 5, 6}};
    auto expected_output = std::vector<std::vector<float>>{{1, 2}, {3, 4}, {5, 6}};

    for (std::size_t i = 0; i < expected_output.size(); ++i)
    {
        auto result_vectors = execute(model[i], args, "INTERPRETER");
        EXPECT_EQ(result_vectors.size(), 1);
        EXPECT_TRUE(test::all_close_f(expected_output[i], result_vectors.front()));
    }
}

TEST(onnx, model_split_equal_parts_2d)
{
    // Split into 2 equal parts along axis=1
    auto model{ngraph::onnx_import::load_onnx_model(
        ngraph::file_util::path_join(SERIALIZED_ZOO, "onnx/split_equal_parts_2d.onnx"))};

    auto args = std::vector<std::vector<float>>{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}};
    // each output we get as a flattened vector
    auto expected_output =
        std::vector<std::vector<float>>{{0, 1, 2, 6, 7, 8}, {3, 4, 5, 9, 10, 11}};

    for (std::size_t i = 0; i < expected_output.size(); ++i)
    {
        auto result_vectors = execute(model[i], args, "INTERPRETER");
        EXPECT_EQ(result_vectors.size(), 1);
        EXPECT_TRUE(test::all_close_f(expected_output[i], result_vectors[0]));
    }
}

TEST(onnx, model_split_variable_parts_2d)
{
    // Split into variable parts {2, 4} along axis=1
    auto model{ngraph::onnx_import::load_onnx_model(
        ngraph::file_util::path_join(SERIALIZED_ZOO, "onnx/split_variable_parts_2d.onnx"))};

    auto args = std::vector<std::vector<float>>{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}};
    // each output we get as a flattened vector
    auto expected_output =
        std::vector<std::vector<float>>{{0, 1, 6, 7}, {2, 3, 4, 5, 8, 9, 10, 11}};

    for (std::size_t i = 0; i < expected_output.size(); ++i)
    {
        auto result_vectors = execute(model[i], args, "INTERPRETER");
        EXPECT_EQ(result_vectors.size(), 1);
        EXPECT_TRUE(test::all_close_f(expected_output[i], result_vectors[0]));
    }
}

namespace
{
    std::vector<std::vector<float>>
        conv2d_execute(const std::shared_ptr<ngraph::Function>& function)
    {
        std::vector<std::vector<float>> args;

        // data (1, 1, 7, 5) input tensor
        args.emplace_back(ngraph::test::NDArray<float, 4>{{{{{0.f, 1.f, 2.f, 3.f, 4.f},
                                                             {5.f, 6.f, 7.f, 8.f, 9.f},
                                                             {10.f, 11.f, 12.f, 13.f, 14.f},
                                                             {15.f, 16.f, 17.f, 18.f, 19.f},
                                                             {20.f, 21.f, 22.f, 23.f, 24.f},
                                                             {25.f, 26.f, 27.f, 28.f, 29.f},
                                                             {30.f, 31.f, 32.f, 33.f, 34.f}}}}}
                              .get_vector());

        // filters (1, 1, 3, 3) aka convolution weights
        args.emplace_back(
            ngraph::test::NDArray<float, 4>{{{{{1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}}}}}
                .get_vector());

        return execute(function, args, "INTERPRETER");
    }
} // namespace

TEST(onnx, conv2d_strides_padding)
{
    // Convolution with strides=2 and padding=1
    auto function{ngraph::onnx_import::import_onnx_function(
        ngraph::file_util::path_join(SERIALIZED_ZOO, "onnx/conv_with_strides_padding.onnx"))};

    // (1, 1, 4, 3)
    auto expected_output = ngraph::test::NDArray<float, 4>({{{{12.f, 27.f, 24.f},
                                                              {63.f, 108.f, 81.f},
                                                              {123.f, 198.f, 141.f},
                                                              {112.f, 177.f, 124.f}}}})
                               .get_vector();

    auto result{conv2d_execute(function)};
    EXPECT_EQ(expected_output, result.front());
}

TEST(onnx, conv2d_strides_no_padding)
{
    // Convolution with strides=2 and padding=1
    auto function{ngraph::onnx_import::import_onnx_function(
        ngraph::file_util::path_join(SERIALIZED_ZOO, "onnx/conv_with_strides_no_padding.onnx"))};

    // (1, 1, 3, 2)
    auto expected_output =
        ngraph::test::NDArray<float, 4>({{{{54.f, 72.f}, {144.f, 162.f}, {234.f, 252.f}}}})
            .get_vector();

    auto result{conv2d_execute(function)};
    EXPECT_EQ(expected_output, result.front());
}

TEST(onnx, conv2d_strides_assymetric_padding)
{
    // Convolution with strides=2 and padding=1
    auto function{ngraph::onnx_import::import_onnx_function(ngraph::file_util::path_join(
        SERIALIZED_ZOO, "onnx/conv_with_strides_and_assymmetric_padding.onnx"))};

    // (1, 1, 4, 2)
    auto expected_output = ngraph::test::NDArray<float, 4>(
                               {{{{21.f, 33.f}, {99.f, 117.f}, {189.f, 207.f}, {171.f, 183.f}}}})
                               .get_vector();

    auto result{conv2d_execute(function)};
    EXPECT_EQ(expected_output, result.front());
}

TEST(onnx, conv2d_strides_padding_bias)
{
    // Convolution with strides=2 and padding=1 and bias=2
    // auto function{ngraph::onnx_import::import_onnx_function(
    //     ngraph::file_util::path_join(SERIALIZED_ZOO, "onnx/conv_with_strides_padding_bias.onnx"))};

    // std::vector<std::vector<float>> args{m_args};
    // args.emplace_back(std::vector<float>{2.});

    // // (1, 1, 4, 3)
    // auto expected_output =
    //     ngraph::test::NDArray<float, 4>(
    //         {{{{14., 29., 26.}, {65., 110., 83.}, {125., 200., 143.}, {114., 179., 126.}}}})
    //         .get_vector();

    // auto result_vectors = execute(function, args, "CPU");
    // EXPECT_EQ(expected_output, result_vectors.front());
}

TEST(onnx, model_batchnorm_default)
{
    // Batch Normalization with default parameters
    auto function{ngraph::onnx_import::import_onnx_function(
        ngraph::file_util::path_join(SERIALIZED_ZOO, "onnx/batchnorm_default.onnx"))};

    std::vector<std::vector<float>> inputs;

    // input data shape (1, 2, 1, 3)
    inputs.emplace_back(
        ngraph::test::NDArray<float, 4>({{{{-1., 0., 1.}}, {{2., 3., 4.}}}}).get_vector());

    // scale (3)
    inputs.emplace_back(std::vector<float>{1., 1.5});
    // bias (3)
    inputs.emplace_back(std::vector<float>{0., 1.});
    // mean (3)
    inputs.emplace_back(std::vector<float>{0., 3});
    // var (3)
    inputs.emplace_back(std::vector<float>{1., 1.5});

    // shape (1, 2, 1, 3)
    auto expected_output = ngraph::test::NDArray<float, 4>({{{{-0.999995f, 0.f, 0.999995f}},
                                                             {{-0.22474074f, 1.f, 2.2247407f}}}})
                               .get_vector();

    auto result_vectors = execute(function, inputs, "INTERPRETER");
    EXPECT_TRUE(test::all_close_f(expected_output, result_vectors.front()));
}

TEST(onnx, model_relu)
{
    // Simple ReLU test
    auto function{ngraph::onnx_import::import_onnx_function(
        ngraph::file_util::path_join(SERIALIZED_ZOO, "onnx/relu.onnx"))};

    auto inputs = std::vector<std::vector<float>>{{-1, -2, 0, 1, 2, 3}};
    auto expected_output = std::vector<std::vector<float>>{{0, 0, 0, 1, 2, 3}};

    auto result_vectors = execute(function, inputs, "INTERPRETER");
    EXPECT_TRUE(test::all_close_f(expected_output.front(), result_vectors.front()));
}
