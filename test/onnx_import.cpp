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

#include <cstdint>
#include <fstream>
#include <sstream>

#include "gtest/gtest.h"
#include "ngraph/frontend/onnx_import/onnx.hpp"
#include "ngraph/ngraph.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/ndarray.hpp"
#include "util/test_tools.hpp"

using namespace ngraph;

using Inputs = std::vector<std::vector<float>>;
using Outputs = std::vector<std::vector<float>>;
using Model = std::vector<std::shared_ptr<Function>>;

TEST(onnx, model_add_abc)
{
    auto function = onnx_import::import_onnx_function(
        file_util::path_join(SERIALIZED_ZOO, "onnx/add_abc.onnx"));

    Inputs inputs{{1}, {2}, {3}};
    Outputs expected_outputs{{6}};

    Outputs outputs{execute(function, inputs, "INTERPRETER")};
    EXPECT_TRUE(test::all_close_f(expected_outputs.front(), outputs.front()));
}

TEST(onnx, model_add_abc_initializers)
{
    auto function = onnx_import::import_onnx_function(
        file_util::path_join(SERIALIZED_ZOO, "onnx/add_abc_initializers.onnx"));

    Inputs inputs{{1, 2, 3, 4}};
    Outputs expected_outputs{{3, 6, 9, 12}};

    Outputs outputs{execute(function, inputs, "INTERPRETER")};
    EXPECT_TRUE(test::all_close_f(expected_outputs.front(), outputs.front()));
}

TEST(onnx, model_addmul_abc)
{
    auto function = onnx_import::import_onnx_function(
        file_util::path_join(SERIALIZED_ZOO, "onnx/addmul_abc.onnx"));

    std::vector<std::vector<float>> inputs;

    Shape shape{1, 2, 2};
    inputs.emplace_back(test::NDArray<float, 3>({{{9, 10}}, {{11, 12}}}).get_vector());
    inputs.emplace_back(test::NDArray<float, 3>({{{5, 6}}, {{7, 8}}}).get_vector());
    inputs.emplace_back(test::NDArray<float, 3>({{{1, 2}}, {{3, 4}}}).get_vector());

    auto expected_output = test::NDArray<float, 3>({{{46, 62}}, {{80, 100}}}).get_vector();

    auto result_vectors = execute(function, inputs, "INTERPRETER");
    EXPECT_TRUE(test::all_close_f(expected_output, result_vectors.front()));
}

TEST(onnx, model_split_equal_parts_default)
{
    Model model{onnx_import::load_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/split_equal_parts_default.onnx"))};

    Inputs inputs{{1, 2, 3, 4, 5, 6}};
    Outputs expected_outputs{{1, 2}, {3, 4}, {5, 6}};

    for (std::size_t i = 0; i < expected_outputs.size(); ++i)
    {
        Outputs outputs{execute(model[i], inputs, "INTERPRETER")};
        EXPECT_EQ(outputs.size(), 1);
        EXPECT_TRUE(test::all_close_f(expected_outputs[i], outputs.front()));
    }
}

TEST(onnx, model_split_equal_parts_2d)
{
    // Split into 2 equal parts along axis=1
    Model model{onnx_import::load_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/split_equal_parts_2d.onnx"))};

    Inputs inputs{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}};
    Outputs expected_outputs{{0, 1, 2, 6, 7, 8}, {3, 4, 5, 9, 10, 11}};

    for (std::size_t i = 0; i < expected_outputs.size(); ++i)
    {
        Outputs outputs{execute(model[i], inputs, "INTERPRETER")};
        EXPECT_EQ(outputs.size(), 1);
        EXPECT_TRUE(test::all_close_f(expected_outputs[i], outputs.front()));
    }
}

TEST(onnx, model_split_variable_parts_2d)
{
    // Split into variable parts {2, 4} along axis=1
    Model model{onnx_import::load_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/split_variable_parts_2d.onnx"))};

    Inputs inputs{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}};
    Outputs expected_outputs{{0, 1, 6, 7}, {2, 3, 4, 5, 8, 9, 10, 11}};

    for (std::size_t i = 0; i < expected_outputs.size(); ++i)
    {
        Outputs outputs{execute(model[i], inputs, "INTERPRETER")};
        EXPECT_EQ(outputs.size(), 1);
        EXPECT_TRUE(test::all_close_f(expected_outputs[i], outputs.front()));
    }
}

namespace
{
    std::vector<std::vector<float>> conv2d_execute(const std::shared_ptr<Function>& function)
    {
        std::vector<std::vector<float>> args;

        // data (1, 1, 7, 5) input tensor
        args.emplace_back(test::NDArray<float, 4>{{{{{0.f, 1.f, 2.f, 3.f, 4.f},
                                                     {5.f, 6.f, 7.f, 8.f, 9.f},
                                                     {10.f, 11.f, 12.f, 13.f, 14.f},
                                                     {15.f, 16.f, 17.f, 18.f, 19.f},
                                                     {20.f, 21.f, 22.f, 23.f, 24.f},
                                                     {25.f, 26.f, 27.f, 28.f, 29.f},
                                                     {30.f, 31.f, 32.f, 33.f, 34.f}}}}}
                              .get_vector());

        // filters (1, 1, 3, 3) aka convolution weights
        args.emplace_back(
            test::NDArray<float, 4>{{{{{1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}}}}}
                .get_vector());

        return execute(function, args, "INTERPRETER");
    }
} // namespace

TEST(onnx, model_conv2d_strides_padding)
{
    // Convolution with strides=2 and padding=1
    auto function = onnx_import::import_onnx_function(
        file_util::path_join(SERIALIZED_ZOO, "onnx/conv_with_strides_padding.onnx"));

    // (1, 1, 4, 3)
    auto expected_output = test::NDArray<float, 4>({{{{12.f, 27.f, 24.f},
                                                      {63.f, 108.f, 81.f},
                                                      {123.f, 198.f, 141.f},
                                                      {112.f, 177.f, 124.f}}}})
                               .get_vector();

    auto result = conv2d_execute(function);
    EXPECT_EQ(expected_output, result.front());
}

TEST(onnx, model_conv2d_strides_no_padding)
{
    // Convolution with strides=2 and padding=1
    auto function = onnx_import::import_onnx_function(
        file_util::path_join(SERIALIZED_ZOO, "onnx/conv_with_strides_no_padding.onnx"));

    // (1, 1, 3, 2)
    auto expected_output =
        test::NDArray<float, 4>({{{{54.f, 72.f}, {144.f, 162.f}, {234.f, 252.f}}}}).get_vector();

    auto result = conv2d_execute(function);
    EXPECT_EQ(expected_output, result.front());
}

TEST(onnx, model_conv2d_strides_assymetric_padding)
{
    // Convolution with strides=2 and padding=1
    auto function = onnx_import::import_onnx_function(
        file_util::path_join(SERIALIZED_ZOO, "onnx/conv_with_strides_and_asymmetric_padding.onnx"));

    // (1, 1, 4, 2)
    auto expected_output =
        test::NDArray<float, 4>({{{{21.f, 33.f}, {99.f, 117.f}, {189.f, 207.f}, {171.f, 183.f}}}})
            .get_vector();

    auto result = conv2d_execute(function);
    EXPECT_EQ(expected_output, result.front());
}

TEST(onnx, model_average_pool_2d)
{
    // Pooling with strides=2 and no padding
    auto model = onnx_import::import_onnx_function(
        file_util::path_join(SERIALIZED_ZOO, "onnx/average_pool_2d.onnx"));

    // input data shape (1, 1, 4, 4)
    Inputs inputs;
    inputs.push_back(test::NDArray<float, 4>({{{{0.f, 1.f, 2.f, 3.f},
                                                {4.f, 5.f, 6.f, 7.f},
                                                {8.f, 9.f, 10.f, 11.f},
                                                {12.f, 13.f, 14.f, 15.f}}}})
                         .get_vector());

    // (1, 1, 2, 2)
    auto expected_output = test::NDArray<float, 4>({{{{2.5f, 4.5f}, {10.5f, 12.5f}}}}).get_vector();

    Outputs outputs{execute(model, inputs, "INTERPRETER")};

    EXPECT_EQ(expected_output, outputs.front());
}

TEST(onnx, model_average_pool_2d_pads)
{
    // Pooling with strides=2 and padding=1
    auto model = onnx_import::import_onnx_function(
        file_util::path_join(SERIALIZED_ZOO, "onnx/average_pool_2d_pads.onnx"));

    // input data shape (1, 1, 4, 4)
    Inputs inputs;
    inputs.push_back(test::NDArray<float, 4>({{{{0.f, 1.f, 2.f, 3.f},
                                                {4.f, 5.f, 6.f, 7.f},
                                                {8.f, 9.f, 10.f, 11.f},
                                                {12.f, 13.f, 14.f, 15.f}}}})
                         .get_vector());

    // (1, 1, 3, 3)
    auto expected_output =
        test::NDArray<float, 4>({{{{0.f, 1.5f, 3.f}, {6.f, 7.5f, 9.f}, {12.f, 13.5f, 15.f}}}})
            .get_vector();

    Outputs outputs = execute(model, inputs, "INTERPRETER");

    EXPECT_EQ(expected_output, outputs.front());
}

TEST(onnx, model_max_pool_2d_pads)
{
    // Pooling with strides=2 and padding=1
    auto model = onnx_import::import_onnx_function(
        file_util::path_join(SERIALIZED_ZOO, "onnx/max_pool_2d_pads.onnx"));

    // input data shape (1, 1, 4, 4)
    Inputs inputs;
    inputs.push_back(test::NDArray<float, 4>({{{{0.f, 1.f, 2.f, 3.f},
                                                {4.f, 5.f, 6.f, 7.f},
                                                {8.f, 9.f, 10.f, 11.f},
                                                {12.f, 13.f, 14.f, 15.f}}}})
                         .get_vector());

    // (1, 1, 3, 3)
    auto expected_output =
        test::NDArray<float, 4>({{{{0.f, 2.f, 3.f}, {8.f, 10.f, 11.f}, {12.f, 14.f, 15.f}}}})
            .get_vector();

    Outputs outputs{execute(model, inputs, "INTERPRETER")};

    EXPECT_EQ(expected_output, outputs.front());
}

TEST(onnx, model_batchnorm_default)
{
    // Batch Normalization with default parameters
    Model model{onnx_import::import_onnx_function(
        file_util::path_join(SERIALIZED_ZOO, "onnx/batchnorm_default.onnx"))};

    Inputs inputs;

    // input data shape (1, 2, 1, 3)
    inputs.push_back(
        test::NDArray<float, 4>({{{{-1.f, 0.f, 1.f}}, {{2.f, 3.f, 4.f}}}}).get_vector());

    // scale (3)
    inputs.emplace_back(std::vector<float>{1.f, 1.5f});
    // bias (3)
    inputs.emplace_back(std::vector<float>{0.f, 1.f});
    // mean (3)
    inputs.emplace_back(std::vector<float>{0.f, 3.f});
    // var (3)
    inputs.emplace_back(std::vector<float>{1.f, 1.5f});

    // shape (1, 2, 1, 3)
    Outputs expected_outputs{test::NDArray<float, 4>{
        {{{{-0.999995f, 0.f, 0.999995f}}, {{-0.22474074f, 1.f, 2.2247407f}}}}}
                                 .get_vector()};

    Outputs outputs{execute(model.front(), inputs, "INTERPRETER")};
    EXPECT_TRUE(test::all_close_f(expected_outputs.front(), outputs.front()));
}

TEST(onnx, model_relu)
{
    // Simple ReLU test
    auto function =
        onnx_import::import_onnx_function(file_util::path_join(SERIALIZED_ZOO, "onnx/relu.onnx"));

    Inputs inputs{{-1, -2, 0, 1, 2, 3}};
    Outputs expected_outputs{{0, 0, 0, 1, 2, 3}};

    Outputs outputs{execute(function, inputs, "INTERPRETER")};
    EXPECT_TRUE(test::all_close_f(expected_outputs.front(), outputs.front()));
}

TEST(onnx, model_sum)
{
    // Simple Sum test
    auto function =
        onnx_import::import_onnx_function(file_util::path_join(SERIALIZED_ZOO, "onnx/sum.onnx"));

    // input data shape (3, )
    Inputs inputs;
    inputs.emplace_back(std::vector<float>{3.f, 0.f, 2.f});
    inputs.emplace_back(std::vector<float>{1.f, 3.f, 4.f});
    inputs.emplace_back(std::vector<float>{2.f, 6.f, 6.f});

    Outputs expected_outputs{{6.f, 9.f, 12.f}};
    Outputs outputs{execute(function, inputs, "INTERPRETER")};
    EXPECT_TRUE(test::all_close_f(expected_outputs.front(), outputs.front()));
}

TEST(onnx, model_sum_one_input)
{
    auto function = onnx_import::import_onnx_function(
        file_util::path_join(SERIALIZED_ZOO, "onnx/sum_one_input.onnx"));

    // input data shape (3, )
    Inputs inputs{{3.f, 0.f, 2.f}};
    Outputs expected_outputs{{3.f, 0.f, 2.f}};
    Outputs outputs{execute(function, inputs, "INTERPRETER")};
    EXPECT_TRUE(test::all_close_f(expected_outputs.front(), outputs.front()));
}

TEST(onnx, model_min_two_inputs)
{
    auto function = onnx_import::import_onnx_function(
        file_util::path_join(SERIALIZED_ZOO, "onnx/min_two_inputs.onnx"));

    // input data shape (3, )
    Inputs inputs;
    inputs.emplace_back(std::vector<float>{1.f, 2.f, 1.f});
    inputs.emplace_back(std::vector<float>{1.f, 4.f, 4.f});

    Outputs expected_outputs{{1.f, 2.f, 1.f}};
    Outputs outputs{execute(function, inputs, "INTERPRETER")};
    EXPECT_TRUE(test::all_close_f(expected_outputs.front(), outputs.front()));
}

TEST(onnx, model_max)
{
    auto function =
        onnx_import::import_onnx_function(file_util::path_join(SERIALIZED_ZOO, "onnx/max.onnx"));

    // input data shape (3, )
    Inputs inputs;
    inputs.emplace_back(std::vector<float>{3.f, 2.f, 1.f});
    inputs.emplace_back(std::vector<float>{1.f, 4.f, 4.f});
    inputs.emplace_back(std::vector<float>{2.f, 5.f, 3.f});

    Outputs expected_outputs{{3.f, 5.f, 4.f}};
    Outputs outputs{execute(function, inputs, "INTERPRETER")};
    EXPECT_TRUE(test::all_close_f(expected_outputs.front(), outputs.front()));
}

TEST(onnx, model_mean)
{
    auto function =
        onnx_import::import_onnx_function(file_util::path_join(SERIALIZED_ZOO, "onnx/mean.onnx"));

    // input data shape (3, )
    Inputs inputs;
    inputs.emplace_back(std::vector<float>{3.f, 0.f, 2.f});
    inputs.emplace_back(std::vector<float>{1.f, 3.f, 4.f});
    inputs.emplace_back(std::vector<float>{2.f, 6.f, 6.f});

    Outputs expected_outputs{{2.f, 3.f, 4.f}};
    Outputs outputs{execute(function, inputs, "INTERPRETER")};
    EXPECT_TRUE(test::all_close_f(expected_outputs.front(), outputs.front()));
}

TEST(onnx, model_gemm_abc)
{
    auto function = onnx_import::import_onnx_function(
        file_util::path_join(SERIALIZED_ZOO, "onnx/gemm_abc.onnx"));

    Inputs inputs;
    inputs.emplace_back(test::NDArray<float, 2>(
                            {{1, 2, 3, 4, 5, 6}, {7, 8, 9, 10, 11, 12}, {13, 14, 15, 16, 17, 18}})
                            .get_vector());

    inputs.emplace_back(test::NDArray<float, 2>({{19, 20, 21, 22},
                                                 {23, 24, 25, 26},
                                                 {27, 28, 29, 30},
                                                 {31, 32, 33, 34},
                                                 {35, 36, 37, 38},
                                                 {39, 40, 41, 42}})
                            .get_vector());

    inputs.emplace_back(
        test::NDArray<float, 2>({{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}}).get_vector());

    Outputs expected_outputs{
        test::NDArray<float, 2>(
            {{340, 350.5, 361, 371.5}, {862, 890.5, 919, 947.5}, {1384, 1430.5, 1477, 1523.5}})
            .get_vector()};

    Outputs outputs{execute(function, inputs, "INTERPRETER")};
    EXPECT_TRUE(test::all_close_f(expected_outputs.front(), outputs.front()));
}

TEST(onnx, model_matmul)
{
    auto function =
        onnx_import::import_onnx_function(file_util::path_join(SERIALIZED_ZOO, "onnx/matmul.onnx"));

    std::vector<std::vector<float>> inputs;

    inputs.emplace_back(
        test::NDArray<float, 2>({{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}}).get_vector());

    inputs.emplace_back(
        test::NDArray<float, 2>({{13, 14, 15}, {16, 17, 18}, {19, 20, 21}, {22, 23, 24}})
            .get_vector());

    Outputs expected_outputs{
        test::NDArray<float, 2>({{190, 200, 210}, {470, 496, 522}, {750, 792, 834}}).get_vector()};

    Outputs outputs{execute(function, inputs, "INTERPRETER")};
    EXPECT_TRUE(test::all_close_f(expected_outputs.front(), outputs.front()));
}

TEST(onnx, model_softmax)
{
    auto function = onnx_import::import_onnx_function(
        file_util::path_join(SERIALIZED_ZOO, "onnx/softmax.onnx"));

    Inputs inputs;
    inputs.emplace_back(
        test::NDArray<float, 3>(
            {{{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}, {16, 17, 18, 19, 20}},

             {{21, 22, 23, 24, 25},
              {26, 27, 28, 29, 30},
              {31, 32, 33, 34, 35},
              {36, 37, 38, 39, 40}},

             {{41, 42, 43, 44, 45},
              {46, 47, 48, 49, 50},
              {51, 52, 53, 54, 55},
              {56, 57, 58, 59, 60}}})
            .get_vector());

    auto expected_output =
        test::NDArray<float, 3>(
            {{{1.50461533e-26f, 4.08996852e-26f, 1.11176871e-25f, 3.02210068e-25f, 8.21492137e-25f},
              {2.23304715e-24f, 6.07005148e-24f, 1.65001106e-23f, 4.48519509e-23f, 1.21920243e-22f},
              {3.31413582e-22f, 9.00875516e-22f, 2.44883355e-21f, 6.65661973e-21f, 1.80945684e-20f},
              {4.91861366e-20f,
               1.33701781e-19f,
               3.63439123e-19f,
               9.87929963e-19f,
               2.68547207e-18f}},

             {{7.29986992e-18f, 1.98431037e-17f, 5.39391483e-17f, 1.46621807e-16f, 3.98559393e-16f},
              {1.08339676e-15f, 2.94497771e-15f, 8.00527940e-15f, 2.17606055e-14f, 5.91514586e-14f},
              {1.60790335e-13f, 4.37073446e-13f, 1.18808881e-12f, 3.22956021e-12f, 8.77885484e-12f},
              {2.38634016e-11f,
               6.48674509e-11f,
               1.76328013e-10f,
               4.79309234e-10f,
               1.30289758e-09f}},

             {{3.54164282e-09f, 9.62718331e-09f, 2.61693974e-08f, 7.11357975e-08f, 1.93367146e-07f},
              {5.25626399e-07f, 1.42880069e-06f, 3.88388295e-06f, 1.05574884e-05f, 2.86982290e-05f},
              {7.80098743e-05f, 2.12052824e-04f, 5.76419338e-04f, 1.56687021e-03f, 4.25919482e-03f},
              {1.15776919e-02f,
               3.14714295e-02f,
               8.55482149e-02f,
               2.32544158e-01f,
               6.32120559e-01f}}})
            .get_vector();

    auto result_vectors = execute(function, inputs, "INTERPRETER");
    EXPECT_TRUE(test::all_close_f(expected_output, result_vectors.front()));
}

TEST(onnx, model_concat)
{
    auto function =
        onnx_import::import_onnx_function(file_util::path_join(SERIALIZED_ZOO, "onnx/concat.onnx"));

    Inputs inputs;

    inputs.emplace_back(test::NDArray<float, 1>({1, 2}).get_vector());
    inputs.emplace_back(test::NDArray<float, 1>({3, 4}).get_vector());

    Outputs expected_outputs{test::NDArray<float, 1>({1, 2, 3, 4}).get_vector()};

    Outputs outputs{execute(function, inputs, "INTERPRETER")};
    EXPECT_TRUE(test::all_close_f(expected_outputs.front(), outputs.front()));
}

TEST(onnx, model_flatten)
{
    auto function = onnx_import::import_onnx_function(
        file_util::path_join(SERIALIZED_ZOO, "onnx/flatten.onnx"));

    Inputs inputs;

    inputs.emplace_back(
        test::NDArray<float, 4>({{{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}}).get_vector());

    Outputs expected_outputs{test::NDArray<float, 3>({{{1, 2, 3, 4}, {5, 6, 7, 8}}}).get_vector()};

    Outputs outputs{execute(function, inputs, "INTERPRETER")};
    EXPECT_TRUE(test::all_close_f(expected_outputs.front(), outputs.front()));
}

TEST(onnx, model_sub)
{
    auto function =
        onnx_import::import_onnx_function(file_util::path_join(SERIALIZED_ZOO, "onnx/sub.onnx"));

    Inputs inputs;
    inputs.emplace_back(test::NDArray<float, 3>({{{1, 2, 3}}}).get_vector());

    inputs.emplace_back(test::NDArray<float, 3>({{{4, 5, 7}}}).get_vector());

    auto expected_output = test::NDArray<float, 3>({{{-3, -3, -4}}}).get_vector();

    auto result_vectors = execute(function, inputs, "INTERPRETER");
    EXPECT_TRUE(test::all_close_f(expected_output, result_vectors.front()));
}

TEST(onnx, model_unsqueeze)
{
    auto function = onnx_import::import_onnx_function(
        file_util::path_join(SERIALIZED_ZOO, "onnx/unsqueeze.onnx"));

    Inputs inputs;
    inputs.emplace_back(test::NDArray<float, 3>(
                            {{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                             {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                             {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}})
                            .get_vector());

    Outputs expected_output{
        test::NDArray<float, 4>(
            {{{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
              {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
              {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}}})
            .get_vector()};

    Outputs outputs{execute(function, inputs, "INTERPRETER")};
    EXPECT_TRUE(test::all_close_f(expected_output.front(), outputs.front()));
}

TEST(onnx, model_div)
{
    auto function =
        onnx_import::import_onnx_function(file_util::path_join(SERIALIZED_ZOO, "onnx/div.onnx"));

    Inputs inputs;
    inputs.emplace_back(test::NDArray<float, 3>({{{1, 2, 3}}}).get_vector());

    inputs.emplace_back(test::NDArray<float, 3>({{{1, 4, 12}}}).get_vector());

    auto expected_output = test::NDArray<float, 3>({{{1, 0.5, 0.25}}}).get_vector();

    auto result_vectors = execute(function, inputs, "INTERPRETER");
    EXPECT_TRUE(test::all_close_f(expected_output, result_vectors.front()));
}

TEST(onnx, model_add_bcast)
{
    auto function = onnx_import::import_onnx_function(
        file_util::path_join(SERIALIZED_ZOO, "onnx/add_bcast.onnx"));

    Inputs inputs;
    inputs.emplace_back(test::NDArray<float, 3>(
                            {{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                             {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                             {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}})
                            .get_vector());

    inputs.emplace_back(test::NDArray<float, 1>({1, 2, 3, 4, 5}).get_vector());

    Outputs expected_output{
        test::NDArray<float, 4>(
            {{{{2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}},
              {{2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}},
              {{2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}}}})
            .get_vector()};

    Outputs outputs{execute(function, inputs, "INTERPRETER")};
    EXPECT_TRUE(test::all_close_f(expected_output.front(), outputs.front()));
}

TEST(onnx, model_reshape_reduced_dims)
{
    auto function = onnx_import::import_onnx_function(
        file_util::path_join(SERIALIZED_ZOO, "onnx/reshape_reduced_dims.onnx"));

    // input data shape (2, 3, 4)
    Inputs inputs{test::NDArray<float, 3>({{{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}},
                                           {{12, 13, 14, 15}, {16, 17, 18, 19}, {20, 21, 22, 23}}})
                      .get_vector()};

    // output data shape (2, 12)
    Outputs expected_outputs{
        test::NDArray<float, 2>({{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
                                 {12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}})
            .get_vector()};

    Outputs outputs{execute(function, inputs, "INTERPRETER")};
    EXPECT_TRUE(test::all_close_f(expected_outputs.front(), outputs.front()));
}

TEST(onnx, model_reshape_reordered_dims)
{
    auto function = onnx_import::import_onnx_function(
        file_util::path_join(SERIALIZED_ZOO, "onnx/reshape_reordered_dims.onnx"));

    // input data shape (2, 3, 4)
    Inputs inputs{test::NDArray<float, 3>({{{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}},
                                           {{12, 13, 14, 15}, {16, 17, 18, 19}, {20, 21, 22, 23}}})
                      .get_vector()};

    // output data shape (4, 2, 3)
    Outputs expected_outputs{test::NDArray<float, 3>({{{0, 1, 2}, {3, 4, 5}},
                                                      {{6, 7, 8}, {9, 10, 11}},
                                                      {{12, 13, 14}, {15, 16, 17}},
                                                      {{18, 19, 20}, {21, 22, 23}}})
                                 .get_vector()};

    Outputs outputs{execute(function, inputs, "INTERPRETER")};
    EXPECT_TRUE(test::all_close_f(expected_outputs.front(), outputs.front()));
}

TEST(onnx, model_reshape_extended_dims)
{
    auto function = onnx_import::import_onnx_function(
        file_util::path_join(SERIALIZED_ZOO, "onnx/reshape_extended_dims.onnx"));

    // input data shape (2, 3, 4)
    Inputs inputs{test::NDArray<float, 3>({{{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}},
                                           {{12, 13, 14, 15}, {16, 17, 18, 19}, {20, 21, 22, 23}}})
                      .get_vector()};

    // output data shape (3, 2, 2, 2)
    Outputs expected_outputs{test::NDArray<float, 4>({{{{0, 1}, {2, 3}}, {{4, 5}, {6, 7}}},
                                                      {{{8, 9}, {10, 11}}, {{12, 13}, {14, 15}}},
                                                      {{{16, 17}, {18, 19}}, {{20, 21}, {22, 23}}}})
                                 .get_vector()};

    Outputs outputs{execute(function, inputs, "INTERPRETER")};
    EXPECT_TRUE(test::all_close_f(expected_outputs.front(), outputs.front()));
}

TEST(onnx, model_reshape_single_dim)
{
    auto function = onnx_import::import_onnx_function(
        file_util::path_join(SERIALIZED_ZOO, "onnx/reshape_single_dim.onnx"));

    // input data shape (2, 3, 4)
    Inputs inputs{test::NDArray<float, 3>({{{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}},
                                           {{12, 13, 14, 15}, {16, 17, 18, 19}, {20, 21, 22, 23}}})
                      .get_vector()};

    // output data shape (24, )
    Outputs expected_outputs{
        test::NDArray<float, 1>(
            {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23})
            .get_vector()};

    Outputs outputs{execute(function, inputs, "INTERPRETER")};
    EXPECT_TRUE(test::all_close_f(expected_outputs.front(), outputs.front()));
}

TEST(onnx, model_reshape_negative_dim)
{
    auto function = onnx_import::import_onnx_function(
        file_util::path_join(SERIALIZED_ZOO, "onnx/reshape_negative_dim.onnx"));

    // input data shape (2, 3, 4)
    Inputs inputs{test::NDArray<float, 3>({{{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}},
                                           {{12, 13, 14, 15}, {16, 17, 18, 19}, {20, 21, 22, 23}}})
                      .get_vector()};

    // output data shape (6, 2, 2)
    Outputs expected_outputs{test::NDArray<float, 3>({{{0, 1}, {2, 3}},
                                                      {{4, 5}, {6, 7}},
                                                      {{8, 9}, {10, 11}},
                                                      {{12, 13}, {14, 15}},
                                                      {{16, 17}, {18, 19}},
                                                      {{20, 21}, {22, 23}}})
                                 .get_vector()};

    Outputs outputs{execute(function, inputs, "INTERPRETER")};
    EXPECT_TRUE(test::all_close_f(expected_outputs.front(), outputs.front()));
}

TEST(onnx, model_reshape_negative_with_zero_dim)
{
    auto function = onnx_import::import_onnx_function(
        file_util::path_join(SERIALIZED_ZOO, "onnx/reshape_negative_with_zero_dims.onnx"));

    // input data shape (2, 3, 4)
    Inputs inputs{test::NDArray<float, 3>({{{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}},
                                           {{12, 13, 14, 15}, {16, 17, 18, 19}, {20, 21, 22, 23}}})
                      .get_vector()};

    // output data shape (2, 6, 2)
    Outputs expected_outputs{
        test::NDArray<float, 3>({{{0, 1}, {2, 3}, {4, 5}, {6, 7}, {8, 9}, {10, 11}},
                                 {{12, 13}, {14, 15}, {16, 17}, {18, 19}, {20, 21}, {22, 23}}})
            .get_vector()};

    Outputs outputs{execute(function, inputs, "INTERPRETER")};
    EXPECT_TRUE(test::all_close_f(expected_outputs.front(), outputs.front()));
}

TEST(onnx, model_reshape_output_shape_as_input)
{
    auto function = onnx_import::import_onnx_function(
        file_util::path_join(SERIALIZED_ZOO, "onnx/reshape_output_shape_as_input.onnx"));

    // input data shape (2, 3, 4)
    Inputs inputs{test::NDArray<float, 3>({{{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}},
                                           {{12, 13, 14, 15}, {16, 17, 18, 19}, {20, 21, 22, 23}}})
                      .get_vector()};

    // output data shape (2, 6, 2)
    Outputs expected_outputs{
        test::NDArray<float, 3>({{{0, 1}, {2, 3}, {4, 5}, {6, 7}, {8, 9}, {10, 11}},
                                 {{12, 13}, {14, 15}, {16, 17}, {18, 19}, {20, 21}, {22, 23}}})
            .get_vector()};

    Outputs outputs{execute(function, inputs, "INTERPRETER")};
    EXPECT_TRUE(test::all_close_f(expected_outputs.front(), outputs.front()));
}

TEST(onnx, model_reduce_log_sum)
{
    auto function = onnx_import::import_onnx_function(
        file_util::path_join(SERIALIZED_ZOO, "onnx/reduce_log_sum.onnx"));

    // input data shape (1, 1, 4, 4)
    Inputs inputs{
        test::NDArray<float, 4>({{{{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}}}})
            .get_vector()};

    // output data shape (1,)
    Outputs expected_outputs{test::NDArray<float, 4>({{{{2.77258872f}}}}).get_vector()};

    Outputs outputs{execute(function, inputs, "INTERPRETER")};
    EXPECT_TRUE(test::all_close_f(expected_outputs.front(), outputs.front()));
}

TEST(onnx, model_reduce_log_sum_exp)
{
    auto function = onnx_import::import_onnx_function(
        file_util::path_join(SERIALIZED_ZOO, "onnx/reduce_log_sum_exp.onnx"));

    // input data shape (1, 1, 4, 4)
    Inputs inputs{
        test::NDArray<float, 4>({{{{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}}}})
            .get_vector()};

    // output data shape (1,)
    Outputs expected_outputs{test::NDArray<float, 4>({{{{3.77258872f}}}}).get_vector()};

    Outputs outputs{execute(function, inputs, "INTERPRETER")};
    EXPECT_TRUE(test::all_close_f(expected_outputs.front(), outputs.front()));
}

TEST(onnx, model_reduce_l1)
{
    auto function = onnx_import::import_onnx_function(
        file_util::path_join(SERIALIZED_ZOO, "onnx/reduce_l1.onnx"));

    // input data shape (1, 1, 4, 4)
    Inputs inputs{
        test::NDArray<float, 4>({{{{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}}}})
            .get_vector()};

    // output data shape (1,)
    Outputs expected_outputs{test::NDArray<float, 4>({{{{16}}}}).get_vector()};

    Outputs outputs{execute(function, inputs, "INTERPRETER")};
    EXPECT_TRUE(test::all_close_f(expected_outputs.front(), outputs.front()));
}

TEST(onnx, model_reduce_l2)
{
    auto function = onnx_import::import_onnx_function(
        file_util::path_join(SERIALIZED_ZOO, "onnx/reduce_l2.onnx"));

    // input data shape (1, 1, 4, 4)
    Inputs inputs{
        test::NDArray<float, 4>({{{{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}}}})
            .get_vector()};

    // output data shape (1,)
    Outputs expected_outputs{test::NDArray<float, 4>({{{{4}}}}).get_vector()};

    Outputs outputs{execute(function, inputs, "INTERPRETER")};
    EXPECT_TRUE(test::all_close_f(expected_outputs.front(), outputs.front()));
}

TEST(onnx, model_reduce_max)
{
    auto function = onnx_import::import_onnx_function(
        file_util::path_join(SERIALIZED_ZOO, "onnx/reduce_max.onnx"));

    // input data shape (1, 1, 4, 4)
    Inputs inputs{
        test::NDArray<float, 4>({{{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}}}})
            .get_vector()};

    // output data shape (1,)
    Outputs expected_outputs{test::NDArray<float, 4>({{{{16}}}}).get_vector()};

    Outputs outputs{execute(function, inputs, "INTERPRETER")};
    EXPECT_TRUE(test::all_close_f(expected_outputs.front(), outputs.front()));
}

TEST(onnx, model_reduce_mean)
{
    auto function = onnx_import::import_onnx_function(
        file_util::path_join(SERIALIZED_ZOO, "onnx/reduce_mean.onnx"));

    // input data shape (1, 1, 4, 4)
    Inputs inputs{
        test::NDArray<float, 4>({{{{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}}}})
            .get_vector()};

    // output data shape (1,)
    Outputs expected_outputs{test::NDArray<float, 4>({{{{1}}}}).get_vector()};

    Outputs outputs{execute(function, inputs, "INTERPRETER")};
    EXPECT_TRUE(test::all_close_f(expected_outputs.front(), outputs.front()));
}

TEST(onnx, model_reduce_min)
{
    auto function = onnx_import::import_onnx_function(
        file_util::path_join(SERIALIZED_ZOO, "onnx/reduce_min.onnx"));

    // input data shape (1, 1, 4, 4)
    Inputs inputs{
        test::NDArray<float, 4>({{{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}}}})
            .get_vector()};

    // output data shape (1,)
    Outputs expected_outputs{test::NDArray<float, 4>({{{{1}}}}).get_vector()};

    Outputs outputs{execute(function, inputs, "INTERPRETER")};
    EXPECT_TRUE(test::all_close_f(expected_outputs.front(), outputs.front()));
}

TEST(onnx, model_reduce_prod)
{
    auto function = onnx_import::import_onnx_function(
        file_util::path_join(SERIALIZED_ZOO, "onnx/reduce_prod.onnx"));

    // input data shape (1, 1, 4, 4)
    Inputs inputs{
        test::NDArray<float, 4>({{{{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}}}})
            .get_vector()};

    // output data shape (1,)
    Outputs expected_outputs{test::NDArray<float, 4>({{{{1}}}}).get_vector()};

    Outputs outputs{execute(function, inputs, "INTERPRETER")};
    EXPECT_TRUE(test::all_close_f(expected_outputs.front(), outputs.front()));
}

TEST(onnx, model_reduce_sum)
{
    auto function = onnx_import::import_onnx_function(
        file_util::path_join(SERIALIZED_ZOO, "onnx/reduce_sum.onnx"));

    // input data shape (1, 1, 4, 4)
    Inputs inputs{
        test::NDArray<float, 4>({{{{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}}}})
            .get_vector()};

    // output data shape (1,)
    Outputs expected_outputs{test::NDArray<float, 4>({{{{16}}}}).get_vector()};

    Outputs outputs{execute(function, inputs, "INTERPRETER")};
    EXPECT_TRUE(test::all_close_f(expected_outputs.front(), outputs.front()));
}

TEST(onnx, model_reduce_sum_square)
{
    auto function = onnx_import::import_onnx_function(
        file_util::path_join(SERIALIZED_ZOO, "onnx/reduce_sum_square.onnx"));

    // input data shape (1, 1, 4, 4)
    Inputs inputs{
        test::NDArray<float, 4>({{{{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}}}})
            .get_vector()};

    // output data shape (1,)
    Outputs expected_outputs{test::NDArray<float, 4>({{{{16}}}}).get_vector()};

    Outputs outputs{execute(function, inputs, "INTERPRETER")};
    EXPECT_TRUE(test::all_close_f(expected_outputs.front(), outputs.front()));
}

TEST(onnx, model_shape)
{
    auto function =
        onnx_import::import_onnx_function(file_util::path_join(SERIALIZED_ZOO, "onnx/shape.onnx"));

    Inputs inputs;
    inputs.emplace_back(test::NDArray<float, 3>(
                            {{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                             {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                             {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}})
                            .get_vector());

    std::vector<std::vector<int64_t>> expected_output{{3, 4, 5}};

    std::vector<std::vector<int64_t>> outputs =
        execute<float, int64_t>(function, inputs, "INTERPRETER");
    EXPECT_TRUE(test::all_close(expected_output.front(), outputs.front()));
}

TEST(onnx, model_elu)
{
    auto function =
        onnx_import::import_onnx_function(file_util::path_join(SERIALIZED_ZOO, "onnx/elu.onnx"));

    Inputs inputs;
    inputs.emplace_back(
        test::NDArray<float, 3>(
            {{{-9, -8, -7, -6, -5}, {-4, -3, -2, -1, 0}, {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}},
             {{-4, -3, -2, -1, 0}, {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}},
             {{1, 1, 1, 1, 1}, {-1, -1, -1, -1, -1}, {0, 0, 0, 0, 0}, {2, 2, 2, 2, 2}}})
            .get_vector());

    Outputs expected_output{test::NDArray<float, 3>({{{-1.999753180391830f,
                                                       -1.999329074744190f,
                                                       -1.998176236068890f,
                                                       -1.995042495646670f,
                                                       -1.986524106001830f},
                                                      {-1.963368722222530f,
                                                       -1.900425863264270f,
                                                       -1.729329433526770f,
                                                       -1.264241117657120f,
                                                       0},
                                                      {1, 2, 3, 4, 5},
                                                      {6, 7, 8, 9, 10}},
                                                     {{-1.963368722222530f,
                                                       -1.900425863264270f,
                                                       -1.729329433526770f,
                                                       -1.264241117657120f,
                                                       0},
                                                      {1, 2, 3, 4, 5},
                                                      {6, 7, 8, 9, 10},
                                                      {11, 12, 13, 14, 15}},
                                                     {{1, 1, 1, 1, 1},
                                                      {-1.264241117657120f,
                                                       -1.264241117657120f,
                                                       -1.264241117657120f,
                                                       -1.264241117657120f,
                                                       -1.264241117657120f},
                                                      {0, 0, 0, 0, 0},
                                                      {2, 2, 2, 2, 2}}})
                                .get_vector()};

    Outputs outputs{execute(function, inputs, "INTERPRETER")};
    EXPECT_TRUE(test::all_close_f(expected_output.front(), outputs.front()));
}

TEST(onnx, model_leaky_relu)
{
    auto function = onnx_import::import_onnx_function(
        file_util::path_join(SERIALIZED_ZOO, "onnx/leaky_relu.onnx"));

    Inputs inputs;
    inputs.emplace_back(
        test::NDArray<float, 3>(
            {{{-9, -8, -7, -6, -5}, {-4, -3, -2, -1, 0}, {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}},
             {{-4, -3, -2, -1, 0}, {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}},
             {{1, 1, 1, 1, 1}, {-1, -1, -1, -1, -1}, {0, 0, 0, 0, 0}, {2, 2, 2, 2, 2}}})
            .get_vector());

    Outputs expected_output{test::NDArray<float, 3>({{{-0.9f, -0.8f, -0.7f, -0.6f, -0.5f},
                                                      {-0.4f, -0.3f, -0.2f, -0.1f, 0},
                                                      {1, 2, 3, 4, 5},
                                                      {6, 7, 8, 9, 10}},
                                                     {{-0.4f, -0.3f, -0.2f, -0.1f, 0},
                                                      {1, 2, 3, 4, 5},
                                                      {6, 7, 8, 9, 10},
                                                      {11, 12, 13, 14, 15}},
                                                     {{1, 1, 1, 1, 1},
                                                      {-0.1f, -0.1f, -0.1f, -0.1f, -0.1f},
                                                      {0, 0, 0, 0, 0},
                                                      {2, 2, 2, 2, 2}}})
                                .get_vector()};

    Outputs outputs{execute(function, inputs, "INTERPRETER")};
    EXPECT_TRUE(test::all_close_f(expected_output.front(), outputs.front()));
}

TEST(onnx, prelu)
{
    auto function =
        onnx_import::import_onnx_function(file_util::path_join(SERIALIZED_ZOO, "onnx/prelu.onnx"));

    Inputs inputs;
    inputs.emplace_back(
        test::NDArray<float, 3>(
            {{{-9, -8, -7, -6, -5}, {-4, -3, -2, -1, 0}, {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}},
             {{-4, -3, -2, -1, 0}, {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}},
             {{1, 1, 1, 1, 1}, {-1, -1, -1, -1, -1}, {0, 0, 0, 0, 0}, {2, 2, 2, 2, 2}}})
            .get_vector());

    inputs.emplace_back(test::NDArray<float, 3>(
                            {{{1, 0, 1, 0, 1}, {0, 1, 0, 1, 0}, {1, 0, 1, 0, 1}, {0, 1, 0, 1, 0}},
                             {{0, 1, 0, 1, 0}, {1, 0, 1, 0, 1}, {0, 1, 0, 1, 0}, {1, 0, 1, 0, 1}},
                             {{1, 0, 1, 0, 1}, {0, 1, 0, 1, 0}, {1, 0, 1, 0, 1}, {0, 1, 0, 1, 0}}})
                            .get_vector());

    Outputs expected_output{
        test::NDArray<float, 3>(
            {{{-9, 0, -7, 0, -5}, {0, -3, 0, -1, 0}, {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}},
             {{0, -3, 0, -1, 0}, {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}},
             {{1, 1, 1, 1, 1}, {0, -1, 0, -1, 0}, {0, 0, 0, 0, 0}, {2, 2, 2, 2, 2}}})
            .get_vector()};

    Outputs outputs{execute(function, inputs, "INTERPRETER")};
    EXPECT_TRUE(test::all_close_f(expected_output.front(), outputs.front()));
}

TEST(onnx, model_selu)
{
    auto function =
        onnx_import::import_onnx_function(file_util::path_join(SERIALIZED_ZOO, "onnx/selu.onnx"));

    Inputs inputs;
    inputs.emplace_back(
        test::NDArray<float, 3>(
            {{{-9, -8, -7, -6, -5}, {-4, -3, -2, -1, 0}, {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}},
             {{-4, -3, -2, -1, 0}, {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}},
             {{1, 1, 1, 1, 1}, {-1, -1, -1, -1, -1}, {0, 0, 0, 0, 0}, {2, 2, 2, 2, 2}}})
            .get_vector());

    Outputs expected_output{
        test::NDArray<float, 3>(
            {{{-5.99925954117548f,
               -5.99798722423258f,
               -5.99452870820667f,
               -5.98512748694000f,
               -5.95957231800549f},
              {-5.89010616666759f, -5.70127758979282f, -5.18798830058032f, -3.79272335297135f, 0},
              {3, 6, 9, 12, 15},
              {18, 21, 24, 27, 30}},
             {{-5.89010616666759f, -5.70127758979282f, -5.18798830058032f, -3.79272335297135f, 0},
              {3, 6, 9, 12, 15},
              {18, 21, 24, 27, 30},
              {33, 36, 39, 42, 45}},
             {{3, 3, 3, 3, 3},
              {-3.79272335297135f,
               -3.79272335297135f,
               -3.79272335297135f,
               -3.79272335297135f,
               -3.79272335297135f},
              {0, 0, 0, 0, 0},
              {6, 6, 6, 6, 6}}})
            .get_vector()};

    Outputs outputs{execute(function, inputs, "INTERPRETER")};
    EXPECT_TRUE(test::all_close_f(expected_output.front(), outputs.front()));
}

TEST(onnx, model_sigmoid)
{
    auto function = onnx_import::import_onnx_function(
        file_util::path_join(SERIALIZED_ZOO, "onnx/sigmoid.onnx"));

    Inputs inputs;
    inputs.emplace_back(
        test::NDArray<float, 3>(
            {{{-9, -8, -7, -6, -5}, {-4, -3, -2, -1, 0}, {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}},
             {{-4, -3, -2, -1, 0}, {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}},
             {{1, 1, 1, 1, 1}, {-1, -1, -1, -1, -1}, {0, 0, 0, 0, 0}, {2, 2, 2, 2, 2}}})
            .get_vector());

    Outputs expected_output{test::NDArray<float, 3>({{{0.00012339457598623f,
                                                       0.00033535013046648f,
                                                       0.00091105119440065f,
                                                       0.00247262315663477f,
                                                       0.00669285092428486f},
                                                      {0.01798620996209160f,
                                                       0.04742587317756680f,
                                                       0.119202922022118f,
                                                       0.268941421369995f,
                                                       0.5f},
                                                      {0.731058578630005f,
                                                       0.880797077977882f,
                                                       0.952574126822433f,
                                                       0.982013790037908f,
                                                       0.993307149075715f},
                                                      {0.997527376843365f,
                                                       0.999088948805599f,
                                                       0.999664649869534f,
                                                       0.999876605424014f,
                                                       0.999954602131298f}},
                                                     {{0.01798620996209160f,
                                                       0.04742587317756680f,
                                                       0.119202922022118f,
                                                       0.268941421369995f,
                                                       0.5f},
                                                      {0.731058578630005f,
                                                       0.880797077977882f,
                                                       0.952574126822433f,
                                                       0.982013790037908f,
                                                       0.993307149075715f},
                                                      {0.997527376843365f,
                                                       0.999088948805599f,
                                                       0.999664649869534f,
                                                       0.999876605424014f,
                                                       0.999954602131298f},
                                                      {0.999983298578152f,
                                                       0.999993855825398f,
                                                       0.999997739675702f,
                                                       0.999999168471972f,
                                                       0.999999694097773f}},
                                                     {{0.731058578630005f,
                                                       0.731058578630005f,
                                                       0.731058578630005f,
                                                       0.731058578630005f,
                                                       0.731058578630005f},
                                                      {0.268941421369995f,
                                                       0.268941421369995f,
                                                       0.268941421369995f,
                                                       0.268941421369995f,
                                                       0.268941421369995f},
                                                      {0.5f, 0.5f, 0.5f, 0.5f, 0.5f},
                                                      {0.880797077977882f,
                                                       0.880797077977882f,
                                                       0.880797077977882f,
                                                       0.880797077977882f,
                                                       0.880797077977882f}}})
                                .get_vector()};

    Outputs outputs{execute(function, inputs, "INTERPRETER")};
    EXPECT_TRUE(test::all_close_f(expected_output.front(), outputs.front()));
}

TEST(onnx, model_tanh)
{
    auto function =
        onnx_import::import_onnx_function(file_util::path_join(SERIALIZED_ZOO, "onnx/tanh.onnx"));

    Inputs inputs;
    inputs.emplace_back(
        test::NDArray<float, 3>(
            {{{-9, -8, -7, -6, -5}, {-4, -3, -2, -1, 0}, {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}},
             {{-4, -3, -2, -1, 0}, {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}},
             {{1, 1, 1, 1, 1}, {-1, -1, -1, -1, -1}, {0, 0, 0, 0, 0}, {2, 2, 2, 2, 2}}})
            .get_vector());

    Outputs expected_output{test::NDArray<float, 3>({{{-0.999999969540041f,
                                                       -0.999999774929676f,
                                                       -0.999998336943945f,
                                                       -0.999987711650796f,
                                                       -0.999909204262595f},
                                                      {-0.999329299739067f,
                                                       -0.995054753686731f,
                                                       -0.964027580075817f,
                                                       -0.761594155955765f,
                                                       0},
                                                      {0.761594155955765f,
                                                       0.964027580075817f,
                                                       0.995054753686731f,
                                                       0.999329299739067f,
                                                       0.999909204262595f},
                                                      {0.999987711650796f,
                                                       0.999998336943945f,
                                                       0.999999774929676f,
                                                       0.999999969540041f,
                                                       0.999999995877693f}},
                                                     {{-0.999329299739067f,
                                                       -0.995054753686731f,
                                                       -0.964027580075817f,
                                                       -0.761594155955765f,
                                                       0},
                                                      {0.761594155955765f,
                                                       0.964027580075817f,
                                                       0.995054753686731f,
                                                       0.999329299739067f,
                                                       0.999909204262595f},
                                                      {0.999987711650796f,
                                                       0.999998336943945f,
                                                       0.999999774929676f,
                                                       0.999999969540041f,
                                                       0.999999995877693f},
                                                      {0.999999999442106f,
                                                       0.999999999924497f,
                                                       0.999999999989782f,
                                                       0.999999999998617f,
                                                       0.999999999999813f}},
                                                     {{0.761594155955765f,
                                                       0.761594155955765f,
                                                       0.761594155955765f,
                                                       0.761594155955765f,
                                                       0.761594155955765f},
                                                      {-0.761594155955765f,
                                                       -0.761594155955765f,
                                                       -0.761594155955765f,
                                                       -0.761594155955765f,
                                                       -0.761594155955765f},
                                                      {0, 0, 0, 0, 0},
                                                      {0.964027580075817f,
                                                       0.964027580075817f,
                                                       0.964027580075817f,
                                                       0.964027580075817f,
                                                       0.964027580075817f}}})
                                .get_vector()};

    Outputs outputs{execute(function, inputs, "INTERPRETER")};
    EXPECT_TRUE(test::all_close_f(expected_output.front(), outputs.front()));
}

TEST(onnx, model_thresholded_relu)
{
    auto function = onnx_import::import_onnx_function(
        file_util::path_join(SERIALIZED_ZOO, "onnx/thresholded_relu.onnx"));

    Inputs inputs;
    inputs.emplace_back(
        test::NDArray<float, 3>(
            {{{-9, -8, -7, -6, -5}, {-4, -3, -2, -1, 0}, {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}},
             {{-4, -3, -2, -1, 0}, {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}},
             {{1, 1, 1, 1, 1}, {-1, -1, -1, -1, -1}, {0, 0, 0, 0, 0}, {2, 2, 2, 2, 2}}})
            .get_vector());

    Outputs expected_output{
        test::NDArray<float, 3>(
            {{{0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 3, 4, 5}, {6, 7, 8, 9, 10}},
             {{0, 0, 0, 0, 0}, {0, 0, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}},
             {{0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}}})
            .get_vector()};

    Outputs outputs{execute(function, inputs, "INTERPRETER")};
    EXPECT_TRUE(test::all_close_f(expected_output.front(), outputs.front()));
}

TEST(onnx, model_superres)
{
    auto function = onnx_import::import_onnx_function(
        file_util::path_join(SERIALIZED_ZOO, "onnx/superres.onnx"));

    Inputs inputs{{224 * 224, 1.0f}};
    Outputs outputs{execute(function, inputs, "CPU")};
}
