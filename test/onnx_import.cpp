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

using Inputs = std::vector<std::vector<float>>;
using Outputs = std::vector<std::vector<float>>;
using Model = std::vector<std::shared_ptr<Function>>;
using Backend = std::shared_ptr<runtime::Backend>;
using TensorView = std::shared_ptr<runtime::TensorView>;

TEST(onnx, model_add_abc)
{
    auto function{onnx_import::import_onnx_function(
        file_util::path_join(SERIALIZED_ZOO, "onnx/add_abc.onnx"))};
    Backend backend{runtime::Backend::create("INTERPRETER")};

    Shape shape{1};
    TensorView a{backend->create_tensor(element::f32, shape)};
    copy_data(a, std::vector<float>{1});
    TensorView b{backend->create_tensor(element::f32, shape)};
    copy_data(b, std::vector<float>{2});
    TensorView c{backend->create_tensor(element::f32, shape)};
    copy_data(c, std::vector<float>{3});

    TensorView r{backend->create_tensor(element::f32, shape)};

    backend->call(function, {r}, {a, b, c});
    EXPECT_TRUE(test::all_close_f({6}, read_vector<float>(r)));
}

TEST(onnx, model_add_abc_initializers)
{
    auto function{onnx_import::import_onnx_function(
        file_util::path_join(SERIALIZED_ZOO, "onnx/add_abc_initializers.onnx"))};
    Backend backend{runtime::Backend::create("INTERPRETER")};

    Shape shape{2, 2};

    TensorView c{backend->create_tensor(element::f32, shape)};
    copy_data(c, std::vector<float>{1, 2, 3, 4});

    TensorView r{backend->create_tensor(element::f32, shape)};

    backend->call(function, {r}, {c});
    EXPECT_TRUE(test::all_close_f({3, 6, 9, 12}, read_vector<float>(r)));
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
    auto function{ngraph::onnx_import::import_onnx_function(
        ngraph::file_util::path_join(SERIALIZED_ZOO, "onnx/relu.onnx"))};

    Inputs inputs{{-1, -2, 0, 1, 2, 3}};
    Outputs expected_outputs{{0, 0, 0, 1, 2, 3}};

    Outputs outputs{execute(function, inputs, "INTERPRETER")};
    EXPECT_TRUE(test::all_close_f(expected_outputs.front(), outputs.front()));
}
