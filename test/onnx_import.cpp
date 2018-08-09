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
#include "util/test_tools.hpp"

TEST(onnx, model_add_abc)
{
    auto model{ngraph::onnx_import::load_onnx_model(
        ngraph::file_util::path_join(SERIALIZED_ZOO, "onnx/add_abc.onnx"))};
    auto backend{ngraph::runtime::Backend::create("CPU")};

    ngraph::Shape shape{1};
    auto a{backend->create_tensor(ngraph::element::f32, shape)};
    copy_data(a, std::vector<float>{1});
    auto b{backend->create_tensor(ngraph::element::f32, shape)};
    copy_data(b, std::vector<float>{2});
    auto c{backend->create_tensor(ngraph::element::f32, shape)};
    copy_data(c, std::vector<float>{3});

    auto r{backend->create_tensor(ngraph::element::f32, shape)};

    backend->call(model.front(), {r}, {a, b, c});
    EXPECT_EQ((std::vector<float>{6}), read_vector<float>(r));
}

TEST(onnx, model_add_abc_initializers)
{
    auto model{ngraph::onnx_import::load_onnx_model(
        ngraph::file_util::path_join(SERIALIZED_ZOO, "onnx/add_abc_initializers.onnx"))};
    auto backend{ngraph::runtime::Backend::create("CPU")};

    ngraph::Shape shape{2, 2};

    auto c{backend->create_tensor(ngraph::element::f32, shape)};
    copy_data(c, std::vector<float>{1, 2, 3, 4});

    auto r{backend->create_tensor(ngraph::element::f32, shape)};

    backend->call(model.front(), {r}, {c});
    EXPECT_EQ((std::vector<float>{3, 6, 9, 12}), read_vector<float>(r));
}

TEST(onnx, model_split_default)
{
    auto function{ngraph::onnx_import::import_onnx_function(
            ngraph::file_util::path_join(SERIALIZED_ZOO, "onnx/split_default.onnx"))};
    auto backend{ngraph::runtime::Backend::create("CPU")};
}

TEST(onnx, model_split)
{
    auto function{ngraph::onnx_import::import_onnx_function(
            ngraph::file_util::path_join(SERIALIZED_ZOO, "onnx/split.onnx"))};
    auto backend{ngraph::runtime::Backend::create("CPU")};
}

TEST(onnx, model_batchnorm_default)
{
    // Batch Normalization with default parameters
    auto function{ngraph::onnx_import::import_onnx_function(
        ngraph::file_util::path_join(SERIALIZED_ZOO, "onnx/batchnorm_default.onnx"))};

    // std::vector<std::vector<float>> inputs;
    // inputs.emplace_back{read_tensor_proto_data_file<float>(
    //     ngraph::file_util::path_join(SERIALIZED_ZOO, "onnx/batchnorm_default_input_0.pb"))};

    // auto expected_output = read_tensor_proto_data_file<float>(
    //     ngraph::file_util::path_join(SERIALIZED_ZOO, "onnx/batchnorm_default_output_0.pb"));

    // auto result_vectors = execute(function, inputs, "CPU");
    // EXPECT_EQ(expected_output, result_vectors.front());
}