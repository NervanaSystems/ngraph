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
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "gtest/gtest.h"
#include "ngraph/frontend/onnx_import/onnx.hpp"
#include "ngraph/ngraph.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/ndarray.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace ngraph;

static std::string s_manifest = "${MANIFEST}";

using Inputs = std::vector<std::vector<float>>;
using Outputs = std::vector<std::vector<float>>;

// ############################################################################ CORE TESTS
NGRAPH_TEST(onnx_${BACKEND_NAME}, model_output_names_check)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/split_equal_parts_default.prototxt"));

    std::size_t size = function->get_output_size();
    for (std::size_t i{0}; i < size; ++i)
    {
        std::shared_ptr<Node> node = function->get_output_op(i);
        EXPECT_EQ(node->get_friendly_name(), "output_" + std::to_string(i + 1));
    }
}

NGRAPH_TEST(onnx_${BACKEND_NAME}, model_add_abc)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/add_abc.prototxt"));

    Inputs inputs{{1}, {2}, {3}};
    Outputs expected_outputs{{6}};

    Outputs outputs{execute(function, inputs, "${BACKEND_NAME}")};
    EXPECT_TRUE(test::all_close_f(expected_outputs.front(), outputs.front()));
}

NGRAPH_TEST(onnx_${BACKEND_NAME}, model_binary_add_abc)
{
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/add_abc.onnx"));

    Inputs inputs{{1}, {2}, {3}};
    Outputs expected_outputs{{6}};

    Outputs outputs{execute(function, inputs, "${BACKEND_NAME}")};
    EXPECT_TRUE(test::all_close_f(expected_outputs.front(), outputs.front()));
}

NGRAPH_TEST(onnx_${BACKEND_NAME}, model_add_abc_initializers)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/add_abc_initializers.prototxt"));

    Inputs inputs{{1, 2, 3, 4}};
    Outputs expected_outputs{{3, 6, 9, 12}};

    Outputs outputs{execute(function, inputs, "${BACKEND_NAME}")};
    EXPECT_TRUE(test::all_close_f(expected_outputs.front(), outputs.front()));
}

NGRAPH_TEST(onnx_${BACKEND_NAME}, model_override_op)
{
    onnx_import::register_operator(
        "FalseAdd", 1, "", [](const onnx_import::Node& node) -> NodeVector {
            NodeVector ng_inputs{node.get_ng_inputs()};
            return {std::make_shared<ngraph::op::Add>(ng_inputs.at(0), ng_inputs.at(1))};
        });

    onnx_import::register_operator(
        "FalseAdd", 1, "", [](const onnx_import::Node& node) -> NodeVector {
            NodeVector ng_inputs{node.get_ng_inputs()};
            return {std::make_shared<ngraph::op::Subtract>(ng_inputs.at(0), ng_inputs.at(1))};
        });

    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/override_op.prototxt"));

    Inputs inputs;
    inputs.emplace_back(std::vector<float>{0.f, 1.f, 2.f, 3.f});
    inputs.emplace_back(std::vector<float>{3.f, 2.f, 1.f, 0.f});

    Outputs expected_output{std::vector<float>{-3.f, -1.f, 1.f, 3.f}};

    Outputs outputs{execute(function, inputs, "${BACKEND_NAME}")};
    EXPECT_TRUE(test::all_close_f(expected_output.front(), outputs.front()));
}

NGRAPH_TEST(onnx_${BACKEND_NAME}, import_non_existing_file)
{
    try
    {
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/i.dont.exist"));
    }
    catch (const std::runtime_error& exc)
    {
        // asserts that an exception was thrown and that the error message contains the file name
        std::string msg{exc.what()};
        EXPECT_TRUE(msg.find("i.dont.exist") != std::string::npos);
    }
}

NGRAPH_TEST(onnx_${BACKEND_NAME}, model_unsupported_op)
{
    try
    {
        onnx_import::import_onnx_model(
            file_util::path_join(SERIALIZED_ZOO, "onnx/unsupported_op.prototxt"));
        FAIL() << "Expected ngraph::ngraph_error";
    }
    catch (ngraph::ngraph_error const& err)
    {
        std::string what{err.what()};
        EXPECT_NE(what.find("nGraph does not support"), std::string::npos);
        EXPECT_NE(what.find("FakeOpName"), std::string::npos);
        EXPECT_NE(what.find("AnotherFakeOpName"), std::string::npos);
    }
    catch (...)
    {
        FAIL() << "Expected ngraph::ngraph_error";
    }
}

NGRAPH_TEST(onnx_${BACKEND_NAME}, model_custom_op)
{
    onnx_import::register_operator(
        "AddQ", 1, "com.intel.ai", [](const onnx_import::Node& node) -> NodeVector {
            NodeVector ng_inputs{node.get_ng_inputs()};
            return {std::make_shared<ngraph::op::Add>(ng_inputs.at(0), ng_inputs.at(1))};
        });

    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/custom_operator.prototxt"));

    Inputs inputs{{1, 2, 3, 4}};
    Outputs expected_outputs{{3, 6, 9, 12}};

    Outputs outputs{execute(function, inputs, "${BACKEND_NAME}")};
    EXPECT_TRUE(test::all_close_f(expected_outputs.front(), outputs.front()));
}

NGRAPH_TEST(onnx_${BACKEND_NAME}, model_custom_op_default_domain)
{
    onnx_import::register_operator(
        "AddQ", 1, "com.intel.ai", [](const onnx_import::Node& node) -> NodeVector {
            NodeVector ng_inputs{node.get_ng_inputs()};
            return {std::make_shared<ngraph::op::Add>(ng_inputs.at(0), ng_inputs.at(1))};
        });

    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/custom_operator_default_domain.prototxt"));

    Inputs inputs{{1, 2, 3, 4}};
    Outputs expected_outputs{{3, 6, 9, 12}};

    Outputs outputs{execute(function, inputs, "${BACKEND_NAME}")};
    EXPECT_TRUE(test::all_close_f(expected_outputs.front(), outputs.front()));
}

NGRAPH_TEST(onnx_${BACKEND_NAME}, is_op_supported)
{
    // Simple case
    EXPECT_TRUE(onnx_import::is_operator_supported("Sum", 1, "ai.onnx"));
    // With fallback
    EXPECT_TRUE(onnx_import::is_operator_supported("Sum", 100, "ai.onnx"));

    // Different opset versions
    EXPECT_TRUE(onnx_import::is_operator_supported("Add", 1, "ai.onnx"));
    EXPECT_TRUE(onnx_import::is_operator_supported("Add", 7, "ai.onnx"));

    // Default domain name
    EXPECT_TRUE(onnx_import::is_operator_supported("Sum", 1));

    // Unregistered operator
    EXPECT_FALSE(onnx_import::is_operator_supported("DummyOp", 1));
    EXPECT_FALSE(onnx_import::is_operator_supported("DummyOp", 1, "ai.onnx"));
    EXPECT_FALSE(onnx_import::is_operator_supported("DummyOp", 10, "ai.onnx"));

    // Operator with bad domain name
    EXPECT_FALSE(onnx_import::is_operator_supported("Sum", 1, "bad.domain"));

    // Registered custom operator
    onnx_import::register_operator(
        "AddQ", 1, "com.intel.ai", [](const onnx_import::Node& node) -> NodeVector {
            NodeVector ng_inputs{node.get_ng_inputs()};
            return {std::make_shared<ngraph::op::Add>(ng_inputs.at(0), ng_inputs.at(1))};
        });
    EXPECT_TRUE(onnx_import::is_operator_supported("AddQ", 1, "com.intel.ai"));
}

NGRAPH_TEST(onnx_${BACKEND_NAME}, model_missing_op_domain)
{
    onnx_import::register_operator(
        "CustomAdd", 1, "custom.op", [](const onnx_import::Node& node) -> NodeVector {
            NodeVector ng_inputs{node.get_ng_inputs()};
            return {std::make_shared<ngraph::op::Add>(ng_inputs.at(0), ng_inputs.at(1))};
        });

    EXPECT_TRUE(onnx_import::is_operator_supported("CustomAdd", 1, "custom.op"));

    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/missing_op_domain.prototxt"));

    Inputs inputs;
    inputs.emplace_back(std::vector<float>{0.f, 1.f, 2.f, 3.f});
    inputs.emplace_back(std::vector<float>{0.f, 1.f, 2.f, 3.f});

    Outputs expected_output{std::vector<float>{0.f, 2.f, 4.f, 6.f}};

    Outputs outputs{execute(function, inputs, "${BACKEND_NAME}")};
    EXPECT_TRUE(test::all_close_f(expected_output.front(), outputs.front()));
}

NGRAPH_TEST(onnx_${BACKEND_NAME}, model_missing_input)
{
    onnx_import::register_operator(
        "TestMissingInOut", 1, "com.intel.ai", [](const onnx_import::Node& node) -> NodeVector {
            NodeVector ng_inputs{node.get_ng_inputs()};
            std::shared_ptr<ngraph::Node> A = ng_inputs.at(0);
            std::shared_ptr<ngraph::Node> B = ng_inputs.at(1);
            std::shared_ptr<ngraph::Node> C = ng_inputs.at(2);

            A = A * C;
            if (!B->is_null())
            {
                B = B / C;
            }

            C = C + C;
            return {A, B, C};
        });

    onnx_import::register_operator(
        "TestMissingIn", 1, "com.intel.ai", [](const onnx_import::Node& node) -> NodeVector {
            NodeVector ng_inputs{node.get_ng_inputs()};
            std::shared_ptr<ngraph::Node> result = std::make_shared<ngraph::op::Constant>(
                element::f32, ngraph::Shape{2, 2}, std::vector<float>{1, 1, 1, 1});

            for (const auto& ng_input : ng_inputs)
            {
                if (!ng_input->is_null())
                {
                    result = ng_input * result;
                }
            }

            return {result};
        });

    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/missing_input.prototxt"));

    Inputs inputs{{1, 2, 3, 4}, {5, 6, 7, 8}};
    Outputs expected_outputs{{50, 144, 294, 512}};

    Outputs outputs{execute(function, inputs, "${BACKEND_NAME}")};

    EXPECT_TRUE(test::all_close_f(expected_outputs.front(), outputs.front()));
}

NGRAPH_TEST(onnx_${BACKEND_NAME}, model_initializer_wo_input)
{
    // This test checks a model which has an initializer, but no input with the same name
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/initializer_wo_input.prototxt"));

    Inputs inputs;
    inputs.emplace_back(std::vector<float>{0, 1, 2, 3, 4, 5});

    std::vector<float> expected_output{0, 2, 6, 12, 20, 30};

    Outputs output{execute(function, inputs, "${BACKEND_NAME}")};
    EXPECT_TRUE(test::all_close_f(expected_output, output.front()));
}

// ############################################################################ OPERATOR TESTS
NGRAPH_TEST(onnx_${BACKEND_NAME}, model_addmul_abc)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/addmul_abc.prototxt"));

    std::vector<std::vector<float>> inputs;

    Shape shape{1, 2, 2};
    inputs.emplace_back(test::NDArray<float, 3>({{{9, 10}}, {{11, 12}}}).get_vector());
    inputs.emplace_back(test::NDArray<float, 3>({{{5, 6}}, {{7, 8}}}).get_vector());
    inputs.emplace_back(test::NDArray<float, 3>({{{1, 2}}, {{3, 4}}}).get_vector());

    auto expected_output = test::NDArray<float, 3>({{{46, 62}}, {{80, 100}}}).get_vector();

    auto result_vectors = execute(function, inputs, "${BACKEND_NAME}");
    EXPECT_TRUE(test::all_close_f(expected_output, result_vectors.front()));
}

NGRAPH_TEST(onnx_${BACKEND_NAME}, model_argmin_no_keepdims)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/argmin_no_keepdims.prototxt"));

    Inputs inputs{test::NDArray<float, 2>{{2, 1}, {3, 10}}.get_vector()};
    std::vector<std::vector<int64_t>> expected_output{{1, 0}};
    std::vector<std::vector<int64_t>> result{
        execute<float, int64_t>(function, inputs, "${BACKEND_NAME}")};
    EXPECT_EQ(expected_output, result);
}

NGRAPH_TEST(onnx_${BACKEND_NAME}, model_batchnorm_default)
{
    // Batch Normalization with default parameters
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/batchnorm_default.prototxt"));

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

    Outputs outputs{execute(function, inputs, "${BACKEND_NAME}")};
    EXPECT_TRUE(test::all_close_f(expected_outputs.front(), outputs.front()));
}

NGRAPH_TEST(onnx_${BACKEND_NAME}, model_relu)
{
    // Simple ReLU test
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/relu.prototxt"));

    Inputs inputs{{-1, -2, 0, 1, 2, 3}};
    Outputs expected_outputs{{0, 0, 0, 1, 2, 3}};

    Outputs outputs{execute(function, inputs, "${BACKEND_NAME}")};
    EXPECT_TRUE(test::all_close_f(expected_outputs.front(), outputs.front()));
}

NGRAPH_TEST(onnx_${BACKEND_NAME}, model_sum)
{
    // Simple Sum test
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/sum.prototxt"));

    // input data shape (3, )
    Inputs inputs;
    inputs.emplace_back(std::vector<float>{3.f, 0.f, 2.f});
    inputs.emplace_back(std::vector<float>{1.f, 3.f, 4.f});
    inputs.emplace_back(std::vector<float>{2.f, 6.f, 6.f});

    Outputs expected_outputs{{6.f, 9.f, 12.f}};
    Outputs outputs{execute(function, inputs, "${BACKEND_NAME}")};
    EXPECT_TRUE(test::all_close_f(expected_outputs.front(), outputs.front()));
}

NGRAPH_TEST(onnx_${BACKEND_NAME}, model_sum_one_input)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/sum_one_input.prototxt"));

    // input data shape (3, )
    Inputs inputs{{3.f, 0.f, 2.f}};
    Outputs expected_outputs{{3.f, 0.f, 2.f}};
    Outputs outputs{execute(function, inputs, "${BACKEND_NAME}")};
    EXPECT_TRUE(test::all_close_f(expected_outputs.front(), outputs.front()));
}

NGRAPH_TEST(onnx_${BACKEND_NAME}, model_min_two_inputs)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/min_two_inputs.prototxt"));

    // input data shape (3, )
    Inputs inputs;
    inputs.emplace_back(std::vector<float>{1.f, 2.f, 1.f});
    inputs.emplace_back(std::vector<float>{1.f, 4.f, 4.f});

    Outputs expected_outputs{{1.f, 2.f, 1.f}};
    Outputs outputs{execute(function, inputs, "${BACKEND_NAME}")};
    EXPECT_TRUE(test::all_close_f(expected_outputs.front(), outputs.front()));
}

NGRAPH_TEST(onnx_${BACKEND_NAME}, model_max)
{
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/max.prototxt"));

    // input data shape (3, )
    Inputs inputs;
    inputs.emplace_back(std::vector<float>{3.f, 2.f, 1.f});
    inputs.emplace_back(std::vector<float>{1.f, 4.f, 4.f});
    inputs.emplace_back(std::vector<float>{2.f, 5.f, 3.f});

    Outputs expected_outputs{{3.f, 5.f, 4.f}};
    Outputs outputs{execute(function, inputs, "${BACKEND_NAME}")};
    EXPECT_TRUE(test::all_close_f(expected_outputs.front(), outputs.front()));
}

NGRAPH_TEST(onnx_${BACKEND_NAME}, model_mean)
{
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/mean.prototxt"));

    // input data shape (3, )
    Inputs inputs;
    inputs.emplace_back(std::vector<float>{3.f, 0.f, 2.f});
    inputs.emplace_back(std::vector<float>{1.f, 3.f, 4.f});
    inputs.emplace_back(std::vector<float>{2.f, 6.f, 6.f});

    Outputs expected_outputs{{2.f, 3.f, 4.f}};
    Outputs outputs{execute(function, inputs, "${BACKEND_NAME}")};
    EXPECT_TRUE(test::all_close_f(expected_outputs.front(), outputs.front()));
}

NGRAPH_TEST(onnx_${BACKEND_NAME}, model_gemm_abc)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/gemm_abc.prototxt"));

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

    Outputs outputs{execute(function, inputs, "${BACKEND_NAME}")};
    EXPECT_TRUE(test::all_close_f(expected_outputs.front(), outputs.front()));
}

NGRAPH_TEST(onnx_${BACKEND_NAME}, model_matmul)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/matmul.prototxt"));

    std::vector<std::vector<float>> inputs;

    inputs.emplace_back(
        test::NDArray<float, 2>({{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}}).get_vector());

    inputs.emplace_back(
        test::NDArray<float, 2>({{13, 14, 15}, {16, 17, 18}, {19, 20, 21}, {22, 23, 24}})
            .get_vector());

    Outputs expected_outputs{
        test::NDArray<float, 2>({{190, 200, 210}, {470, 496, 522}, {750, 792, 834}}).get_vector()};

    Outputs outputs{execute(function, inputs, "${BACKEND_NAME}")};
    EXPECT_TRUE(test::all_close_f(expected_outputs.front(), outputs.front()));
}

NGRAPH_TEST(onnx_${BACKEND_NAME}, model_softmax)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/softmax.prototxt"));

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

    auto result_vectors = execute(function, inputs, "${BACKEND_NAME}");
    EXPECT_TRUE(test::all_close_f(expected_output, result_vectors.front()));
}

NGRAPH_TEST(onnx_${BACKEND_NAME}, model_sub)
{
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/sub.prototxt"));

    Inputs inputs;
    inputs.emplace_back(test::NDArray<float, 3>({{{1, 2, 3}}}).get_vector());

    inputs.emplace_back(test::NDArray<float, 3>({{{4, 5, 7}}}).get_vector());

    auto expected_output = test::NDArray<float, 3>({{{-3, -3, -4}}}).get_vector();

    auto result_vectors = execute(function, inputs, "${BACKEND_NAME}");
    EXPECT_TRUE(test::all_close_f(expected_output, result_vectors.front()));
}

NGRAPH_TEST(onnx_${BACKEND_NAME}, model_div)
{
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/div.prototxt"));

    Inputs inputs;
    inputs.emplace_back(test::NDArray<float, 3>({{{1, 2, 3}}}).get_vector());

    inputs.emplace_back(test::NDArray<float, 3>({{{1, 4, 12}}}).get_vector());

    auto expected_output = test::NDArray<float, 3>({{{1, 0.5, 0.25}}}).get_vector();

    auto result_vectors = execute(function, inputs, "${BACKEND_NAME}");
    EXPECT_TRUE(test::all_close_f(expected_output, result_vectors.front()));
}

NGRAPH_TEST(onnx_${BACKEND_NAME}, model_add_bcast)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/add_bcast.prototxt"));

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

    Outputs outputs{execute(function, inputs, "${BACKEND_NAME}")};
    EXPECT_TRUE(test::all_close_f(expected_output.front(), outputs.front()));
}

NGRAPH_TEST(onnx_${BACKEND_NAME}, model_reduce_log_sum)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/reduce_log_sum.prototxt"));

    // input data shape (1, 1, 4, 4)
    Inputs inputs{
        test::NDArray<float, 4>({{{{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}}}})
            .get_vector()};

    // output data shape (1,)
    Outputs expected_outputs{test::NDArray<float, 4>({{{{2.77258872f}}}}).get_vector()};

    Outputs outputs{execute(function, inputs, "${BACKEND_NAME}")};
    EXPECT_TRUE(test::all_close_f(expected_outputs.front(), outputs.front()));
}

NGRAPH_TEST(onnx_${BACKEND_NAME}, model_reduce_log_sum_exp)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/reduce_log_sum_exp.prototxt"));

    // input data shape (1, 1, 4, 4)
    Inputs inputs{
        test::NDArray<float, 4>({{{{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}}}})
            .get_vector()};

    // output data shape (1,)
    Outputs expected_outputs{test::NDArray<float, 4>({{{{3.77258872f}}}}).get_vector()};

    Outputs outputs{execute(function, inputs, "${BACKEND_NAME}")};
    EXPECT_TRUE(test::all_close_f(expected_outputs.front(), outputs.front()));
}

NGRAPH_TEST(onnx_${BACKEND_NAME}, model_reduce_l1)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/reduce_l1.prototxt"));

    // input data shape (1, 1, 4, 4)
    Inputs inputs{
        test::NDArray<float, 4>({{{{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}}}})
            .get_vector()};

    // output data shape (1,)
    Outputs expected_outputs{test::NDArray<float, 4>({{{{16}}}}).get_vector()};

    Outputs outputs{execute(function, inputs, "${BACKEND_NAME}")};
    EXPECT_TRUE(test::all_close_f(expected_outputs.front(), outputs.front()));
}

NGRAPH_TEST(onnx_${BACKEND_NAME}, model_reduce_l2)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/reduce_l2.prototxt"));

    // input data shape (1, 1, 4, 4)
    Inputs inputs{
        test::NDArray<float, 4>({{{{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}}}})
            .get_vector()};

    // output data shape (1,)
    Outputs expected_outputs{test::NDArray<float, 4>({{{{4}}}}).get_vector()};

    Outputs outputs{execute(function, inputs, "${BACKEND_NAME}")};
    EXPECT_TRUE(test::all_close_f(expected_outputs.front(), outputs.front()));
}

NGRAPH_TEST(onnx_${BACKEND_NAME}, model_reduce_max)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/reduce_max.prototxt"));

    // input data shape (1, 1, 4, 4)
    Inputs inputs{
        test::NDArray<float, 4>({{{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}}}})
            .get_vector()};

    // output data shape (1,)
    Outputs expected_outputs{test::NDArray<float, 4>({{{{16}}}}).get_vector()};

    Outputs outputs{execute(function, inputs, "${BACKEND_NAME}")};
    EXPECT_TRUE(test::all_close_f(expected_outputs.front(), outputs.front()));
}

NGRAPH_TEST(onnx_${BACKEND_NAME}, model_reduce_mean)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/reduce_mean.prototxt"));

    // input data shape (1, 1, 4, 4)
    Inputs inputs{
        test::NDArray<float, 4>({{{{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}}}})
            .get_vector()};

    // output data shape (1,)
    Outputs expected_outputs{test::NDArray<float, 4>({{{{1}}}}).get_vector()};

    Outputs outputs{execute(function, inputs, "${BACKEND_NAME}")};
    EXPECT_TRUE(test::all_close_f(expected_outputs.front(), outputs.front()));
}

NGRAPH_TEST(onnx_${BACKEND_NAME}, model_reduce_min)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/reduce_min.prototxt"));

    // input data shape (1, 1, 4, 4)
    Inputs inputs{
        test::NDArray<float, 4>({{{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}}}})
            .get_vector()};

    // output data shape (1,)
    Outputs expected_outputs{test::NDArray<float, 4>({{{{1}}}}).get_vector()};

    Outputs outputs{execute(function, inputs, "${BACKEND_NAME}")};
    EXPECT_TRUE(test::all_close_f(expected_outputs.front(), outputs.front()));
}

NGRAPH_TEST(onnx_${BACKEND_NAME}, model_reduce_prod)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/reduce_prod.prototxt"));

    // input data shape (1, 1, 4, 4)
    Inputs inputs{
        test::NDArray<float, 4>({{{{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}}}})
            .get_vector()};

    // output data shape (1,)
    Outputs expected_outputs{test::NDArray<float, 4>({{{{1}}}}).get_vector()};

    Outputs outputs{execute(function, inputs, "${BACKEND_NAME}")};
    EXPECT_TRUE(test::all_close_f(expected_outputs.front(), outputs.front()));
}

NGRAPH_TEST(onnx_${BACKEND_NAME}, model_reduce_sum)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/reduce_sum.prototxt"));

    // input data shape (1, 1, 4, 4)
    Inputs inputs{
        test::NDArray<float, 4>({{{{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}}}})
            .get_vector()};

    // output data shape (1,)
    Outputs expected_outputs{test::NDArray<float, 4>({{{{16}}}}).get_vector()};

    Outputs outputs{execute(function, inputs, "${BACKEND_NAME}")};
    EXPECT_TRUE(test::all_close_f(expected_outputs.front(), outputs.front()));
}

NGRAPH_TEST(onnx_${BACKEND_NAME}, model_reduce_sum_square)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/reduce_sum_square.prototxt"));

    // input data shape (1, 1, 4, 4)
    Inputs inputs{
        test::NDArray<float, 4>({{{{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}}}})
            .get_vector()};

    // output data shape (1,)
    Outputs expected_outputs{test::NDArray<float, 4>({{{{16}}}}).get_vector()};

    Outputs outputs{execute(function, inputs, "${BACKEND_NAME}")};
    EXPECT_TRUE(test::all_close_f(expected_outputs.front(), outputs.front()));
}

NGRAPH_TEST(onnx_${BACKEND_NAME}, model_shape)
{
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/shape.prototxt"));

    Inputs inputs;
    inputs.emplace_back(test::NDArray<float, 3>(
                            {{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                             {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                             {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}})
                            .get_vector());

    std::vector<std::vector<int64_t>> expected_output{{3, 4, 5}};

    std::vector<std::vector<int64_t>> outputs =
        execute<float, int64_t>(function, inputs, "${BACKEND_NAME}");
    EXPECT_TRUE(test::all_close(expected_output.front(), outputs.front()));
}

NGRAPH_TEST(onnx_${BACKEND_NAME}, model_elu)
{
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/elu.prototxt"));

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

    Outputs outputs{execute(function, inputs, "${BACKEND_NAME}")};
    EXPECT_TRUE(test::all_close_f(expected_output.front(), outputs.front()));
}

NGRAPH_TEST(onnx_${BACKEND_NAME}, model_leaky_relu)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/leaky_relu.prototxt"));

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

    Outputs outputs{execute(function, inputs, "${BACKEND_NAME}")};
    EXPECT_TRUE(test::all_close_f(expected_output.front(), outputs.front()));
}

NGRAPH_TEST(onnx_${BACKEND_NAME}, model_prelu)
{
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/prelu.prototxt"));

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

    Outputs outputs{execute(function, inputs, "${BACKEND_NAME}")};
    EXPECT_TRUE(test::all_close_f(expected_output.front(), outputs.front()));
}

NGRAPH_TEST(onnx_${BACKEND_NAME}, model_selu)
{
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/selu.prototxt"));

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

    Outputs outputs{execute(function, inputs, "${BACKEND_NAME}")};
    EXPECT_TRUE(test::all_close_f(expected_output.front(), outputs.front()));
}

NGRAPH_TEST(onnx_${BACKEND_NAME}, model_sigmoid)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/sigmoid.prototxt"));

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

    Outputs outputs{execute(function, inputs, "${BACKEND_NAME}")};
    EXPECT_TRUE(test::all_close_f(expected_output.front(), outputs.front()));
}

NGRAPH_TEST(onnx_${BACKEND_NAME}, model_tanh)
{
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/tanh.prototxt"));

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

    Outputs outputs{execute(function, inputs, "${BACKEND_NAME}")};
    EXPECT_TRUE(test::all_close_f(expected_output.front(), outputs.front()));
}

NGRAPH_TEST(onnx_${BACKEND_NAME}, model_thresholded_relu)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/thresholded_relu.prototxt"));

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

    Outputs outputs{execute(function, inputs, "${BACKEND_NAME}")};
    EXPECT_TRUE(test::all_close_f(expected_output.front(), outputs.front()));
}

NGRAPH_TEST(onnx_${BACKEND_NAME}, model_matmul_vec_ten3d)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/matmul_vec_ten3d.prototxt"));

    Inputs inputs;
    inputs.emplace_back(std::vector<float>{0.f, 1.f});
    inputs.emplace_back(
        test::NDArray<float, 3>{{{0.f}, {1.f}}, {{2.f}, {3.f}}, {{4.f}, {5.f}}}.get_vector());

    Outputs expected_output{test::NDArray<float, 2>{{1.f}, {3.f}, {5.f}}};

    Outputs outputs{execute(function, inputs, "${BACKEND_NAME}")};
    EXPECT_TRUE(test::all_close_f(expected_output.front(), outputs.front()));
}

NGRAPH_TEST(onnx_${BACKEND_NAME}, model_softplus)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/softplus.prototxt"));

    // -1.0f, 0, 1.0f, 10.f,                    normal input values for activation
    // 100.0f, -100.0f, 1000.0f, -1000.0f,      input values that leads to exp() overflow
    // FLT_MIN, FLT_MIN / 16, -FLT_MIN / 16,    min, denorm, -denorm
    // FLT_MAX, -FLT_MAX,                       max, -max;
    Inputs inputs{std::vector<float>{-1.0f,
                                     0,
                                     1.0f,
                                     10.f,
                                     100.0f,
                                     -100.0f,
                                     1000.0f,
                                     -1000.0f,
                                     FLT_MIN,
                                     FLT_MIN / 16,
                                     -FLT_MIN / 16,
                                     FLT_MAX,
                                     -FLT_MAX}};

    std::vector<float>& input = inputs.back();
    std::vector<float> output;
    auto softplus_impl = [](float x) -> float {
        if (x > 0)
        {
            return x + std::log(std::exp(-x) + 1);
        }
        else
        {
            return std::log(std::exp(x) + 1);
        }
    };

    std::transform(std::begin(input), std::end(input), std::back_inserter(output), softplus_impl);

    Outputs expected_output{output};
    Outputs outputs{execute(function, inputs, "${BACKEND_NAME}")};
    EXPECT_TRUE(test::all_close_f(expected_output.front(), outputs.front()));
}

NGRAPH_TEST(onnx_${BACKEND_NAME}, model_softplus_infinity)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/softplus.prototxt"));

    Inputs inputs{std::vector<float>{std::numeric_limits<float>::infinity(),
                                     std::numeric_limits<float>::infinity(),
                                     std::numeric_limits<float>::infinity(),
                                     std::numeric_limits<float>::infinity(),
                                     std::numeric_limits<float>::infinity(),
                                     std::numeric_limits<float>::infinity(),
                                     std::numeric_limits<float>::infinity(),
                                     std::numeric_limits<float>::infinity(),
                                     std::numeric_limits<float>::infinity(),
                                     std::numeric_limits<float>::infinity(),
                                     std::numeric_limits<float>::infinity(),
                                     std::numeric_limits<float>::infinity(),
                                     std::numeric_limits<float>::infinity()}};

    Outputs outputs{execute(function, inputs, "${BACKEND_NAME}")};
    for (float v : outputs.front())
    {
        EXPECT_TRUE(std::isinf(v));
    }
}

NGRAPH_TEST(onnx_${BACKEND_NAME}, model_sum_opset8)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/sum_opset8.prototxt"));

    Inputs inputs;
    inputs.emplace_back(std::vector<float>{1.0f, 2.0f, 3.0f});
    inputs.emplace_back(test::NDArray<float, 2>{{10.0f}, {20.0f}, {30.0f}}.get_vector());
    inputs.emplace_back(test::NDArray<float, 3>{{{100.0f}}, {{200.0f}}, {{300.0f}}}.get_vector());

    Outputs expected_output{test::NDArray<float, 3>{
        {{111.0f, 112.0f, 113.0f}, {121.0f, 122.0f, 123.0f}, {131.0f, 132.0f, 133.0f}},

        {{211.0f, 212.0f, 213.0f}, {221.0f, 222.0f, 223.0f}, {231.0f, 232.0f, 233.0f}},

        {{311.0f, 312.0f, 313.0f}, {321.0f, 322.0f, 323.0f}, {331.0f, 332.0f, 333.0f}}}
                                .get_vector()};

    Outputs outputs{execute(function, inputs, "${BACKEND_NAME}")};
    EXPECT_TRUE(test::all_close_f(expected_output.front(), outputs.front()));
}

NGRAPH_TEST(onnx_${BACKEND_NAME}, model_argmax_int32)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/argmax_int32.prototxt"));

    std::vector<std::vector<std::int32_t>> inputs{
        std::vector<std::int32_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}};

    std::vector<std::vector<std::int64_t>> expected_output{
        std::vector<std::int64_t>{1, 1, 1, 1, 1, 1}};

    std::vector<std::vector<std::int64_t>> outputs{
        execute<std::int32_t, std::int64_t>(function, inputs, "${BACKEND_NAME}")};
    EXPECT_TRUE(test::all_close(expected_output.front(), outputs.front()));
}

NGRAPH_TEST(onnx_${BACKEND_NAME}, model_argmin_int32)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/argmin_int32.prototxt"));

    std::vector<std::vector<std::int32_t>> inputs{
        std::vector<std::int32_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}};

    std::vector<std::vector<std::int64_t>> expected_output{std::vector<std::int64_t>{0, 0, 0, 0}};

    std::vector<std::vector<std::int64_t>> outputs{
        execute<std::int32_t, std::int64_t>(function, inputs, "${BACKEND_NAME}")};
    EXPECT_TRUE(test::all_close(expected_output.front(), outputs.front()));
}

NGRAPH_TEST(onnx_${BACKEND_NAME}, model_top_k)
{
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/top_k.prototxt"));

    Inputs inputs;
    inputs.emplace_back(std::vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});

    std::vector<float> expected_values_output{3, 2, 1, 7, 6, 5, 11, 10, 9};
    std::vector<std::int64_t> expected_indices_output{3, 2, 1, 3, 2, 1, 3, 2, 1};

    std::vector<std::shared_ptr<ngraph::runtime::Tensor>> result_tensors =
        prepare_and_run(function, inputs, "${BACKEND_NAME}");

    std::vector<float> values_output = read_vector<float>(result_tensors.at(0));
    std::vector<std::int64_t> indices_output = read_vector<std::int64_t>(result_tensors.at(1));

    EXPECT_TRUE(test::all_close_f(expected_values_output, values_output));
    EXPECT_TRUE(test::all_close(expected_indices_output, indices_output));
}

NGRAPH_TEST(onnx_${BACKEND_NAME}, model_sinh)
{
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/sinh.prototxt"));

    Inputs inputs{std::vector<float>{-1.0f, 0.0f, 1.0f}};
    Outputs expected_outputs{std::vector<float>{-1.1752012f, 0.f, 1.1752012f}};

    Outputs outputs{execute(function, inputs, "${BACKEND_NAME}")};

    EXPECT_TRUE(test::all_close_f(expected_outputs.front(), outputs.front()));
}

NGRAPH_TEST(onnx_${BACKEND_NAME}, model_cosh)
{
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/cosh.prototxt"));

    Inputs inputs{std::vector<float>{-1.0f, 0.0f, 1.0f}};
    Outputs expected_outputs{std::vector<float>{1.54308069f, 1.f, 1.54308069f}};

    Outputs outputs{execute(function, inputs, "${BACKEND_NAME}")};

    EXPECT_TRUE(test::all_close_f(expected_outputs.front(), outputs.front()));
}

NGRAPH_TEST(onnx_${BACKEND_NAME}, model_sign)
{
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/sign.prototxt"));

    Inputs inputs{std::vector<float>{-std::numeric_limits<float>::infinity(),
                                     -3.141592f,
                                     0.0f,
                                     2.71828f,
                                     std::numeric_limits<float>::infinity()}};

    Outputs expected_outputs{std::vector<float>{-1.0f, -1.0f, 0.0f, 1.0f, 1.0f}};

    Outputs outputs{execute<float>(function, inputs, "${BACKEND_NAME}")};

    EXPECT_TRUE(test::all_close_f(expected_outputs.front(), outputs.front()));
}

NGRAPH_TEST(onnx_${BACKEND_NAME}, model_one_hot_with_axis)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/one_hot_axis.prototxt"));

    Inputs inputs{{1.0, 9.0, 2.0, 4.0}, {1.0, 3.0}};
    Outputs expected_outputs{{1.0, 1.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                              1.0, 1.0, 1.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0,
                              1.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}};

    Outputs outputs{execute(function, inputs, "${BACKEND_NAME}")};

    EXPECT_TRUE(test::all_close_f(expected_outputs.front(), outputs.front()));
}

NGRAPH_TEST(onnx_${BACKEND_NAME}, model_one_hot_without_axis)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/one_hot_no_axis.prototxt"));

    std::vector<std::vector<std::int64_t>> inputs{{0, 7, 8}, {2, 5}};
    std::vector<std::vector<std::int64_t>> expected_outputs{{5, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                                             2, 2, 2, 2, 2, 2, 2, 5, 2, 2, 2, 2,
                                                             2, 2, 2, 2, 2, 2, 2, 2, 5, 2, 2, 2}};

    std::vector<std::vector<std::int64_t>> outputs{execute(function, inputs, "${BACKEND_NAME}")};

    EXPECT_TRUE(test::all_close(expected_outputs.front(), outputs.front()));
}

NGRAPH_TEST(onnx_${BACKEND_NAME}, model_where)
{
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/where.prototxt"));

    // conditions tensor - 3x3x3
    auto condition = std::vector<int>{
        {0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0}};

    // 1x3 tensor of "1"
    auto x1 = std::vector<int>{1, 1, 1};
    // 3x1 tensor of "2"
    auto x2 = std::vector<int>{2, 2, 2};

    std::vector<std::vector<int>> inputs;
    inputs.push_back(std::move(condition));
    inputs.push_back(std::move(x1));
    inputs.push_back(std::move(x2));

    // y = 3x3x3
    std::vector<std::vector<int>> expected_outputs{
        {2, 1, 2, 1, 2, 1, 2, 1, 2, 2, 1, 2, 1, 2, 1, 2, 1, 2, 2, 1, 2, 1, 2, 1, 2, 1, 2}};

    std::vector<std::vector<int>> outputs{execute(function, inputs, "${BACKEND_NAME}")};

    EXPECT_EQ(expected_outputs.front(), outputs.front());
}

NGRAPH_TEST(onnx_${BACKEND_NAME}, model_erf)
{
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/erf.prototxt"));

    Inputs inputs;
    inputs.emplace_back(test::NDArray<float, 2>{
        {-std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity()},
        {-3.141592f, 0.0f},
        {0.5f, 1.0f}}.get_vector());

    const std::vector<float> expected_outputs = test::NDArray<float, 2>{
        {-1.0f, 1.0f},
        {-0.99999112f, 0.0f},
        {0.52049988f, 0.84270079f}}.get_vector();

    const Outputs outputs{execute(function, inputs, "${BACKEND_NAME}")};

    EXPECT_TRUE(test::all_close_f(expected_outputs, outputs.front()));
}
