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

NGRAPH_TEST(onnx_${BACKEND_NAME}, model_lstm_fwd_with_clip)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/lstm_fwd_with_clip.prototxt"));

    Inputs inputs{};
    // X
    inputs.emplace_back(std::vector<float>{-0.455351, -0.276391, -0.185934, -0.269585});

    // W
    inputs.emplace_back(std::vector<float>{-0.494659f,
                                           0.0453352f,
                                           -0.487793f,
                                           0.417264f,
                                           -0.0175329f,
                                           0.489074f,
                                           -0.446013f,
                                           0.414029f,
                                           -0.0091708f,
                                           -0.255364f,
                                           -0.106952f,
                                           -0.266717f,
                                           -0.0888852f,
                                           -0.428709f,
                                           -0.283349f,
                                           0.208792f});

    // R
    inputs.emplace_back(std::vector<float>{0.146626f,
                                           -0.0620289f,
                                           -0.0815302f,
                                           0.100482f,
                                           -0.219535f,
                                           -0.306635f,
                                           -0.28515f,
                                           -0.314112f,
                                           -0.228172f,
                                           0.405972f,
                                           0.31576f,
                                           0.281487f,
                                           -0.394864f,
                                           0.42111f,
                                           -0.386624f,
                                           -0.390225f});

    // B
    inputs.emplace_back(std::vector<float>{0.381619f,
                                           0.0323954f,
                                           -0.14449f,
                                           0.420804f,
                                           -0.258721f,
                                           0.45056f,
                                           -0.250755f,
                                           0.0967895f,
                                           0.0f,
                                           0.0f,
                                           0.0f,
                                           0.0f,
                                           0.0f,
                                           0.0f,
                                           0.0f,
                                           0.0f});
    // P
    inputs.emplace_back(std::vector<float>{0.2345f, 0.5235f, 0.4378f, 0.3475f, 0.8927f, 0.3456f});

    Outputs expected_output{};
    // Y_data
    expected_output.emplace_back(
        std::vector<float>{-0.02280854f, 0.02744377f, -0.03516197f, 0.03875681f});
    // Y_h_data
    expected_output.emplace_back(std::vector<float>{-0.03516197f, 0.03875681f});
    // Y_c_data
    expected_output.emplace_back(std::vector<float>{-0.07415761f, 0.07395997f});

    Outputs outputs{execute(function, inputs, "${BACKEND_NAME}")};

    EXPECT_TRUE(outputs.size() == expected_output.size());
    for (std::size_t i{0}; i < expected_output.size(); ++i)
    {
        // We have to enlarge tolerance bits to 3 - it's only one bit more than default value.
        // The discrepancies may occur at most on 7th decimal position.
        EXPECT_TRUE(test::all_close_f(expected_output.at(i), outputs.at(i), 3));
    }
}

NGRAPH_TEST(onnx_${BACKEND_NAME}, model_lstm_fwd_mixed_seq)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/lstm_fwd_mixed_seq.prototxt"));

    int hidden_size{3};
    int parameters_cout{5};

    // X
    std::vector<float> in_x{1.f, 2.f, 10.f, 11.f};
    // W
    std::vector<float> in_w{0.1f, 0.2f, 0.3f, 0.4f, 1.f, 2.f, 3.f, 4.f, 10.f, 11.f, 12.f, 13.f};
    // R
    std::vector<float> in_r(4 * hidden_size * hidden_size, 0.1f);
    // B
    std::vector<float> in_b(8 * hidden_size, 0.0f);

    std::vector<int> in_seq_lengths{1, 2};

    std::vector<float> out_y_data{0.28828835f,
                                  0.36581863f,
                                  0.45679406f,
                                  0.34526032f,
                                  0.47220859f,
                                  0.55850911f,
                                  0.f,
                                  0.f,
                                  0.f,
                                  0.85882828f,
                                  0.90703777f,
                                  0.92382453f};

    std::vector<float> out_y_h_data{
        0.28828835f, 0.36581863f, 0.45679406f, 0.85882828f, 0.90703777f, 0.92382453f};

    std::vector<float> out_y_c_data{
        0.52497941f, 0.54983425f, 0.5744428f, 1.3249796f, 1.51063104f, 1.61451544f};

    Outputs expected_output;

    expected_output.emplace_back(out_y_data);
    expected_output.emplace_back(out_y_h_data);
    expected_output.emplace_back(out_y_c_data);

    auto backend = ngraph::runtime::Backend::create("${BACKEND_NAME}");
    auto parameters = function->get_parameters();

    EXPECT_TRUE(parameters.size() == parameters_cout);

    std::vector<std::shared_ptr<ngraph::runtime::Tensor>> arg_tensors;

    auto add_tensor = [&arg_tensors, &backend](const std::vector<float>& v,
                                               const std::shared_ptr<ngraph::op::Parameter>& p) {
        auto t = backend->create_tensor(p->get_element_type(), p->get_shape());
        copy_data(t, v);
        arg_tensors.push_back(t);
    };

    add_tensor(in_x, parameters.at(0));
    add_tensor(in_w, parameters.at(1));
    add_tensor(in_r, parameters.at(2));
    add_tensor(in_b, parameters.at(3));

    auto t_in_seq_lengths =
        backend->create_tensor(parameters.at(4)->get_element_type(), parameters.at(4)->get_shape());
    copy_data(t_in_seq_lengths, in_seq_lengths);
    arg_tensors.push_back(t_in_seq_lengths);

    auto results = function->get_results();
    std::vector<std::shared_ptr<ngraph::runtime::Tensor>> result_tensors(results.size());

    for (std::size_t i{0}; i < results.size(); ++i)
    {
        result_tensors.at(i) =
            backend->create_tensor(results.at(i)->get_element_type(), results.at(i)->get_shape());
    }

    auto handle = backend->compile(function);
    handle->call_with_validate(result_tensors, arg_tensors);

    Outputs outputs;
    for (auto rt : result_tensors)
    {
        outputs.push_back(read_vector<float>(rt));
    }

    EXPECT_TRUE(outputs.size() == expected_output.size());
    for (std::size_t i{0}; i < expected_output.size(); ++i)
    {
        // We have to enlarge tolerance bits to 3 - it's only one bit more than default value.
        // The discrepancies may occur at most on 7th decimal position.
        EXPECT_TRUE(test::all_close_f(expected_output.at(i), outputs.at(i), 3));
    }
}
