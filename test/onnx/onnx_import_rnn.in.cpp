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
#include "util/test_case.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace ngraph;

static std::string s_manifest = "${MANIFEST}";

NGRAPH_TEST(onnx_${BACKEND_NAME}, model_lstm_fwd_with_clip)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/lstm_fwd_with_clip.prototxt"));

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_input<float>({-0.455351, -0.276391, -0.185934, -0.269585}); // X
    test_case.add_input<float>({-0.494659f,                                   // W
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
                                0.208792f}); // W
    test_case.add_input<float>({0.146626f,
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
                                -0.390225f}); // R

    test_case.add_input<float>({0.381619f,
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
                                0.0f});                                                 // B
    test_case.add_input<float>({0.2345f, 0.5235f, 0.4378f, 0.3475f, 0.8927f, 0.3456f}); // P

    test_case.add_expected_output<float>(
        Shape{2, 1, 1, 2}, {-0.02280854f, 0.02744377f, -0.03516197f, 0.03875681f});    // Y_data
    test_case.add_expected_output<float>(Shape{1, 1, 2}, {-0.03516197f, 0.03875681f}); // Y_h_data
    test_case.add_expected_output<float>(Shape{1, 1, 2}, {-0.07415761f, 0.07395997f}); // Y_c_data

    // We have to enlarge tolerance bits to 3 - it's only one bit more than default value.
    // The discrepancies may occur at most on 7th decimal position.
    test_case.set_tolerance(3);
    test_case.run();
}

NGRAPH_TEST(onnx_${BACKEND_NAME}, model_lstm_fwd_mixed_seq)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/lstm_fwd_mixed_seq.prototxt"));

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    int hidden_size{3};
    test_case.add_input<float>({1.f, 2.f, 10.f, 11.f}); // X
    test_case.add_input<float>(
        {0.1f, 0.2f, 0.3f, 0.4f, 1.f, 2.f, 3.f, 4.f, 10.f, 11.f, 12.f, 13.f});    // W
    test_case.add_input(std::vector<float>(4 * hidden_size * hidden_size, 0.1f)); // R
    test_case.add_input(std::vector<float>(8 * hidden_size, 0.0f));               // B
    test_case.add_input<int>({1, 2});                                             // seq_lengths

    test_case.add_expected_output<float>(Shape{2, 1, 2, 3},
                                         {0.28828835f,
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
                                          0.92382453f}); // Y_data
    test_case.add_expected_output<float>(
        Shape{1, 2, 3},
        {0.28828835f, 0.36581863f, 0.45679406f, 0.85882828f, 0.90703777f, 0.92382453f}); // Y_h_data
    test_case.add_expected_output<float>(
        Shape{1, 2, 3},
        {0.52497941f, 0.54983425f, 0.5744428f, 1.3249796f, 1.51063104f, 1.61451544f}); // Y_c_data

    // We have to enlarge tolerance bits to 3 - it's only one bit more than default value.
    // The discrepancies may occur at most on 7th decimal position.
    test_case.set_tolerance(3);
    test_case.run();
}

NGRAPH_TEST(onnx_${BACKEND_NAME}, model_lstm_fwd_hardsigmoid_activation)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/lstm_fwd_hardsigmoid_activation.prototxt"));

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");

    // X
    test_case.add_input<float>({-0.455351f, -0.276391f, -0.185934f, -0.269585f});
    // W
    test_case.add_input<float>({-0.494659f,
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
    test_case.add_input<float>({0.146626f,
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

    // Y
    test_case.add_expected_output<float>(Shape{2, 1, 1, 2},
                                         {0.09086666f, 0.04378549f, 0.12914555f, 0.00257774f});
    // Y_h
    test_case.add_expected_output<float>(Shape{1, 1, 2}, {0.12914555f, 0.00257774f});
    // Y_c
    test_case.add_expected_output<float>(Shape{1, 1, 2}, {0.19017234f, 0.00356848f});

    // The discrepancies occur at most at 18th mantissa bit - 8th decimal position.
    test_case.set_tolerance(6);
    test_case.run();
}
