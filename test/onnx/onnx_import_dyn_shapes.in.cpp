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

#include "gtest/gtest.h"
#include "ngraph/file_util.hpp"
#include "ngraph/frontend/onnx_import/onnx.hpp"
#include "util/test_control.hpp"
#include "util/type_prop.hpp"
#include "util/test_tools.hpp"

using namespace ngraph;
using namespace ngraph::onnx_import;

static std::string s_manifest = "${MANIFEST}";

NGRAPH_TEST(onnx_${BACKEND_NAME}, onnx_dynamic_dims_to_ngraph_dynamic_dims)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/dynamic_shapes/ab_plus_c.prototxt"));

    const auto& graph_inputs = function->get_parameters();
    EXPECT_EQ(graph_inputs.size(), 3);

    for (const auto& input : graph_inputs)
    {
        const auto& input_ps = input->get_partial_shape();
        EXPECT_TRUE(input_ps.is_dynamic());
        EXPECT_TRUE(input_ps.rank().is_static());
        EXPECT_EQ(static_cast<size_t>(input_ps.rank()), 2);
        EXPECT_TRUE(input_ps[0].is_dynamic());
        EXPECT_TRUE(input_ps[1].is_static());
        EXPECT_EQ(static_cast<size_t>(input_ps[1]), 2);
    }

    const auto& graph_outputs = function->get_results();
    EXPECT_EQ(graph_outputs.size(), 1);

    const auto out = *(graph_outputs.cbegin());
    const auto& out_ps = out->get_output_partial_shape(0);
    EXPECT_TRUE(out_ps.rank().is_static());
    EXPECT_EQ(static_cast<size_t>(out_ps.rank()), 2);
    EXPECT_TRUE(out_ps[0].is_dynamic());
    EXPECT_TRUE(out_ps[1].is_static());
    EXPECT_EQ(static_cast<size_t>(out_ps[1]), 2);
}

NGRAPH_TEST(onnx_${BACKEND_NAME}, ab_plus_c_dynamic_execution)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/dynamic_shapes/ab_plus_c.prototxt"));

    auto backend = runtime::Backend::create("${BACKEND_NAME}", true);
    auto executable = backend->compile(function);

    auto out_tensor = backend->create_dynamic_tensor(function->get_output_element_type(0),
                                                     function->get_output_partial_shape(0));

    auto input_A = backend->create_tensor(element::i64, Shape{1, 2});
    auto input_B = backend->create_tensor(element::i64, Shape{1, 2});
    auto input_C = backend->create_tensor(element::i64, Shape{1, 2});

    copy_data(input_A, std::vector<int64_t>{1, 2});
    copy_data(input_B, std::vector<int64_t>{3, 4});
    copy_data(input_C, std::vector<int64_t>{5, 6});

    executable->call_with_validate({out_tensor}, {input_A, input_B, input_C});

    const auto results = read_vector<int64_t>(out_tensor);
    const auto expected_results = std::vector<int64_t>{8, 14};

    EXPECT_TRUE(results == expected_results);
}
