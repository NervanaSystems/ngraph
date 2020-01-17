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
        EXPECT_EQ(static_cast<size_t>(input_ps[1]), 5);
    }

    const auto& graph_outputs = function->get_results();
    EXPECT_EQ(graph_outputs.size(), 1);

    const auto out = *(graph_outputs.cbegin());
    const auto& out_ps = out->get_output_partial_shape(0);
    EXPECT_TRUE(out_ps.rank().is_static());
    EXPECT_EQ(static_cast<size_t>(out_ps.rank()), 2);
    EXPECT_TRUE(out_ps[0].is_dynamic());
    EXPECT_TRUE(out_ps[1].is_static());
    EXPECT_EQ(static_cast<size_t>(out_ps[1]), 5);
}
