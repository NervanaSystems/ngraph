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
#include "ngraph/ngraph.hpp"
#include "ngraph/onnx_import/graph.hpp"
#include "ngraph/onnx_import/model.hpp"
#include "ngraph/onnx_util.hpp"
#include "onnx.pb.h"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

TEST(onnx, basic)
{
    using namespace ngraph;

    const string filepath = file_util::path_join(SERIALIZED_ZOO, "onnx/cntk_FeedForward.onnx");
    onnx::ModelProto model_proto = ngraph::onnx_util::load_model_file(filepath);
    ASSERT_EQ("CNTK", model_proto.producer_name());

    onnx_import::Model model_wrapper = onnx_import::Model(model_proto);
    std::stringstream model_stream;
    model_stream << model_wrapper;
    ASSERT_EQ("<Model: CNTK>", model_stream.str());

    onnx_import::Graph graph_wrapper = onnx_import::Graph(model_proto.graph());
    std::stringstream graph_stream;
    graph_stream << graph_wrapper;
    ASSERT_EQ("<Graph: CNTKGraph>", graph_stream.str());

    auto value = graph_wrapper.get_values()[0];
    std::stringstream value_stream;
    value_stream << value;
    ASSERT_EQ("<ValueInfo: Parameter15>", value_stream.str());
    ASSERT_EQ(element::f32, value.get_element_type());
}
