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

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <list>
#include <memory>

#include "gtest/gtest.h"

#include "ngraph/file_util.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/relu.hpp"
#include "ngraph/pass/core_fusion.hpp"
#include "ngraph/pass/graph_rewrite.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pattern/matcher.hpp"
#include "ngraph/pattern/op/label.hpp"
#include "ngraph/pattern/op/skip.hpp"
#include "ngraph/serializer.hpp"
#include "ngraph/util.hpp"
#include "nlohmann/json.hpp"
#include "util/autodiff/backprop_function.hpp"
#include "util/matcher.hpp"
#include "util/test_tools.hpp"

using namespace ngraph;
using namespace std;

TEST(core_fusion, core_fusion_pass_basic)
{
    auto shape_a = Shape{1, 5};
    auto A = op::Constant::create(element::f32, shape_a, {0, 0, 0, 0, 0});
    auto B = make_shared<op::Parameter>(element::f32, shape_a);
    auto max = make_shared<op::Maximum>(A, B);
    auto graph = make_shared<op::Abs>(max);
    pass::Manager pass_manager;
    pass_manager.register_pass<pass::CoreFusion>();
    auto func = make_shared<Function>(graph, op::ParameterVector{B});
    pass_manager.run_passes(func);
    ASSERT_NE(std::dynamic_pointer_cast<op::Relu>(graph->get_argument(0)), nullptr);
}

TEST(core_fusion, sigmoid_fprop_fusion)
{
    pass::Manager pass_manager;
    pass_manager.register_pass<pass::CoreFusion>();
    const string json_path = file_util::path_join(SERIALIZED_ZOO, "mxnet/Graph_fprop_sigmoid.json");
    const string json_string = file_util::read_file_to_string(json_path);
    stringstream ss(json_string);
    shared_ptr<Function> func = ngraph::deserialize(ss);
    pass_manager.run_passes(func);
    size_t ccg = count_ops_of_type<op::Sigmoid>(func);
    ASSERT_EQ(ccg, 1);
}

TEST(core_fusion, sigmoid_bprop_fusion)
{
    const string json_path = file_util::path_join(SERIALIZED_ZOO, "mxnet/Graph_fprop_sigmoid.json");
    const string json_string = file_util::read_file_to_string(json_path);
    stringstream ss(json_string);
    shared_ptr<Function> func = ngraph::deserialize(ss);
    auto df = autodiff::backprop_function(func);
    auto backend = runtime::Backend::create("CPU");
    backend->compile(df);
    size_t ccg = count_ops_of_type<op::SigmoidBackprop>(df);
    ASSERT_EQ(ccg, 1);
}
