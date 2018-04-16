/*******************************************************************************
* Copyright 2018 Intel Corporation
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
#include "ngraph/op/sum.hpp"
#include "ngraph/pass/graph_rewrite.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/reshape_elimination.hpp"
#include "ngraph/pattern/matcher.hpp"
#include "ngraph/pattern/op/any.hpp"
#include "ngraph/pattern/op/label.hpp"
#include "ngraph/serializer.hpp"
#include "ngraph/util.hpp"
#include "nlohmann/json.hpp"
#include "util/matcher.hpp"
#include "util/test_tools.hpp"

using namespace ngraph;
using namespace std;

TEST(reshape_elimination, remove_reshape)
{
    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ReshapeElimination>();
    const string json_path = file_util::path_join(SERIALIZED_ZOO, "mxnet/bn_fprop.json");
    const string json_string = file_util::read_file_to_string(json_path);
    stringstream ss(json_string);
    shared_ptr<Function> func = ngraph::deserialize(ss);
    size_t count_before = count_ops_of_type<op::Reshape>(func);
    pass_manager.run_passes(func);
    size_t count_after = count_ops_of_type<op::Reshape>(func);
    ASSERT_TRUE(count_after < count_before);
}

TEST(reshape_elimination, remove_tranpose)
{
    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ReshapeElimination>();
    const string json_path = file_util::path_join(SERIALIZED_ZOO, "mxnet/tranpose.json");
    const string json_string = file_util::read_file_to_string(json_path);
    stringstream ss(json_string);
    shared_ptr<Function> func = ngraph::deserialize(ss);
    size_t count_before = count_ops_of_type<op::Reshape>(func);
    pass_manager.run_passes(func);
    size_t count_after = count_ops_of_type<op::Reshape>(func);
    ASSERT_TRUE(count_after < count_before);
}

TEST(reshape_elimination, bn_bprop_rewrite)
{
    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ReshapeElimination>();
    const string json_path = file_util::path_join(SERIALIZED_ZOO, "mxnet/bn_bprop.json");
    const string json_string = file_util::read_file_to_string(json_path);
    stringstream ss(json_string);
    shared_ptr<Function> func = ngraph::deserialize(ss);
    size_t count_before = count_ops_of_type<op::Reshape>(func);
    pass_manager.run_passes(func);
    size_t count_after = count_ops_of_type<op::Reshape>(func);
    ASSERT_TRUE(count_after < count_before);
}

TEST(reshape_elimination, dot_transpose_to_dot_w_transpose_args)
{
    Shape shape_w{2, 4};
    Shape shape_x{4, 1};
    auto W = make_shared<op::Parameter>(element::f32, shape_w);
    auto x = make_shared<op::Parameter>(element::f32, shape_x);

    auto dot = make_shared<op::Dot>(W, x);
    auto reshape_dot = std::make_shared<op::Reshape>(dot, AxisVector{1, 0}, Shape{1, 2});
    auto graph = make_shared<op::Abs>(reshape_dot);

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ReshapeElimination>();
    auto func = make_shared<Function>(graph, op::ParameterVector{W, x});
    pass_manager.run_passes(func);
    auto gdot = graph->get_argument(0);
    ASSERT_TRUE(std::dynamic_pointer_cast<op::Dot>(gdot));
    ASSERT_TRUE(std::dynamic_pointer_cast<op::Reshape>(gdot->get_argument(0)));
    ASSERT_TRUE(std::dynamic_pointer_cast<op::Reshape>(gdot->get_argument(1)));
    ASSERT_EQ(gdot->get_argument(0)->get_argument(0), x);
    ASSERT_EQ(gdot->get_argument(1)->get_argument(0), W);
    ASSERT_EQ(gdot->get_shape(), (Shape{1, 2}));
}
