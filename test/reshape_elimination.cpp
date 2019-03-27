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
#include "ngraph/pass/visualize_tree.hpp"
#include "ngraph/pattern/matcher.hpp"
#include "ngraph/pattern/op/label.hpp"
#include "ngraph/pattern/op/skip.hpp"
#include "ngraph/serializer.hpp"
#include "ngraph/util.hpp"
#include "ngraph/util.hpp"
#include "nlohmann/json.hpp"
#include "util/all_close.hpp"
#include "util/matcher.hpp"
#include "util/random.hpp"
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
    auto func = make_shared<Function>(graph, ParameterVector{W, x});
    pass_manager.run_passes(func);
    auto gdot = graph->get_argument(0);
    ASSERT_TRUE(std::dynamic_pointer_cast<op::Dot>(gdot));
    ASSERT_TRUE(std::dynamic_pointer_cast<op::Reshape>(gdot->get_argument(0)));
    ASSERT_TRUE(std::dynamic_pointer_cast<op::Reshape>(gdot->get_argument(1)));
    ASSERT_EQ(gdot->get_argument(0)->get_argument(0), x);
    ASSERT_EQ(gdot->get_argument(1)->get_argument(0), W);
    ASSERT_EQ(gdot->get_shape(), (Shape{1, 2}));
}

TEST(reshape_elimination, recurrent_reshapes)
{
    Shape shape_a{128, 2048, 1, 1};
    auto generate_func = [shape_a]() {
        auto A = make_shared<op::Parameter>(element::f32, shape_a);

        auto reshape_1 = make_shared<op::Reshape>(A, AxisVector{0, 1, 2, 3}, shape_a);
        auto reshape_2 = make_shared<op::Reshape>(reshape_1, AxisVector{0, 1, 2, 3}, shape_a);
        auto reshape_3 = make_shared<op::Reshape>(reshape_2, AxisVector{0, 1, 2, 3}, shape_a);
        auto f_ = make_shared<Function>(NodeVector{reshape_3}, ParameterVector{A});
        return f_;
    };

    auto baseline_f = generate_func();
    auto optimized_f = generate_func();
    auto baseline_input_shape = baseline_f->get_parameters().at(0)->get_shape();

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::VisualizeTree>("before_recurrent_reshapes.pdf");
    pass_manager.register_pass<pass::RecurrentReshapeElimination>();
    //pass_manager.register_pass<pass::ReshapeElimination>();
    pass_manager.register_pass<pass::VisualizeTree>("after_recurrent_reshapes.pdf");
    pass_manager.run_passes(optimized_f);

    test::Uniform<float> rng(0.0f, 100.0f);
    vector<vector<float>> args;
    vector<float> tensor_val(shape_size(baseline_input_shape));
    rng.initialize(tensor_val);
    args.push_back(tensor_val);

    auto baseline_results = execute(baseline_f, args, "INTERPRETER");
    auto optimized_results = execute(optimized_f, args, "INTERPRETER");

    EXPECT_TRUE(test::all_close(baseline_results.at(0), optimized_results.at(0)));
}

TEST(reshape_elimination, recurrent_reshapes_fan_out)
{
    Shape shape_a{128, 2048, 1, 1};
    auto generate_func = [shape_a]() {
        auto A = make_shared<op::Parameter>(element::f32, shape_a);

        auto reshape_1 = make_shared<op::Reshape>(A, AxisVector{0, 1, 2, 3}, shape_a);
        auto reshape_2 = make_shared<op::Reshape>(reshape_1, AxisVector{0, 1, 2, 3}, shape_a);
        auto reshape_3 = make_shared<op::Reshape>(reshape_2, AxisVector{0, 1, 2, 3}, shape_a);
        auto f_ = make_shared<Function>(NodeVector{reshape_2, reshape_3}, ParameterVector{A});
        return f_;
    };

    auto baseline_f = generate_func();
    auto optimized_f = generate_func();
    auto baseline_input_shape = baseline_f->get_parameters().at(0)->get_shape();

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::VisualizeTree>("before_recurrent_reshapes_fan_out.pdf");
    pass_manager.register_pass<pass::RecurrentReshapeElimination>();
    //pass_manager.register_pass<pass::ReshapeElimination>();
    pass_manager.register_pass<pass::VisualizeTree>("after_recurrent_reshapes_fan_out.pdf");
    pass_manager.run_passes(optimized_f);

    test::Uniform<float> rng(0.0f, 100.0f);
    vector<vector<float>> args;
    vector<float> tensor_val(shape_size(baseline_input_shape));
    rng.initialize(tensor_val);
    args.push_back(tensor_val);

    auto baseline_results = execute(baseline_f, args, "INTERPRETER");
    auto optimized_results = execute(optimized_f, args, "INTERPRETER");

    EXPECT_TRUE(test::all_close(baseline_results.at(0), optimized_results.at(0)));
}
