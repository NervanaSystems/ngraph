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

#include "ngraph/pass/concat_fusion.hpp"
#include "ngraph/pass/visualize_tree.hpp"
#include "ngraph/file_util.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/pass/graph_rewrite.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pattern/matcher.hpp"
#include "ngraph/pattern/op/label.hpp"
#include "ngraph/pattern/op/skip.hpp"
#include "ngraph/serializer.hpp"
#include "ngraph/util.hpp"
#include "util/all_close.hpp"
#include "util/autodiff/backprop_function.hpp"
#include "util/matcher.hpp"
#include "util/random.hpp"
#include "util/test_tools.hpp"

using namespace ngraph;
using namespace std;

TEST(concat_fusion, multiple_branches)
{
    Shape shape_a{128, 2048, 1, 1};
    auto generate_func = [shape_a]() {
        auto A = make_shared<op::Parameter>(element::f32, shape_a);
        // Function 0
        auto concat_1 = make_shared<op::Concat>(NodeVector{A}, 2);
        auto concat_2 = make_shared<op::Concat>(NodeVector{concat_1}, 2);
        auto concat_3 = make_shared<op::Concat>(
            NodeVector{concat_2, concat_2, concat_2, concat_2, concat_2, concat_2, concat_2}, 2);
        auto concat_4 = make_shared<op::Concat>(
            NodeVector{concat_3, concat_3, concat_3, concat_3, concat_3, concat_3, concat_3}, 3);

        auto concat_5 = make_shared<op::Concat>(NodeVector{A, A}, 2);
        auto concat_6 = make_shared<op::Concat>(NodeVector{concat_5, concat_5, concat_5}, 3);
        auto f_concat_1 = make_shared<Function>(NodeVector{concat_4, concat_6}, ParameterVector{A});
        return f_concat_1;
    };
 
    auto baseline_f = generate_func();
    auto optimized_f = generate_func();
    auto baseline_input_shape = baseline_f->get_parameters().at(0)->get_shape();

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::VisualizeTree>("before.pdf");
    pass_manager.register_pass<pass::ConcatElimination>();
    pass_manager.register_pass<pass::SelfConcatFusion>();
    pass_manager.register_pass<pass::VisualizeTree>("after.pdf");
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
