//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
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
#include "ngraph/autodiff/adjoints.hpp"
#include "ngraph/file_util.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/batch_norm.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/pass/algebraic_simplification.hpp"
#include "ngraph/pass/core_fusion.hpp"
#include "ngraph/pass/cse.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/reshape_elimination.hpp"
#include "ngraph/pass/visualize_tree.hpp"
#include "ngraph/runtime/cpu/pass/cpu_fusion.hpp"
#include "ngraph/runtime/cpu/pass/cpu_reshape_sinking.hpp"
#include "ngraph/serializer.hpp"
#include "ngraph/util.hpp"
#include "nlohmann/json.hpp"
#include "util/all_close.hpp"
#include "util/autodiff/backprop_function.hpp"
#include "util/autodiff/numeric_compare.hpp"
#include "util/ndarray.hpp"
#include "util/random.hpp"
#include "util/test_tools.hpp"

using namespace ngraph;
using namespace std;

TEST(cpu_reshape_sinking, edge_splitting)
{
    //checks if Reshapes are pushed through op::Abs, but stopped by Sum
    Shape shape_nhwc{16, 28, 28, 1};
    Shape shape_nchw{16, 1, 28, 28};
    auto a = make_shared<op::Parameter>(element::i32, shape_nhwc);
    auto reshape = make_shared<op::Reshape>(a, AxisVector{0, 3, 1, 2}, shape_nchw);
    auto absn = make_shared<op::Abs>(reshape);
    auto absn2 = make_shared<op::Abs>(absn);
    auto sum = make_shared<op::Sum>(reshape, AxisSet{0, 1, 2, 3});
    auto func = make_shared<Function>(NodeVector{absn2, sum}, op::ParameterVector{a});
    pass::Manager pass_manager;
    //size_t before_count = count_ops_of_type<op::Reshape>(func);
    pass_manager.register_pass<pass::VisualizeTree>("before.pdf");
    pass_manager.register_pass<runtime::cpu::pass::CPUReshapeSinking>();
    pass_manager.register_pass<pass::ReshapeElimination>();
    pass_manager.register_pass<pass::CommonSubexpressionElimination>();
    pass_manager.register_pass<pass::VisualizeTree>("after.pdf");
    pass_manager.run_passes(func);
    ASSERT_EQ(func->get_results().at(1)->get_argument(0), sum);
    auto new_reshape =
        std::dynamic_pointer_cast<op::Reshape>(func->get_results().at(0)->get_argument(0));
    ASSERT_TRUE(new_reshape);
    ASSERT_EQ(new_reshape->get_shape(), shape_nchw);
}

TEST(cpu_reshape_sinking, mnist_conv)
{
    //const string json_path = file_util::path_join(SERIALIZED_ZOO, "tf_conv_mnist_nhwc.json");
    //const string json_path = file_util::path_join(SERIALIZED_ZOO, "tf_function_ngraph_cluster_39.json");
    const string json_path = file_util::path_join(SERIALIZED_ZOO, "dcgan_tf_function_ngraph_cluster_withbatchnorm.json");
    const string json_string = file_util::read_file_to_string(json_path);
    stringstream ss(json_string);
    shared_ptr<Function> func = ngraph::deserialize(ss);
    pass::Manager pass_manager;
    size_t before_count = count_ops_of_type<op::Reshape>(func);
    pass_manager.register_pass<pass::VisualizeTree>("before.pdf");
    pass_manager.register_pass<pass::CoreFusion>();
    pass_manager.register_pass<runtime::cpu::pass::CPUReshapeSinking>();
    pass_manager.register_pass<pass::ReshapeElimination>();
    pass_manager.register_pass<pass::CommonSubexpressionElimination>();
    pass_manager.register_pass<pass::AlgebraicSimplification>();
    //pass_manager.register_pass<runtime::cpu::pass::CPUFusion>();
    pass_manager.register_pass<pass::VisualizeTree>("after.pdf");
    pass_manager.run_passes(func);
    size_t before_after = count_ops_of_type<op::Reshape>(func);
    std::cout <<"before: " << before_count << ", after: " << before_after << "\n";
    ASSERT_LE(before_after, before_count);
}
