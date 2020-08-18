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

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <list>
#include <memory>

#include "gtest/gtest.h"
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/pass/graph_rewrite.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/reshape_elimination_v1.hpp"
#include "ngraph/pass/visualize_tree.hpp"
#include "ngraph/pattern/matcher.hpp"
#include "ngraph/pattern/op/label.hpp"
#include "ngraph/util.hpp"
#include "util/all_close.hpp"
#include "util/matcher.hpp"
#include "util/random.hpp"
#include "util/test_tools.hpp"

using namespace ngraph;
using namespace std;

TEST(reshape_elimination_v1, reshape_elimination_v1)
{
    auto generate_func = [](bool zero) {
        auto arg = std::make_shared<op::v0::Parameter>(element::i64, PartialShape{8, 16, 2, 3});
        auto pattern =
            op::v0::Constant::create(element::i64, Shape{4}, vector<int64_t>{8, 16, 2, 3});
        auto reshape_v1 = std::make_shared<op::v1::Reshape>(arg, pattern, zero);
        auto abs = std::make_shared<op::v0::Abs>(reshape_v1);
        return std::make_shared<Function>(OutputVector{abs}, ParameterVector{arg});
    };

    auto func = generate_func(false);
    auto nopass_func = generate_func(false);
    auto func_zero = generate_func(true);
    auto nopass_func_zero = generate_func(true);

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ReshapeEliminationV1>();
    pass_manager.run_passes(func);
    pass_manager.run_passes(func_zero);
    ASSERT_TRUE(count_ops_of_type<op::v1::Reshape>(nopass_func) == 1);
    ASSERT_TRUE(count_ops_of_type<op::v1::Reshape>(func) == 0);
    ASSERT_TRUE(count_ops_of_type<op::v1::Reshape>(nopass_func_zero) == 1);
    ASSERT_TRUE(count_ops_of_type<op::v1::Reshape>(func_zero) == 0);
}

TEST(reshape_elimination_v1, reshape_elimination_v1_dynamic)
{
    auto arg = std::make_shared<op::v0::Parameter>(element::i64, PartialShape::dynamic());
    auto pattern = make_shared<op::v0::Parameter>(element::i64, PartialShape::dynamic(1));
    auto reshape_v1 = std::make_shared<op::v1::Reshape>(arg, pattern, false);
    auto abs = std::make_shared<op::v0::Abs>(reshape_v1);
    auto f = std::make_shared<Function>(OutputVector{abs}, ParameterVector{arg, pattern});
    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ReshapeEliminationV1>();
    pass_manager.run_passes(f);
    ASSERT_TRUE(count_ops_of_type<op::v1::Reshape>(f) == 1);
}
