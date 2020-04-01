//*****************************************************************************
// Copyright 2020 Intel Corporation
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

#include "ngraph/ngraph.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/transpose_elimination.hpp"
#include "util/test_tools.hpp"

using namespace ngraph;
using namespace std;

TEST(transpose_elimination, remove_transpose)
{
    // add a simple graph with transpose
    Shape shape_in{1, 2};
    auto param = make_shared<op::Parameter>(element::boolean, shape_in);
    auto constant_perm = make_shared<op::Constant>(element::i64, Shape{2}, vector<int64_t>{1, 0});
    auto transpose = make_shared<op::Transpose>(param, constant_perm);
    auto f = make_shared<Function>(transpose, ParameterVector{param});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::TransposeElimination>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::Transpose>(f), 0);
}
