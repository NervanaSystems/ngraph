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

#include "ngraph/pass/dyn_elimination.hpp"
#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/pass/manager.hpp"
#include "util/all_close_f.hpp"
#include "util/test_tools.hpp"

using namespace ngraph;
using namespace std;

TEST(dyn_elimination, transpose)
{
    Shape shape_in{2, 4, 6, 8};
    auto param = make_shared<op::Parameter>(element::boolean, shape_in);

    auto constant_perm =
        make_shared<op::Constant>(element::i64, Shape{4}, vector<int64_t>{2, 3, 1, 0});

    auto transpose = make_shared<op::Transpose>(param, constant_perm);

    auto f = make_shared<Function>(transpose, ParameterVector{param});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::DynElimination>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::Transpose>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Reshape>(f), 1);

    auto new_reshape =
        std::dynamic_pointer_cast<op::Reshape>(f->get_results().at(0)->get_argument(0));
    ASSERT_TRUE(new_reshape);

    ASSERT_EQ(new_reshape->get_input_order(), (AxisVector{2, 3, 1, 0}));
    ASSERT_EQ(new_reshape->output(0).get_shape(), (Shape{6, 8, 4, 2}));
    ASSERT_EQ(new_reshape->get_output_element_type(0), element::boolean);
}

// For now, we can't handle the case where the input has dynamic shapes,
// because the classic Reshape op demands a Shape. Probably won't be able to
// deal with this until/unless we make a "StaticTranspose". Just make sure
// we don't crash or mangle the graph.
TEST(dyn_elimination, transpose_dyn_shape)
{
    PartialShape shape_in{2, 4, Dimension::dynamic(), 8};

    auto param = make_shared<op::Parameter>(element::boolean, shape_in);

    auto constant_perm =
        make_shared<op::Constant>(element::i64, Shape{4}, vector<int64_t>{2, 3, 1, 0});

    auto transpose = make_shared<op::Transpose>(param, constant_perm);

    auto f = make_shared<Function>(transpose, ParameterVector{param});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::DynElimination>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::Transpose>(f), 1);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_transpose =
        std::dynamic_pointer_cast<op::Transpose>(f->get_results().at(0)->get_argument(0));
    ASSERT_TRUE(new_transpose);

    ASSERT_EQ(new_transpose->get_output_element_type(0), element::boolean);
    ASSERT_TRUE(new_transpose->get_output_partial_shape(0).relaxes(
        PartialShape{Dimension::dynamic(), 8, 4, 2}));
}
