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

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "ngraph/ngraph.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/shape_specialization.hpp"

using namespace ngraph;
using namespace std;

TEST(shape_specialization, as_constants_shape_of)
{
    auto param = make_shared<op::Parameter>(element::boolean, Shape{2, 4, 6, 8});
    auto shape_of = make_shared<op::ShapeOf>(param);

    vector<shared_ptr<op::Constant>> replacements;
    ASSERT_TRUE(shape_of->as_constants(&replacements));
    ASSERT_EQ(replacements.size(), 1);
    ASSERT_EQ(replacements[0]->get_shape(), Shape{4});
    ASSERT_EQ(replacements[0]->get_element_type(), element::i64);
    ASSERT_EQ(replacements[0]->get_vector<int64_t>(), (vector<int64_t>{2, 4, 6, 8}));
}

TEST(shape_specialization, specialization_pass_shape_of_transpose)
{
    auto param0 = make_shared<op::Parameter>(element::boolean, Shape{4, 6});
    auto param1 = make_shared<op::Parameter>(element::boolean, Shape{1, 0});
    auto shape_of = make_shared<op::ShapeOf>(param1);
    auto transpose = make_shared<op::Transpose>(param0, shape_of);
    auto f = make_shared<Function>(transpose, ParameterVector{param0, param1});

    pass::Manager manager;
    manager.register_pass<pass::ShapeSpecialization>();
    manager.run_passes(f);

    auto transpose_after =
        dynamic_pointer_cast<op::Transpose>(f->get_results().at(0)->get_argument(0));
    ASSERT_NE(transpose_after, nullptr);

    auto constant_after = dynamic_pointer_cast<op::Constant>(transpose_after->get_argument(1));
    ASSERT_NE(constant_after, nullptr);

    ASSERT_EQ(constant_after->get_shape(), Shape{2});
    ASSERT_EQ(constant_after->get_element_type(), element::i64);
    ASSERT_EQ(constant_after->get_vector<int64_t>(), (vector<int64_t>{1, 0}));
}
