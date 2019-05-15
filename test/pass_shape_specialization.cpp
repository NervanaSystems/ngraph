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

    vector<shared_ptr<op::Constant>> replacements = shape_of->as_constants();
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

TEST(shape_specialization, as_constants_concat)
{
    auto k0 = op::Constant::create(element::i64, Shape{4}, {1, 2, 3, 4});
    auto k1 = op::Constant::create(element::i64, Shape{3}, {2, 5, 1});
    auto k2 = op::Constant::create(element::i64, Shape{0}, std::vector<int64_t>{});

    auto concat = make_shared<op::Concat>(NodeVector{k0, k1, k2}, 0);

    vector<shared_ptr<op::Constant>> replacements = concat->as_constants();
    ASSERT_EQ(replacements.size(), 1);
    ASSERT_EQ(replacements[0]->get_shape(), Shape{7});
    ASSERT_EQ(replacements[0]->get_element_type(), element::i64);
    ASSERT_EQ(replacements[0]->get_vector<int64_t>(), (vector<int64_t>{1, 2, 3, 4, 2, 5, 1}));
}

TEST(shape_specialization, as_constants_concat_noni64)
{
    auto k0 = op::Constant::create(element::i32, Shape{4}, {1, 2, 3, 4});
    auto k1 = op::Constant::create(element::i32, Shape{3}, {2, 5, 1});
    auto k2 = op::Constant::create(element::i32, Shape{0}, std::vector<int32_t>{});

    auto concat = make_shared<op::Concat>(NodeVector{k0, k1, k2}, 0);

    vector<shared_ptr<op::Constant>> replacements = concat->as_constants();
    ASSERT_EQ(replacements.size(), 0);
}

TEST(shape_specialization, as_constants_concat_nonvec_dim0)
{
    auto k0 = op::Constant::create(element::i64, Shape{2, 2}, {1, 2, 3, 4});
    auto k1 = op::Constant::create(element::i64, Shape{1, 2}, {2, 5});
    auto k2 = op::Constant::create(element::i64, Shape{0, 2}, std::vector<int64_t>{});

    auto concat = make_shared<op::Concat>(NodeVector{k0, k1, k2}, 0);

    vector<shared_ptr<op::Constant>> replacements = concat->as_constants();
    ASSERT_EQ(replacements.size(), 0);
}

TEST(shape_specialization, as_constants_concat_nonvec_dim1)
{
    auto k0 = op::Constant::create(element::i64, Shape{2, 2}, {1, 2, 3, 4});
    auto k1 = op::Constant::create(element::i64, Shape{2, 1}, {2, 5});
    auto k2 = op::Constant::create(element::i64, Shape{2, 0}, std::vector<int64_t>{});

    auto concat = make_shared<op::Concat>(NodeVector{k0, k1, k2}, 1);

    vector<shared_ptr<op::Constant>> replacements = concat->as_constants();
    ASSERT_EQ(replacements.size(), 0);
}

TEST(shape_specialization, as_constants_concat_nonconst)
{
    auto k0 = op::Constant::create(element::i64, Shape{2, 2}, {1, 2, 3, 4});
    auto k1 = op::Constant::create(element::i64, Shape{2, 2}, {2, 5, 2, 5});
    auto add = k0 + k1;

    auto concat = make_shared<op::Concat>(NodeVector{k0, k1, add}, 0);

    vector<shared_ptr<op::Constant>> replacements = concat->as_constants();
    ASSERT_EQ(replacements.size(), 0);
}

TEST(shape_specialization, specialization_pass_concat_transpose)
{
    auto param0 = make_shared<op::Parameter>(element::boolean, Shape{4, 6});
    auto k0 = op::Constant::create(element::i64, Shape{1}, {0});
    auto k1 = op::Constant::create(element::i64, Shape{1}, {1});

    auto concat = make_shared<op::Concat>(NodeVector{k1, k0}, 0);

    auto transpose = make_shared<op::Transpose>(param0, concat);
    auto f = make_shared<Function>(transpose, ParameterVector{param0});

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

// Slight variation on the above test, where the "Concat" does not already have constants going
// into it. (The permutation is Concat(Const<1>,Concat(Const<>,Const<0>)) rather than simply
// Concat(Const<1>,Const<0>).)
TEST(shape_specialization, specialization_pass_add_concat_transpose)
{
    auto param0 = make_shared<op::Parameter>(element::boolean, Shape{4, 6});
    auto k0 = op::Constant::create(element::i64, Shape{1}, {0});
    auto k1 = op::Constant::create(element::i64, Shape{1}, {1});
    auto kempty = op::Constant::create(element::i64, Shape{0}, vector<int64_t>{});

    auto concat = make_shared<op::Concat>(
        NodeVector{k1, make_shared<op::Concat>(NodeVector{kempty, k0}, 0)}, 0);

    auto transpose = make_shared<op::Transpose>(param0, concat);
    auto f = make_shared<Function>(transpose, ParameterVector{param0});

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

TEST(shape_specialization, pass_property)
{
    auto pass = std::make_shared<ngraph::pass::ShapeSpecialization>();
    ASSERT_EQ(false, pass->get_property(pass::PassProperty::REQUIRE_STATIC_SHAPE));
    ASSERT_EQ(true, pass->get_property(pass::PassProperty::CHANGE_DYNAMIC_STATE));
}
