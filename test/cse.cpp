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

#include <memory>

#include "gtest/gtest.h"
#include "ngraph/file_util.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/abs.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/product.hpp"
#include "ngraph/op/sqrt.hpp"
#include "ngraph/op/subtract.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/pass/cse.hpp"
#include "ngraph/pass/manager.hpp"
#include "util/test_tools.hpp"

using namespace ngraph;
using namespace std;

TEST(CSE, abs_abs)
{
    Shape zero_shape{0};
    auto A = std::make_shared<op::Parameter>(element::i32, zero_shape);
    auto abs1 = std::make_shared<op::Abs>(A);
    auto abs2 = std::make_shared<op::Abs>(A);
    auto f = std::make_shared<Function>(NodeVector{abs1, abs2}, op::ParameterVector{A});
    pass::Manager pass_manager;

    pass_manager.register_pass<ngraph::pass::CommonSubexpressionElimination>();
    pass_manager.run_passes(f);
    ASSERT_EQ(f->get_results().at(0)->get_argument(0), f->get_results().at(1)->get_argument(0));
}

TEST(CSE, abs_abs_negative)
{
    Shape zero_shape{0};
    auto A = std::make_shared<op::Parameter>(element::i32, zero_shape);
    auto B = std::make_shared<op::Parameter>(element::i32, zero_shape);
    auto abs1 = std::make_shared<op::Abs>(A);
    auto abs2 = std::make_shared<op::Abs>(B);
    auto f = std::make_shared<Function>(NodeVector{abs1, abs2}, op::ParameterVector{A, B});
    pass::Manager pass_manager;

    pass_manager.register_pass<ngraph::pass::CommonSubexpressionElimination>();
    pass_manager.run_passes(f);
    ASSERT_EQ(f->get_results().at(0)->get_argument(0), abs1);
    ASSERT_EQ(f->get_results().at(1)->get_argument(0), abs2);
}

TEST(CSE, add_add)
{
    Shape zero_shape{0};
    auto A = std::make_shared<op::Parameter>(element::i32, zero_shape);
    auto B = std::make_shared<op::Parameter>(element::i32, zero_shape);
    auto add1 = std::make_shared<op::Add>(A, B);
    auto add2 = std::make_shared<op::Add>(A, B);
    auto f = std::make_shared<Function>(NodeVector{add1, add2}, op::ParameterVector{A, B});
    pass::Manager pass_manager;

    pass_manager.register_pass<ngraph::pass::CommonSubexpressionElimination>();
    pass_manager.run_passes(f);
    ASSERT_EQ(f->get_results().at(0)->get_argument(0), f->get_results().at(1)->get_argument(0));
}

TEST(CSE, add_add_commutative)
{
    Shape zero_shape{0};
    auto A = std::make_shared<op::Parameter>(element::i32, zero_shape);
    auto B = std::make_shared<op::Parameter>(element::i32, zero_shape);
    auto add1 = std::make_shared<op::Add>(A, B);
    auto add2 = std::make_shared<op::Add>(B, A);
    auto f = std::make_shared<Function>(NodeVector{add1, add2}, op::ParameterVector{A, B});
    pass::Manager pass_manager;

    pass_manager.register_pass<ngraph::pass::CommonSubexpressionElimination>();
    pass_manager.run_passes(f);
    ASSERT_EQ(f->get_results().at(0)->get_argument(0), f->get_results().at(1)->get_argument(0));
}

TEST(CSE, add_add_negative)
{
    Shape zero_shape{0};
    auto A = std::make_shared<op::Parameter>(element::i32, zero_shape);
    auto B = std::make_shared<op::Parameter>(element::i32, zero_shape);
    auto C = std::make_shared<op::Parameter>(element::i32, zero_shape);
    auto D = std::make_shared<op::Parameter>(element::i32, zero_shape);
    auto add1 = std::make_shared<op::Add>(A, B);
    auto add2 = std::make_shared<op::Add>(C, D);
    auto f = std::make_shared<Function>(NodeVector{add1, add2}, op::ParameterVector{A, B, C, D});
    pass::Manager pass_manager;

    pass_manager.register_pass<ngraph::pass::CommonSubexpressionElimination>();
    pass_manager.run_passes(f);
    ASSERT_EQ(f->get_results().at(0)->get_argument(0), add1);
    ASSERT_EQ(f->get_results().at(1)->get_argument(0), add2);
}

TEST(CSE, abs_add)
{
    Shape zero_shape{0};
    auto A = std::make_shared<op::Parameter>(element::i32, zero_shape);
    auto B = std::make_shared<op::Parameter>(element::i32, zero_shape);
    auto abs_a1 = std::make_shared<op::Abs>(A);
    auto abs_b1 = std::make_shared<op::Abs>(B);
    auto abs_a2 = std::make_shared<op::Abs>(A);
    auto abs_b2 = std::make_shared<op::Abs>(B);
    auto add1 = std::make_shared<op::Add>(abs_a1, abs_b1);
    auto add2 = std::make_shared<op::Add>(abs_a2, abs_b2);
    auto f = std::make_shared<Function>(NodeVector{add1, add2}, op::ParameterVector{A, B});
    pass::Manager pass_manager;

    pass_manager.register_pass<ngraph::pass::CommonSubexpressionElimination>();
    pass_manager.run_passes(f);
    ASSERT_EQ(f->get_results().at(0)->get_argument(0), f->get_results().at(1)->get_argument(0));
}

TEST(CSE, abs_add_abs_add)
{
    Shape zero_shape{0};
    auto A = std::make_shared<op::Parameter>(element::i32, zero_shape);
    auto B = std::make_shared<op::Parameter>(element::i32, zero_shape);
    auto abs_a1 = std::make_shared<op::Abs>(A);
    auto abs_b1 = std::make_shared<op::Abs>(B);
    auto abs_a2 = std::make_shared<op::Abs>(A);
    auto abs_b2 = std::make_shared<op::Abs>(B);
    auto add1 = std::make_shared<op::Add>(abs_a1, abs_b1);
    auto add2 = std::make_shared<op::Add>(abs_a2, abs_b2);
    auto abs_add1 = std::make_shared<op::Abs>(add1);
    auto abs_add2 = std::make_shared<op::Abs>(add2);
    auto C = std::make_shared<op::Parameter>(element::i32, zero_shape);
    auto add3 = std::make_shared<op::Add>(abs_add1, C);
    auto add4 = std::make_shared<op::Add>(abs_add2, C);
    auto f = std::make_shared<Function>(NodeVector{add3, add4}, op::ParameterVector{A, B, C});
    pass::Manager pass_manager;

    pass_manager.register_pass<ngraph::pass::CommonSubexpressionElimination>();
    pass_manager.run_passes(f);
    ASSERT_EQ(f->get_results().at(0)->get_argument(0), f->get_results().at(1)->get_argument(0));
}

TEST(CSE, abs_add_abs_add_negative)
{
    Shape zero_shape{0};
    auto A = std::make_shared<op::Parameter>(element::i32, zero_shape);
    auto B = std::make_shared<op::Parameter>(element::i32, zero_shape);
    auto abs_a1 = std::make_shared<op::Abs>(A);
    auto abs_b1 = std::make_shared<op::Abs>(B);
    auto abs_a2 = std::make_shared<op::Abs>(A);
    auto abs_b2 = std::make_shared<op::Abs>(B);
    auto add1 = std::make_shared<op::Add>(abs_a1, abs_b1);
    auto add2 = std::make_shared<op::Add>(abs_a2, abs_b2);
    auto abs_add1 = std::make_shared<op::Abs>(add1);
    auto abs_add2 = std::make_shared<op::Abs>(add2);
    auto C = std::make_shared<op::Parameter>(element::i32, zero_shape);
    auto D = std::make_shared<op::Parameter>(element::i32, zero_shape);
    auto add3 = std::make_shared<op::Add>(abs_add1, C);
    auto add4 = std::make_shared<op::Add>(abs_add2, D);
    auto f = std::make_shared<Function>(NodeVector{add3, add4}, op::ParameterVector{A, B, C, D});
    pass::Manager pass_manager;

    pass_manager.register_pass<ngraph::pass::CommonSubexpressionElimination>();
    pass_manager.run_passes(f);
    auto oadd3 = f->get_results().at(0)->get_argument(0);
    auto oadd4 = f->get_results().at(1)->get_argument(0);
    ASSERT_EQ(oadd3, add3);
    ASSERT_EQ(oadd4, add4);
    ASSERT_EQ(oadd3->get_argument(1), C);
    ASSERT_EQ(oadd4->get_argument(1), D);
    ASSERT_EQ(oadd3->get_argument(0), oadd4->get_argument(0));
}
