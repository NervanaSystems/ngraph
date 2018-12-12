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

#include "ngraph/pass/any_all_insertion.hpp"
#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/pass/manager.hpp"
#include "util/test_tools.hpp"

using namespace ngraph;
using namespace std;

// Ripped off of pass/any_all_replacement.cpp.
static std::shared_ptr<op::Reduce> make_any(std::shared_ptr<Node> arg,
                                            const AxisSet& reduction_axes)
{
    auto f_arg0 = std::make_shared<op::Parameter>(element::boolean, Shape{});
    auto f_arg1 = std::make_shared<op::Parameter>(element::boolean, Shape{});
    auto f_or = std::make_shared<op::Or>(f_arg0, f_arg1);
    auto f = std::make_shared<Function>(f_or, ParameterVector{f_arg0, f_arg1});

    auto k_false = op::Constant::create(element::boolean, Shape{}, std::vector<char>{0});

    return std::make_shared<op::Reduce>(arg, k_false, f, reduction_axes);
}

// Ripped off of pass/any_all_replacement.cpp.
static std::shared_ptr<op::Reduce> make_all(std::shared_ptr<Node> arg,
                                            const AxisSet& reduction_axes)
{
    auto f_arg0 = std::make_shared<op::Parameter>(element::boolean, Shape{});
    auto f_arg1 = std::make_shared<op::Parameter>(element::boolean, Shape{});
    auto f_and = std::make_shared<op::And>(f_arg0, f_arg1);
    auto f = std::make_shared<Function>(f_and, ParameterVector{f_arg0, f_arg1});

    auto k_true = op::Constant::create(element::boolean, Shape{}, std::vector<char>{1});

    return std::make_shared<op::Reduce>(arg, k_true, f, reduction_axes);
}

static void
    check_any_replacement(std::shared_ptr<Node> n, std::shared_ptr<Node> arg, const AxisSet& axes)
{
    auto any = std::dynamic_pointer_cast<op::Any>(n);
    ASSERT_NE(any, nullptr);
    ASSERT_EQ(any->get_reduction_axes(), axes);
    ASSERT_EQ(any->get_argument(0), arg);
}

static void
    check_all_replacement(std::shared_ptr<Node> n, std::shared_ptr<Node> arg, const AxisSet& axes)
{
    auto all = std::dynamic_pointer_cast<op::All>(n);
    ASSERT_NE(all, nullptr);
    ASSERT_EQ(all->get_reduction_axes(), axes);
    ASSERT_EQ(all->get_argument(0), arg);
}

TEST(any_all_insertion, any_simple)
{
    auto param = make_shared<op::Parameter>(element::boolean, Shape{2, 3, 4});
    auto any = make_any(param, AxisSet{1});
    auto f = make_shared<Function>(any, ParameterVector{param});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::AnyAllInsertion>();
    pass_manager.run_passes(f);

    check_any_replacement(
        f->get_results().at(0)->get_argument(0), param, any->get_reduction_axes());
}

TEST(any_all_insertion, any_chained)
{
    auto param = make_shared<op::Parameter>(element::boolean, Shape{2, 3, 4});
    auto any_0 = make_any(param, AxisSet{1});
    auto any_1 = make_any(any_0, AxisSet{1});
    auto f = make_shared<Function>(
        ResultVector{make_shared<op::Result>(any_0), make_shared<op::Result>(any_1)},
        ParameterVector{param});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::AnyAllInsertion>();
    pass_manager.run_passes(f);

    check_any_replacement(
        f->get_results().at(0)->get_argument(0), param, any_0->get_reduction_axes());
    check_any_replacement(f->get_results().at(1)->get_argument(0),
                          f->get_results().at(0)->get_argument(0),
                          any_1->get_reduction_axes());
}

TEST(any_all_insertion, all_simple)
{
    auto param = make_shared<op::Parameter>(element::boolean, Shape{2, 3, 4});
    auto all = make_all(param, AxisSet{1});
    auto f = make_shared<Function>(all, ParameterVector{param});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::AnyAllInsertion>();
    pass_manager.run_passes(f);

    check_all_replacement(
        f->get_results().at(0)->get_argument(0), param, all->get_reduction_axes());
}

TEST(any_all_insertion, all_chained)
{
    auto param = make_shared<op::Parameter>(element::boolean, Shape{2, 3, 4});
    auto all_0 = make_all(param, AxisSet{1});
    auto all_1 = make_all(all_0, AxisSet{1});
    auto f = make_shared<Function>(
        ResultVector{make_shared<op::Result>(all_0), make_shared<op::Result>(all_1)},
        ParameterVector{param});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::AnyAllInsertion>();
    pass_manager.run_passes(f);

    check_all_replacement(
        f->get_results().at(0)->get_argument(0), param, all_0->get_reduction_axes());
    check_all_replacement(f->get_results().at(1)->get_argument(0),
                          f->get_results().at(0)->get_argument(0),
                          all_1->get_reduction_axes());
}
