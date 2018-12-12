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

#include "ngraph/pass/any_all_replacement.hpp"
#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/pass/manager.hpp"
#include "util/test_tools.hpp"

using namespace ngraph;
using namespace std;

template <typename T>
static void
    check_replacement(std::shared_ptr<Node> n, std::shared_ptr<Node> arg, const AxisSet& axes)
{
    // NB: We are not checking all properties we could check here. (In particular could check the
    // constant value.)

    auto reduce = std::dynamic_pointer_cast<op::Reduce>(n);
    ASSERT_NE(reduce, nullptr);
    ASSERT_EQ(reduce->get_reduction_axes(), axes);
    ASSERT_EQ(reduce->get_argument(0), arg);
    auto k = std::dynamic_pointer_cast<op::Constant>(reduce->get_argument(1));
    ASSERT_NE(k, nullptr);
    auto reduce_f = reduce->get_functions().at(0);
    auto reduce_f_op = std::dynamic_pointer_cast<T>(reduce_f->get_results().at(0)->get_argument(0));
    ASSERT_NE(reduce_f_op, nullptr);
    ASSERT_EQ(reduce_f_op->get_argument(0), reduce_f->get_parameters().at(0));
    ASSERT_EQ(reduce_f_op->get_argument(1), reduce_f->get_parameters().at(1));
    ASSERT_EQ(reduce_f->get_parameters().at(0)->get_shape(), Shape{});
    ASSERT_EQ(reduce_f->get_parameters().at(1)->get_shape(), Shape{});
}

TEST(any_all_replacement, any_simple)
{
    auto param = make_shared<op::Parameter>(element::boolean, Shape{2, 3, 4});
    auto any = make_shared<op::Any>(param, AxisSet{1});
    auto f = make_shared<Function>(any, ParameterVector{param});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::AnyAllReplacement>();
    pass_manager.run_passes(f);

    check_replacement<op::Or>(
        f->get_results().at(0)->get_argument(0), param, any->get_reduction_axes());
}

TEST(any_all_replacement, any_chained)
{
    auto param = make_shared<op::Parameter>(element::boolean, Shape{2, 3, 4});
    auto any_0 = make_shared<op::Any>(param, AxisSet{1});
    auto any_1 = make_shared<op::Any>(any_0, AxisSet{1});
    auto f = make_shared<Function>(
        ResultVector{make_shared<op::Result>(any_0), make_shared<op::Result>(any_1)},
        ParameterVector{param});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::AnyAllReplacement>();
    pass_manager.run_passes(f);

    check_replacement<op::Or>(
        f->get_results().at(0)->get_argument(0), param, any_0->get_reduction_axes());
    check_replacement<op::Or>(f->get_results().at(1)->get_argument(0),
                              f->get_results().at(0)->get_argument(0),
                              any_1->get_reduction_axes());
}

TEST(any_all_replacement, all_simple)
{
    auto param = make_shared<op::Parameter>(element::boolean, Shape{2, 3, 4});
    auto all = make_shared<op::All>(param, AxisSet{1});
    auto f = make_shared<Function>(all, ParameterVector{param});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::AnyAllReplacement>();
    pass_manager.run_passes(f);

    check_replacement<op::And>(
        f->get_results().at(0)->get_argument(0), param, all->get_reduction_axes());
}

TEST(any_all_replacement, all_chained)
{
    auto param = make_shared<op::Parameter>(element::boolean, Shape{2, 3, 4});
    auto all_0 = make_shared<op::All>(param, AxisSet{1});
    auto all_1 = make_shared<op::All>(all_0, AxisSet{1});
    auto f = make_shared<Function>(
        ResultVector{make_shared<op::Result>(all_0), make_shared<op::Result>(all_1)},
        ParameterVector{param});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::AnyAllReplacement>();
    pass_manager.run_passes(f);

    check_replacement<op::And>(
        f->get_results().at(0)->get_argument(0), param, all_0->get_reduction_axes());
    check_replacement<op::And>(f->get_results().at(1)->get_argument(0),
                               f->get_results().at(0)->get_argument(0),
                               all_1->get_reduction_axes());
}
