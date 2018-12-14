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

#include "any_all_replacement.hpp"
#include "ngraph/op/all.hpp"
#include "ngraph/op/and.hpp"
#include "ngraph/op/any.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/or.hpp"
#include "ngraph/op/reduce.hpp"
#include "ngraph/util.hpp"

using namespace ngraph;

static std::shared_ptr<Node> make_any(std::shared_ptr<Node> arg, const AxisSet& reduction_axes)
{
    auto f_arg0 = std::make_shared<op::Parameter>(element::boolean, Shape{});
    auto f_arg1 = std::make_shared<op::Parameter>(element::boolean, Shape{});
    auto f_or = std::make_shared<op::Or>(f_arg0, f_arg1);
    auto f = std::make_shared<Function>(f_or, ParameterVector{f_arg0, f_arg1});

    auto k_false = op::Constant::create(element::boolean, Shape{}, std::vector<char>{0});

    return std::make_shared<op::Reduce>(arg, k_false, f, reduction_axes);
}

static std::shared_ptr<Node> make_all(std::shared_ptr<Node> arg, const AxisSet& reduction_axes)
{
    auto f_arg0 = std::make_shared<op::Parameter>(element::boolean, Shape{});
    auto f_arg1 = std::make_shared<op::Parameter>(element::boolean, Shape{});
    auto f_or = std::make_shared<op::And>(f_arg0, f_arg1);
    auto f = std::make_shared<Function>(f_or, ParameterVector{f_arg0, f_arg1});

    auto k_true = op::Constant::create(element::boolean, Shape{}, std::vector<char>{1});

    return std::make_shared<op::Reduce>(arg, k_true, f, reduction_axes);
}

bool ngraph::pass::AnyAllReplacement::run_on_node(std::shared_ptr<ngraph::Node> node)
{
    bool clobbered = false;

    if (auto any = std::dynamic_pointer_cast<ngraph::op::Any>(node))
    {
        ngraph::replace_node(any, make_any(any->get_argument(0), any->get_reduction_axes()));
        clobbered = true;
    }
    else if (auto all = std::dynamic_pointer_cast<ngraph::op::All>(node))
    {
        ngraph::replace_node(all, make_all(all->get_argument(0), all->get_reduction_axes()));
        clobbered = true;
    }

    return clobbered;
}
