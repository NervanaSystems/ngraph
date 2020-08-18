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
#include "ngraph/autodiff/adjoints.hpp"
#include "ngraph/file_util.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/batch_norm.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/visualize_tree.hpp"
#include "ngraph/pattern/matcher.hpp"
#include "ngraph/serializer.hpp"
#include "ngraph/util.hpp"
#include "util/all_close.hpp"
#include "util/autodiff/backprop_function.hpp"
#include "util/autodiff/numeric_compare.hpp"
#include "util/ndarray.hpp"
#include "util/random.hpp"
#include "util/test_tools.hpp"

using namespace ngraph;
using namespace std;

class ControlDependencyOp : public ngraph::op::Op
{
public:
    static constexpr NodeTypeInfo type_info{"ControlDependencyOp", 0};
    const NodeTypeInfo& get_type_info() const override { return type_info; }
    virtual std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override
    {
        auto clone = make_shared<ControlDependencyOp>(new_args, std::set<std::shared_ptr<Node>>{});
        return move(clone);
    }

    ControlDependencyOp(const OutputVector& args, const std::set<std::shared_ptr<Node>>& deps)
        : Op(args)
    {
        if (args.size() == 0 && deps.size() == 0)
        {
            throw ngraph_error("Expected some arguments or dependencies");
        }

        for (auto& node : deps)
        {
            add_control_dependency(node);
        }

        if (args.size() != 0)
        {
            set_output_type(0, args.at(0).get_element_type(), args.at(0).get_shape());
        }
        else
        {
            auto dn = *(deps.begin());
            set_output_type(0, dn->get_output_element_type(0), dn->get_output_shape(0));
        }
    }
};
constexpr NodeTypeInfo ControlDependencyOp::type_info;

TEST(control_dependencies, cdep_ops)
{
    auto A = make_shared<op::v0::Parameter>(element::f32, Shape{});
    auto B = make_shared<op::v0::Parameter>(element::f32, Shape{});
    auto absn = make_shared<op::v0::Abs>(A);
    auto cdop =
        make_shared<ControlDependencyOp>(OutputVector{A}, std::set<std::shared_ptr<Node>>{absn});

    auto f = make_shared<Function>(cdop, ParameterVector{A, B});
    test_ordered_ops(f, NodeVector{absn});
}

TEST(control_dependencies, two_cdep_ops)
{
    auto A = make_shared<op::v0::Parameter>(element::f32, Shape{});
    auto B = make_shared<op::v0::Parameter>(element::f32, Shape{});
    auto absn = make_shared<op::v0::Abs>(A);
    auto C = make_shared<op::v0::Parameter>(element::f32, Shape{});
    auto absn_c = make_shared<op::v0::Abs>(C);
    auto cdop = make_shared<ControlDependencyOp>(OutputVector{A},
                                                 std::set<std::shared_ptr<Node>>{absn, absn_c});

    auto f = make_shared<Function>(cdop, ParameterVector{A, B, C});
    test_ordered_ops(f, NodeVector{absn, absn_c});
}

TEST(control_dependencies, two_cdep_ops_op_on_top)
{
    auto A = make_shared<op::v0::Parameter>(element::f32, Shape{});
    auto absn = make_shared<op::v0::Abs>(A);
    auto B = make_shared<op::v0::Parameter>(element::f32, Shape{});
    auto absn_b = make_shared<op::v0::Abs>(B);
    auto cdop = make_shared<ControlDependencyOp>(OutputVector{A},
                                                 std::set<std::shared_ptr<Node>>{absn, absn_b});
    auto absn_cdop = make_shared<op::v0::Abs>(cdop);

    auto f = make_shared<Function>(absn_cdop, ParameterVector{A, B});
    test_ordered_ops(f, NodeVector{absn, absn_b});
}

TEST(control_dependencies, clone_function_cdop)
{
    auto A = make_shared<op::v0::Parameter>(element::f32, Shape{});
    auto absn = make_shared<op::v0::Abs>(A);
    auto cdop =
        make_shared<ControlDependencyOp>(OutputVector{A}, std::set<std::shared_ptr<Node>>{absn});

    auto f = make_shared<Function>(cdop, ParameterVector{A});
    test_ordered_ops(f, NodeVector{absn});
    auto clone = ngraph::clone_function(*f.get());
    auto matcher = std::make_shared<pattern::Matcher>(cdop);
    auto cdop_clone = clone->get_results().at(0)->get_argument(0);
    ASSERT_TRUE(matcher->match(cdop_clone));
    auto cloned_deps = cdop_clone->get_control_dependencies();
    ASSERT_EQ(cloned_deps.size(), 1);
    auto cloned_abs = *begin(cloned_deps);
    ASSERT_TRUE(is_type<op::v0::Abs>(cloned_abs));
}

TEST(control_dependencies, clone_function_cdop_abs)
{
    auto A = make_shared<op::v0::Parameter>(element::f32, Shape{});
    auto absn = make_shared<op::v0::Abs>(A);
    auto B = make_shared<op::v0::Parameter>(element::f32, Shape{});
    auto absn_b = make_shared<op::v0::Abs>(B);
    auto cdop = make_shared<ControlDependencyOp>(OutputVector{A},
                                                 std::set<std::shared_ptr<Node>>{absn, absn_b});
    auto absn_cdop = make_shared<op::v0::Abs>(cdop);

    auto f = make_shared<Function>(absn_cdop, ParameterVector{A, B});
    auto clone = ngraph::clone_function(*f.get());
    auto matcher = std::make_shared<pattern::Matcher>(cdop);
    auto cdop_clone = clone->get_results().at(0)->get_argument(0)->get_argument(0);
    ASSERT_TRUE(matcher->match(cdop_clone));
    auto cloned_deps = cdop_clone->get_control_dependencies();
    ASSERT_EQ(cloned_deps.size(), 2);
    for (auto ccdep : cloned_deps)
    {
        ASSERT_TRUE(is_type<op::v0::Abs>(ccdep));
    }
}

static size_t count_control_dependencies(const shared_ptr<Node>& node,
                                         const shared_ptr<Node>& dependency)
{
    auto& dependencies = node->get_control_dependencies();
    return count(dependencies.begin(), dependencies.end(), dependency);
}

TEST(control_dependencies, replace_node)
{
    Shape shape{2, 2};
    auto A = make_shared<op::v0::Parameter>(element::f32, shape);
    auto B = make_shared<op::v0::Parameter>(element::f32, shape);
    auto MUL_AB = A * B;
    auto MUL_BA = B * A;
    auto ADD = A + B;
    auto SUM = MUL_AB + ADD;
    ADD->add_control_dependency(MUL_AB);
    ASSERT_TRUE(1 == count_control_dependencies(ADD, MUL_AB));
    ASSERT_TRUE(0 == count_control_dependencies(ADD, MUL_BA));
    replace_node(MUL_AB, MUL_BA);
    ASSERT_TRUE(0 == count_control_dependencies(ADD, MUL_AB));
    ASSERT_TRUE(1 == count_control_dependencies(ADD, MUL_BA));
}

#ifndef NGRAPH_JSON_DISABLE
TEST(control_dependencies, serialize_cdop)
{
    auto A = make_shared<op::v0::Parameter>(element::f32, Shape{});
    auto absn = make_shared<op::v0::Abs>(A);
    auto cdop = make_shared<op::v0::Negative>(A);
    cdop->add_control_dependency(absn);
    auto f = make_shared<Function>(cdop, ParameterVector{A});

    string js = serialize(f, 4);
    shared_ptr<Function> clone = deserialize(js);

    auto matcher = std::make_shared<pattern::Matcher>(cdop);
    auto cdop_clone = clone->get_results().at(0)->get_argument(0);
    ASSERT_TRUE(matcher->match(cdop_clone));
    auto cloned_deps = cdop_clone->get_control_dependencies();
    ASSERT_EQ(cloned_deps.size(), 1);
    auto cloned_abs = *begin(cloned_deps);
    ASSERT_TRUE(is_type<op::v0::Abs>(cloned_abs));
}

TEST(control_dependencies, serialize_cdop_abs)
{
    auto A = make_shared<op::v0::Parameter>(element::f32, Shape{});
    auto absn = make_shared<op::v0::Abs>(A);
    auto B = make_shared<op::v0::Parameter>(element::f32, Shape{});
    auto absn_b = make_shared<op::v0::Abs>(B);
    auto cdop = make_shared<op::v0::Negative>(A);
    cdop->add_control_dependency(absn);
    cdop->add_control_dependency(absn_b);
    auto absn_cdop = make_shared<op::v0::Abs>(cdop);

    auto f = make_shared<Function>(absn_cdop, ParameterVector{A, B});

    string js = serialize(f, 4);
    shared_ptr<Function> clone = deserialize(js);
    auto matcher = std::make_shared<pattern::Matcher>(cdop);
    auto cdop_clone = clone->get_results().at(0)->get_argument(0)->get_argument(0);
    ASSERT_TRUE(matcher->match(cdop_clone));
    auto cloned_deps = cdop_clone->get_control_dependencies();
    ASSERT_EQ(cloned_deps.size(), 2);
    for (auto ccdep : cloned_deps)
    {
        ASSERT_TRUE(is_type<op::v0::Abs>(ccdep));
    }
}
#endif
