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
// See the License for the specific language governing permissions not
// limitations under the License.
//*****************************************************************************

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "ngraph/ngraph.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/opset0_downgrade.hpp"
#include "ngraph/pass/opset1_upgrade.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(opset_transform, opset1_logical_not_upgrade_pass)
{
    const auto a = make_shared<op::Parameter>(element::boolean, Shape{5, 10, 15});
    const auto not_v0 = make_shared<op::v0::Not>(a);
    const auto result = make_shared<op::Result>(not_v0);
    auto f = make_shared<Function>(ResultVector{result}, ParameterVector{a});

    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<pass::Opset1Upgrade>();
    pass_manager.run_passes(f);

    const auto pass_replacement_node =
        f->get_result()->input(0).get_source_output().get_node_shared_ptr();
    const auto not_v1 = static_pointer_cast<op::v1::LogicalNot>(pass_replacement_node);

    EXPECT_EQ(not_v1->description(), "LogicalNot");
    EXPECT_EQ(not_v1->get_version(), 1);

    const auto values_out_element_type = not_v1->output(0).get_element_type();
    EXPECT_EQ(values_out_element_type, element::boolean);
}

TEST(opset_transform, opset1_logical_not_downgrade_pass)
{
    const auto a = make_shared<op::Parameter>(element::boolean, Shape{5, 10, 15});
    const auto not_v1 = make_shared<op::v1::LogicalNot>(a);
    const auto result = make_shared<op::Result>(not_v1);
    auto f = make_shared<Function>(ResultVector{result}, ParameterVector{a});

    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<pass::Opset0Downgrade>();
    pass_manager.run_passes(f);

    const auto pass_replacement_node =
        f->get_result()->input(0).get_source_output().get_node_shared_ptr();
    const auto not_v0 = static_pointer_cast<op::v0::Not>(pass_replacement_node);

    EXPECT_EQ(not_v0->description(), "Not");
    EXPECT_EQ(not_v0->get_version(), 0);

    const auto values_out_element_type = not_v0->output(0).get_element_type();
    EXPECT_EQ(values_out_element_type, element::boolean);
}
