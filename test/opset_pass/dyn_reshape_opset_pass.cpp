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

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "ngraph/ngraph.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/opset0_downgrade.hpp"
#include "ngraph/pass/opset1_upgrade.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(opset_transform, opset1_dyn_reshape_upgrade_pass)
{
    const auto arg = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3});
    const auto pattern = make_shared<op::Parameter>(element::i64, Shape{6});

    const auto dyn_reshape_v0 = make_shared<op::v0::DynReshape>(arg, pattern, true);
    const auto result = make_shared<op::Result>(dyn_reshape_v0);
    auto f = make_shared<Function>(ResultVector{result}, ParameterVector{arg, pattern});

    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<pass::Opset1Upgrade>();
    pass_manager.run_passes(f);

    const auto pass_replacement_node = f->get_result()->input_value(0).get_node_shared_ptr();
    EXPECT_TRUE(is_type<op::v1::Reshape>(pass_replacement_node));
}

TEST(opset_transform, opset1_reshape_downgrade_pass)
{
    const auto arg = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3});
    const auto pattern = make_shared<op::Parameter>(element::i64, Shape{6});

    const auto dyn_reshape_v0 = make_shared<op::v1::Reshape>(arg, pattern, true);
    const auto result = make_shared<op::Result>(dyn_reshape_v0);
    auto f = make_shared<Function>(ResultVector{result}, ParameterVector{arg, pattern});

    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<pass::Opset0Downgrade>();
    pass_manager.run_passes(f);

    const auto pass_replacement_node = f->get_result()->input_value(0).get_node_shared_ptr();
    const auto reshape_v1 = as_type_ptr<op::v0::DynReshape>(pass_replacement_node);
    ASSERT_TRUE(reshape_v1);
    EXPECT_EQ(reshape_v1->get_zero_flag(), true);
}
