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

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "ngraph/ngraph.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/opset1_upgrade.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(serialize, opset1_sum_upgrade)
{
    const auto data = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3});
    const AxisSet reduction_axes{1, 2};

    const auto sum_v0 = make_shared<op::Sum>(data, reduction_axes);
    const auto result = make_shared<op::Result>(sum_v0);
    auto f = make_shared<Function>(ResultVector{result}, ParameterVector{data});

    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<pass::Opset1Upgrade>();
    pass_manager.run_passes(f);

    const auto pass_replacement_node =
        f->get_result()->input(0).get_source_output().get_node_shared_ptr();
    const auto reduce_sum_v1 = static_pointer_cast<op::v1::ReduceProd>(pass_replacement_node);

    EXPECT_EQ(reduce_sum_v1->description(), "Sum");
    EXPECT_EQ(reduce_sum_v1->get_version(), 1);
    EXPECT_EQ(reduce_sum_v1->get_keep_dims(), false);
}
