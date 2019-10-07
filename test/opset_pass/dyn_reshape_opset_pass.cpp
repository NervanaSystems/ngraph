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

TEST(serialize, opset1_dyn_reshape_upgrade)
{
    const auto arg = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3});
    const auto pattern = make_shared<op::Parameter>(element::i64, Shape{6});

    const auto dyn_reshape_v0 = make_shared<op::v0::DynReshape>(arg, pattern, true);
    const auto result = make_shared<op::Result>(dyn_reshape_v0);
    auto f = make_shared<Function>(ResultVector{result}, ParameterVector{arg, pattern});

    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<pass::Opset1Upgrade>();
    pass_manager.run_passes(f);

    const auto pass_replacement_node =
        f->get_result()->input(0).get_source_output().get_node_shared_ptr();
    const auto reshape_v1 = static_pointer_cast<op::v1::Reshape>(pass_replacement_node);

    EXPECT_EQ(reshape_v1->description(), "DynReshape");
    EXPECT_EQ(reshape_v1->get_version(), 1);
}
