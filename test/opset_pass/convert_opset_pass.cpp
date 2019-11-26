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
#include "ngraph/pass/opset0_downgrade.hpp"
#include "ngraph/pass/opset1_upgrade.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(opset_transform, convert_to_convertlike_upgrade)
{
    const auto data_shape = Shape{1, 2, 3, 4};
    const auto data = make_shared<op::Parameter>(element::f32, data_shape);
    const auto convert_v0 = make_shared<op::Convert>(data, element::i32);
    const auto result = make_shared<op::Result>(convert_v0);
    const auto f = make_shared<Function>(ResultVector{result}, ParameterVector{data});

    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<pass::Opset1Upgrade>();
    pass_manager.run_passes(f);

    const auto convert_v1 = f->get_results().at(0);
    const auto node = convert_v1->input(0).get_source_output().get_node_shared_ptr();
    auto convert_like_node = as_type_ptr<op::v1::ConvertLike>(node);

    EXPECT_EQ(convert_like_node->input(1).get_element_type(), element::i32);
    EXPECT_TRUE(
        convert_like_node->get_output_partial_shape(0).same_scheme(PartialShape{data_shape}));
}

TEST(opset_transform, convertlike_to_convert_downgrade)
{
    const auto data_shape = Shape{1, 2, 3};
    const auto data = make_shared<op::Parameter>(element::f32, data_shape);
    const auto target_type = op::Constant::create(element::i32, Shape{}, {0});
    const auto convert_v1 = make_shared<op::v1::ConvertLike>(data, target_type);
    const auto result = make_shared<op::Result>(convert_v1);
    const auto f = make_shared<Function>(ResultVector{result}, ParameterVector{data});

    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<pass::Opset0Downgrade>();
    pass_manager.run_passes(f);

    const auto convert_v0 = f->get_results().at(0);
    const auto node = convert_v0->input(0).get_source_output().get_node_shared_ptr();
    auto convert_node = as_type_ptr<op::Convert>(node);

    EXPECT_EQ(convert_node->get_destination_type(), element::i32);
    EXPECT_TRUE(convert_node->get_output_partial_shape(0).same_scheme(PartialShape{data_shape}));
}
