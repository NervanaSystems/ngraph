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

TEST(opset_transform, opset1_topk_upgrade_pass)
{
    const size_t axis = 2;
    const size_t k = 10;
    const auto data = make_shared<op::Parameter>(element::i32, Shape{5, 10, 15});
    const auto topk_v0 = make_shared<op::TopK>(data, axis, element::i32, k);
    const auto result = make_shared<op::Result>(topk_v0);
    auto f = make_shared<Function>(ResultVector{result}, ParameterVector{data});

    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<pass::Opset1Upgrade>();
    pass_manager.run_passes(f);

    const auto pass_replacement_node =
        f->get_result()->input(0).get_source_output().get_node_shared_ptr();
    const auto topk_v1 = static_pointer_cast<op::v1::TopK>(pass_replacement_node);

    EXPECT_EQ(topk_v1->get_axis(), axis);
    EXPECT_EQ(topk_v1->description(), "TopK");
    EXPECT_EQ(topk_v1->get_version(), 1);
    EXPECT_EQ(topk_v1->get_mode(), op::v1::TopK::Mode::MAX);
    EXPECT_EQ(topk_v1->get_sort_type(), op::v1::TopK::SortType::SORT_VALUES);

    const auto values_out_element_type = topk_v1->output(0).get_element_type();
    EXPECT_EQ(values_out_element_type, data->get_element_type());
}
