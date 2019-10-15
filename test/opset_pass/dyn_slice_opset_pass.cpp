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

TEST(opset_transform, opset1_dyn_slice_upgrade_pass)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 4, 6, 8});
    auto lower_bounds = make_shared<op::Parameter>(element::i64, Shape{4});
    auto upper_bounds = make_shared<op::Parameter>(element::i64, Shape{4});
    auto strides = make_shared<op::Parameter>(element::i64, Shape{4});

    auto dyn_slice_v0 = make_shared<op::v0::DynSlice>(arg,
                                                      lower_bounds,
                                                      upper_bounds,
                                                      strides,
                                                      AxisSet{0, 2},
                                                      AxisSet{0, 1, 2},
                                                      AxisSet{2, 3},
                                                      AxisSet{},
                                                      AxisSet{0, 1, 2, 3});

    const auto result = make_shared<op::Result>(dyn_slice_v0);
    auto f = make_shared<Function>(ResultVector{result},
                                   ParameterVector{arg, lower_bounds, upper_bounds, strides});

    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<pass::Opset1Upgrade>();
    pass_manager.run_passes(f);

    const auto pass_replacement_node =
        f->get_result()->input(0).get_source_output().get_node_shared_ptr();
    const auto strided_slice_v1 = as_type_ptr<op::v1::StridedSlice>(pass_replacement_node);

    EXPECT_EQ(strided_slice_v1->description(), "DynSlice");
    EXPECT_EQ(strided_slice_v1->get_version(), 1);
    EXPECT_EQ(strided_slice_v1->get_begin_mask(), vector<int64_t>({1, 0, 1, 0}));
    EXPECT_EQ(strided_slice_v1->get_end_mask(), vector<int64_t>({1, 1, 1, 0}));
    EXPECT_EQ(strided_slice_v1->get_new_axis_mask(), vector<int64_t>({0, 0, 1, 1}));
    EXPECT_EQ(strided_slice_v1->get_shrink_axis_mask(), vector<int64_t>({0, 0, 0, 0}));
    EXPECT_EQ(strided_slice_v1->get_ellipsis_mask(), vector<int64_t>({1, 1, 1, 1}));
}

TEST(opset_transform, opset1_dyn_slice_upgrade_pass_input_dynamic_rank)
{
    auto arg = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto lower_bounds = make_shared<op::Parameter>(element::i64, Shape{4});
    auto upper_bounds = make_shared<op::Parameter>(element::i64, Shape{4});
    auto strides = make_shared<op::Parameter>(element::i64, Shape{4});

    auto dyn_slice_v0 = make_shared<op::v0::DynSlice>(arg, lower_bounds, upper_bounds, strides);

    const auto result = make_shared<op::Result>(dyn_slice_v0);
    auto f = make_shared<Function>(ResultVector{result},
                                   ParameterVector{arg, lower_bounds, upper_bounds, strides});

    try
    {
        ngraph::pass::Manager pass_manager;
        pass_manager.register_pass<pass::Opset1Upgrade>();
        pass_manager.run_passes(f);
        FAIL() << "Exception after Opset1Upgrade pass was not thrown.";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string(
                "Unable to convert DynSlice:0 to StridedSlice:1 when input rank is dynamic."));
    }
    catch (...)
    {
        FAIL() << "DynSlice pass failed for unexpected reason";
    }
}

TEST(opset_transform, opset1_strided_slice_downgrade_pass)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{2, 4, 6, 8});
    auto begin = make_shared<op::Parameter>(element::i64, Shape{4});
    auto end = make_shared<op::Parameter>(element::i64, Shape{4});

    auto strided_slice_v1 = make_shared<op::v1::StridedSlice>(data,
                                                              begin,
                                                              end,
                                                              vector<int64_t>{1, 0, 1, 0},
                                                              vector<int64_t>{1, 1, 1, 0},
                                                              vector<int64_t>{0, 0, 1, 1},
                                                              vector<int64_t>{0, 0, 0, 0},
                                                              vector<int64_t>{1, 1, 1, 1});

    const auto result = make_shared<op::Result>(strided_slice_v1);
    auto f = make_shared<Function>(ResultVector{result}, ParameterVector{data, begin, end});

    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<pass::Opset0Downgrade>();
    pass_manager.run_passes(f);

    const auto pass_replacement_node =
        f->get_result()->input(0).get_source_output().get_node_shared_ptr();
    const auto dyn_slice_v0 = as_type_ptr<op::v0::DynSlice>(pass_replacement_node);

    EXPECT_EQ(dyn_slice_v0->description(), "DynSlice");
    EXPECT_EQ(dyn_slice_v0->get_version(), 0);
    EXPECT_EQ(dyn_slice_v0->get_lower_bounds_mask(), AxisSet({0, 2}));
    EXPECT_EQ(dyn_slice_v0->get_upper_bounds_mask(), AxisSet({0, 1, 2}));
    EXPECT_EQ(dyn_slice_v0->get_new_axis(), AxisSet({2, 3}));
    EXPECT_EQ(dyn_slice_v0->get_shrink_axis(), AxisSet({}));
    EXPECT_EQ(dyn_slice_v0->get_ellipsis_mask(), AxisSet({0, 1, 2, 3}));
}
