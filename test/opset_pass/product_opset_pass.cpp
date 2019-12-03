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

TEST(opset_transform, opset1_product_upgrade_pass)
{
    const auto data = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3});
    const AxisSet reduction_axes{1, 2};

    const auto product_v0 = make_shared<op::Product>(data, reduction_axes);
    const auto result = make_shared<op::Result>(product_v0);
    auto f = make_shared<Function>(ResultVector{result}, ParameterVector{data});

    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<pass::Opset1Upgrade>();
    pass_manager.run_passes(f);

    const auto pass_replacement_node =
        f->get_result()->input(0).get_source_output().get_node_shared_ptr();
    const auto reduce_prod_v1 = as_type_ptr<op::v1::ReduceProd>(pass_replacement_node);
    ASSERT_TRUE(reduce_prod_v1);
    EXPECT_EQ(reduce_prod_v1->get_keep_dims(), false);
}

TEST(opset_transform, opset0_reduce_prod_downgrade_pass)
{
    const auto data = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3});
    const auto axes = make_shared<op::Constant>(element::i64, Shape{2}, vector<int64_t>{0, 1});

    const auto product_v1 = make_shared<op::v1::ReduceProd>(data, axes, true);
    const auto result = make_shared<op::Result>(product_v1);
    auto f = make_shared<Function>(ResultVector{result}, ParameterVector{data});

    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<pass::Opset0Downgrade>();
    pass_manager.run_passes(f);

    const auto reshape_replacement_node =
        f->get_result()->input(0).get_source_output().get_node_shared_ptr();
    const auto reshape = as_type_ptr<op::Reshape>(reshape_replacement_node);
    ASSERT_TRUE(reshape);
    const auto product_replace_node =
        reshape_replacement_node->input(0).get_source_output().get_node_shared_ptr();
    const auto product_v0 = as_type_ptr<op::v0::Product>(product_replace_node);
    ASSERT_TRUE(product_v0);
}

TEST(opset_transform, opset0_reduce_prod_downgrade_pass_axes_not_constant)
{
    const auto data = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3});
    const auto axes = make_shared<op::Parameter>(element::f32, Shape{1});

    const auto product_v1 = make_shared<op::v1::ReduceProd>(data, axes, true);
    const auto result = make_shared<op::Result>(product_v1);
    auto f = make_shared<Function>(ResultVector{result}, ParameterVector{data, axes});

    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<pass::Opset0Downgrade>();
    try
    {
        pass_manager.run_passes(f);
        FAIL() << "Exception after Opset0Downgrade pass was not thrown.";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Unable to convert ReduceProd:v1 to Product:v0 "
                        "if reduction axes are not constant (for keep_dims=true)"));
    }
    catch (...)
    {
        FAIL() << "ReduceProd pass failed for unexpected reason";
    }
}

TEST(opset_transform, opset0_reduce_prod_downgrade_pass_output_not_static)
{
    const auto data = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    const auto axes = make_shared<op::Constant>(element::i64, Shape{2}, vector<int64_t>{0, 1});

    const auto product_v1 = make_shared<op::v1::ReduceProd>(data, axes, true);
    const auto result = make_shared<op::Result>(product_v1);
    auto f = make_shared<Function>(ResultVector{result}, ParameterVector{data});

    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<pass::Opset0Downgrade>();
    try
    {
        pass_manager.run_passes(f);
        FAIL() << "Exception after Opset0Downgrade pass was not thrown.";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Unable to convert ReduceProd:v1 to Product:v0 "
                                         "if output shape is dynamic (for keep_dims=true)"));
    }
    catch (...)
    {
        FAIL() << "ReduceProd pass failed for unexpected reason";
    }
}

TEST(opset_transform, opset0_reduce_prod_downgrade_pass_out_shape_if_keep_dims)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{3, 4, 5});
    auto axes = make_shared<op::Constant>(element::i64, Shape{2}, vector<int64_t>{1, 2});
    auto keep_dims = true;
    auto reduce_prod_v1 = make_shared<op::v1::ReduceProd>(arg, axes, keep_dims);
    const auto result = make_shared<op::Result>(reduce_prod_v1);
    auto f = make_shared<Function>(ResultVector{result}, ParameterVector{arg});

    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<pass::Opset0Downgrade>();
    pass_manager.run_passes(f);

    const auto replacement_node =
        f->get_result()->input(0).get_source_output().get_node_shared_ptr();

    ASSERT_TRUE(replacement_node->get_output_partial_shape(0).compatible(PartialShape{3, 1, 1}));
}

TEST(opset_transform, opset0_reduce_prod_downgrade_pass_out_shape_if_not_keep_dims)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{3, 4, 5});
    auto axes = make_shared<op::Constant>(element::i64, Shape{2}, vector<int64_t>{1, 2});
    auto keep_dims = false;
    auto reduce_prod_v1 = make_shared<op::v1::ReduceProd>(arg, axes, keep_dims);
    const auto result = make_shared<op::Result>(reduce_prod_v1);
    auto f = make_shared<Function>(ResultVector{result}, ParameterVector{arg});

    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<pass::Opset0Downgrade>();
    pass_manager.run_passes(f);

    const auto replacement_node =
        f->get_result()->input(0).get_source_output().get_node_shared_ptr();

    ASSERT_TRUE(replacement_node->get_output_partial_shape(0).compatible(PartialShape{3}));
}
