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

TEST(opset_transform, opset1_softmax_upgrade_pass_axis)
{
    const size_t axis = 2;
    const AxisSet axes{axis};
    auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 3, 4});
    auto softmax_s0 = make_shared<op::v0::Softmax>(arg, axes);
    auto result = make_shared<op::Result>(softmax_s0);
    auto f = make_shared<Function>(ResultVector{result}, ParameterVector{arg});

    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<pass::Opset1Upgrade>();
    pass_manager.run_passes(f);

    auto softmax_s1_result = f->get_results().at(0);
    auto node = softmax_s1_result->input(0).get_source_output().get_node_shared_ptr();
    auto softmax_s1_node = static_pointer_cast<op::v1::Softmax>(node);

    EXPECT_EQ(softmax_s1_node->get_axis(), axis);
    EXPECT_EQ(softmax_s1_node->description(), "Softmax");
    EXPECT_EQ(softmax_s1_node->get_version(), 1);
}

TEST(opset_transform, opset1_softmax_upgrade_pass_axis_exception)
{
    const AxisSet axes{1, 2};
    auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 3, 4});
    auto softmax_s0 = make_shared<op::v0::Softmax>(arg, axes);
    auto result = make_shared<op::Result>(softmax_s0);
    auto f = make_shared<Function>(ResultVector{result}, ParameterVector{arg});

    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<pass::Opset1Upgrade>();

    try
    {
        pass_manager.run_passes(f);
        FAIL() << "Exception after Opset1Upgrade pass was not thrown.";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string(
                "Unable to convert Softmax:0 to Softmax:1 with zero or more than one axis."));
    }
    catch (...)
    {
        FAIL() << "Softmax pass failed for unexpected reason";
    }
}

namespace fake_v2
{
    class FakeSoftmax : public op::v0::Softmax
    {
    public:
        FakeSoftmax(const Output<Node>& arg, const AxisSet& axes)
            : Softmax{arg, axes}
        {
        }
        size_t get_version() const override { return 2; }
    };
}

TEST(opset_transform, opset1_softmax_upgrade_pass_incorrect_op_version)
{
    const AxisSet axes{2};
    auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 3, 4});
    auto softmax_s2 = make_shared<fake_v2::FakeSoftmax>(arg, axes);
    auto result = make_shared<op::Result>(softmax_s2);
    auto f = make_shared<Function>(ResultVector{result}, ParameterVector{arg});

    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<pass::Opset1Upgrade>();

    try
    {
        pass_manager.run_passes(f);
        FAIL() << "Opset 1 transformation pass failed for";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Op version 1 transformation pass failed for"));
    }
    catch (...)
    {
        FAIL() << "Softmax pass failed for unexpected reason";
    }
}
