#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "ngraph/ngraph.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/opset0_downgrade.hpp"
#include "ngraph/pass/opset1_upgrade.hpp"
#include "util/test_control.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

//------------------------------------------------------------------------------
//
//                  Helper Functions
//
//------------------------------------------------------------------------------

template <typename OpV0, typename OpV1>
static inline void test_opset0_downgrade_pass(const string& node_name,
                                              const element::Type& output_type,
                                              const element::Type& input_type = element::f32)
{
    auto A = make_shared<op::Parameter>(input_type, Shape{1, 3, 2});
    auto B = make_shared<op::Parameter>(input_type, Shape{1, 2});
    const op::AutoBroadcastSpec np_auto_b = op::AutoBroadcastSpec(op::AutoBroadcastType::NUMPY);

    auto add_v1 = make_shared<OpV1>(A, B);
    auto result = make_shared<op::Result>(add_v1);
    auto f = make_shared<Function>(ResultVector{result}, ParameterVector{A, B});

    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<pass::Opset0Downgrade>();
    pass_manager.run_passes(f);

    auto v0_result = f->get_results().at(0);
    auto node = v0_result->input(0).get_source_output().get_node_shared_ptr();
    auto v0_node = static_pointer_cast<OpV0>(node);

    EXPECT_EQ(v0_node->description(), node_name);
    EXPECT_EQ(v0_node->get_version(), 0);
    EXPECT_EQ(v0_node->get_autob(), np_auto_b);
    EXPECT_EQ(v0_node->output(0).get_element_type(), output_type);
    EXPECT_EQ(v0_node->output(0).get_shape(), (Shape{1, 3, 2}));
}

template <typename OpV0, typename OpV1>
static inline void test_opset0_arithmetic_downgrade_pass(const string& node_name)
{
    test_opset0_downgrade_pass<OpV0, OpV1>(node_name, element::f32);
}

template <typename OpV0, typename OpV1>
static inline void test_opset0_comparison_downgrade_pass(const string& node_name)
{
    test_opset0_downgrade_pass<OpV0, OpV1>(node_name, element::boolean);
}

template <typename OpV0, typename OpV1>
static inline void test_opset0_upgrade_pass(const string& node_name,
                                            const element::Type& output_type,
                                            const element::Type& input_type = element::f32)
{
    auto A = make_shared<op::Parameter>(input_type, Shape{1, 3, 2});
    auto B = make_shared<op::Parameter>(input_type, Shape{1, 3, 2});
    const op::AutoBroadcastSpec none_auto_b = op::AutoBroadcastSpec(op::AutoBroadcastType::NONE);

    auto add_v0 = make_shared<OpV0>(A, B);
    auto result = make_shared<op::Result>(add_v0);
    auto f = make_shared<Function>(ResultVector{result}, ParameterVector{A, B});

    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<pass::Opset1Upgrade>();
    pass_manager.run_passes(f);

    auto add_v1_result = f->get_results().at(0);
    auto node = add_v1_result->input(0).get_source_output().get_node_shared_ptr();
    auto add_v1_node = static_pointer_cast<OpV1>(node);

    EXPECT_EQ(add_v1_node->description(), node_name);
    EXPECT_EQ(add_v1_node->get_version(), 1);
    EXPECT_EQ(add_v1_node->get_autob(), none_auto_b);
    EXPECT_EQ(add_v1_node->output(0).get_element_type(), output_type);
    EXPECT_EQ(add_v1_node->output(0).get_shape(), (Shape{1, 3, 2}));
}

template <typename OpV0, typename OpV1>
static inline void test_opset0_arithmetic_upgrade_pass(const string& node_name)
{
    test_opset0_upgrade_pass<OpV0, OpV1>(node_name, element::f32);
}

template <typename OpV0, typename OpV1>
static inline void test_opset0_comparison_upgrade_pass(const string& node_name)
{
    test_opset0_upgrade_pass<OpV0, OpV1>(node_name, element::boolean);
}

//------------------------------------------------------------------------------
//
//                  Test Cases
//
//------------------------------------------------------------------------------

TEST(opset_transform, opset0_add_downgrade_pass)
{
    test_opset0_arithmetic_downgrade_pass<op::v0::Add, op::v1::Add>("Add");
}

TEST(opset_transform, opset1_add_upgrade_pass)
{
    test_opset0_arithmetic_upgrade_pass<op::v0::Add, op::v1::Add>("Add");
}

TEST(opset_transform, opset0_divide_downgrade_pass)
{
    auto A = make_shared<op::Parameter>(element::f32, Shape{1, 3, 2});
    auto B = make_shared<op::Parameter>(element::f32, Shape{1, 2});
    const op::AutoBroadcastSpec np_auto_b = op::AutoBroadcastSpec(op::AutoBroadcastType::NUMPY);
    const bool pydiv = false;

    auto divide_v1 = make_shared<op::v1::Divide>(A, B);
    divide_v1->set_is_pythondiv(pydiv);
    auto result = make_shared<op::Result>(divide_v1);
    auto f = make_shared<Function>(ResultVector{result}, ParameterVector{A, B});

    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<pass::Opset0Downgrade>();
    pass_manager.run_passes(f);

    auto divide_v0_result = f->get_results().at(0);
    auto node = divide_v0_result->input(0).get_source_output().get_node_shared_ptr();
    auto divide_v0_node = static_pointer_cast<op::v0::Divide>(node);

    EXPECT_EQ(divide_v0_node->description(), "Divide");
    EXPECT_EQ(divide_v0_node->get_version(), 0);
    EXPECT_EQ(divide_v0_node->is_pythondiv(), pydiv);
    EXPECT_EQ(divide_v0_node->get_autob(), np_auto_b);
    EXPECT_EQ(divide_v0_node->output(0).get_element_type(), element::f32);
    EXPECT_EQ(divide_v0_node->output(0).get_shape(), (Shape{1, 3, 2}));
}

TEST(opset_transform, opset1_divide_upgrade_pass)
{
    auto A = make_shared<op::Parameter>(element::f32, Shape{1, 3, 2});
    auto B = make_shared<op::Parameter>(element::f32, Shape{1, 3, 2});
    const op::AutoBroadcastSpec none_auto_b = op::AutoBroadcastSpec(op::AutoBroadcastType::NONE);
    const bool pydiv = false;

    auto div_v0 = make_shared<op::v0::Divide>(A, B, pydiv);
    auto result = make_shared<op::Result>(div_v0);
    auto f = make_shared<Function>(ResultVector{result}, ParameterVector{A, B});

    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<pass::Opset1Upgrade>();
    pass_manager.run_passes(f);

    auto divide_v1_result = f->get_results().at(0);
    auto node = divide_v1_result->input(0).get_source_output().get_node_shared_ptr();
    auto divide_v1_node = static_pointer_cast<op::v1::Divide>(node);

    EXPECT_EQ(divide_v1_node->description(), "Divide");
    EXPECT_EQ(divide_v1_node->get_version(), 1);
    EXPECT_EQ(divide_v1_node->is_pythondiv(), pydiv);
    EXPECT_EQ(divide_v1_node->get_autob(), none_auto_b);
    EXPECT_EQ(divide_v1_node->output(0).get_element_type(), element::f32);
    EXPECT_EQ(divide_v1_node->output(0).get_shape(), (Shape{1, 3, 2}));
}

TEST(opset_transform, opset0_equal_downgrade_pass)
{
    test_opset0_comparison_downgrade_pass<op::v0::Equal, op::v1::Equal>("Equal");
}

TEST(opset_transform, opset1_equal_upgrade_pass)
{
    test_opset0_comparison_upgrade_pass<op::v0::Equal, op::v1::Equal>("Equal");
}

TEST(opset_transform, opset0_greater_downgrade_pass)
{
    test_opset0_comparison_downgrade_pass<op::v0::Greater, op::v1::Greater>("Greater");
}

TEST(opset_transform, opset1_greater_upgrade_pass)
{
    test_opset0_comparison_upgrade_pass<op::v0::Greater, op::v1::Greater>("Greater");
}

TEST(opset_transform, opset0_greater_eq_downgrade_pass)
{
    test_opset0_comparison_downgrade_pass<op::v0::GreaterEq, op::v1::GreaterEq>("GreaterEq");
}

TEST(opset_transform, opset1_greater_eq_upgrade_pass)
{
    test_opset0_comparison_upgrade_pass<op::v0::GreaterEq, op::v1::GreaterEq>("GreaterEq");
}
