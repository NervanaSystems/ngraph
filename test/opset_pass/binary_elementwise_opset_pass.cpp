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
void test_type_prop_opset0_downgrade_pass(const element::Type& output_type,
                                          const element::Type& input_type = element::f32,
                                          const string node_name = "")
{
    auto A = make_shared<op::Parameter>(input_type, Shape{1, 3, 2});
    auto B = make_shared<op::Parameter>(input_type, Shape{1, 2});
    const op::AutoBroadcastSpec np_auto_b = op::AutoBroadcastSpec(op::AutoBroadcastType::NUMPY);

    auto v1_node = make_shared<OpV1>(A, B);
    auto result = make_shared<op::Result>(v1_node);
    auto f = make_shared<Function>(ResultVector{result}, ParameterVector{A, B});

    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<pass::Opset0Downgrade>();
    pass_manager.run_passes(f);

    auto v0_result = f->get_results().at(0);
    auto node = v0_result->input(0).get_source_output().get_node_shared_ptr();
    auto v0_node = static_pointer_cast<OpV0>(node);

    EXPECT_EQ(v0_node->description(), (node_name.empty() ? v1_node->description() : node_name));
    EXPECT_EQ(v0_node->get_version(), 0);
    EXPECT_EQ(v0_node->get_autob(), np_auto_b);
    EXPECT_EQ(v0_node->output(0).get_element_type(), output_type);
    EXPECT_EQ(v0_node->output(0).get_shape(), (Shape{1, 3, 2}));
}

template <typename OpV0, typename OpV1>
void test_opset0_arithmetic_downgrade_pass()
{
    test_type_prop_opset0_downgrade_pass<OpV0, OpV1>(element::f32);
}

template <typename OpV0, typename OpV1>
void test_opset0_comparison_downgrade_pass()
{
    test_type_prop_opset0_downgrade_pass<OpV0, OpV1>(element::boolean);
}

template <typename OpV0, typename OpV1>
void test_type_prop_opset1_upgrade_pass(const element::Type& output_type,
                                        const element::Type& input_type = element::f32,
                                        const string node_name = "")
{
    auto A = make_shared<op::Parameter>(input_type, Shape{1, 3, 2});
    auto B = make_shared<op::Parameter>(input_type, Shape{1, 3, 2});
    const op::AutoBroadcastSpec none_auto_b = op::AutoBroadcastSpec(op::AutoBroadcastType::NONE);

    auto v0_node = make_shared<OpV0>(A, B);
    auto result = make_shared<op::Result>(v0_node);
    auto f = make_shared<Function>(ResultVector{result}, ParameterVector{A, B});

    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<pass::Opset1Upgrade>();
    pass_manager.run_passes(f);

    auto v1_result = f->get_results().at(0);
    auto node = v1_result->input(0).get_source_output().get_node_shared_ptr();
    auto v1_node = static_pointer_cast<OpV1>(node);

    EXPECT_EQ(v1_node->description(), (node_name.empty() ? v0_node->description() : node_name));
    EXPECT_EQ(v1_node->get_version(), 1);
    EXPECT_EQ(v1_node->get_autob(), none_auto_b);
    EXPECT_EQ(v1_node->output(0).get_element_type(), output_type);
    EXPECT_EQ(v1_node->output(0).get_shape(), (Shape{1, 3, 2}));
}

template <typename OpV0, typename OpV1>
void test_opset1_arithmetic_upgrade_pass()
{
    test_type_prop_opset1_upgrade_pass<OpV0, OpV1>(element::f32);
}

template <typename OpV0, typename OpV1>
void test_opset1_comparison_upgrade_pass()
{
    test_type_prop_opset1_upgrade_pass<OpV0, OpV1>(element::boolean);
}

//------------------------------------------------------------------------------
//
//                  Test Cases
//
//------------------------------------------------------------------------------

TEST(opset_transform, opset0_add_downgrade_pass)
{
    test_opset0_arithmetic_downgrade_pass<op::v0::Add, op::v1::Add>();
}

TEST(opset_transform, opset1_add_upgrade_pass)
{
    test_opset1_arithmetic_upgrade_pass<op::v0::Add, op::v1::Add>();
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
    test_opset0_comparison_downgrade_pass<op::v0::Equal, op::v1::Equal>();
}

TEST(opset_transform, opset1_equal_upgrade_pass)
{
    test_opset1_comparison_upgrade_pass<op::v0::Equal, op::v1::Equal>();
}

TEST(opset_transform, opset0_greater_downgrade_pass)
{
    test_opset0_comparison_downgrade_pass<op::v0::Greater, op::v1::Greater>();
}

TEST(opset_transform, opset1_greater_upgrade_pass)
{
    test_opset1_comparison_upgrade_pass<op::v0::Greater, op::v1::Greater>();
}

TEST(opset_transform, opset0_greater_eq_downgrade_pass)
{
    test_opset0_comparison_downgrade_pass<op::v0::GreaterEq, op::v1::GreaterEq>();
}

TEST(opset_transform, opset1_greater_eq_upgrade_pass)
{
    test_opset1_comparison_upgrade_pass<op::v0::GreaterEq, op::v1::GreaterEq>();
}

TEST(opset_transform, opset0_less_downgrade_pass)
{
    test_opset0_comparison_downgrade_pass<op::v0::Less, op::v1::Less>();
}

TEST(opset_transform, opset1_less_upgrade_pass)
{
    test_opset1_comparison_upgrade_pass<op::v0::Less, op::v1::Less>();
}

TEST(opset_transform, opset0_less_eq_downgrade_pass)
{
    test_type_prop_opset0_downgrade_pass<op::v0::LessEq, op::v1::LessEqual>(
        element::boolean, element::f32, "LessEq");
}

TEST(opset_transform, opset1_less_eq_upgrade_pass)
{
    test_type_prop_opset1_upgrade_pass<op::v0::LessEq, op::v1::LessEqual>(
        element::boolean, element::f32, "LessEqual");
}

TEST(opset_transform, opset0_maximum_downgrade_pass)
{
    test_opset0_arithmetic_downgrade_pass<op::v0::Maximum, op::v1::Maximum>();
}

TEST(opset_transform, opset1_maximum_upgrade_pass)
{
    test_opset1_arithmetic_upgrade_pass<op::v0::Maximum, op::v1::Maximum>();
}

TEST(opset_transform, opset0_minimum_downgrade_pass)
{
    test_opset0_arithmetic_downgrade_pass<op::v0::Minimum, op::v1::Minimum>();
}

TEST(opset_transform, opset1_minimum_upgrade_pass)
{
    test_opset1_arithmetic_upgrade_pass<op::v0::Minimum, op::v1::Minimum>();
}

TEST(opset_transform, opset0_multiply_downgrade_pass)
{
    test_opset0_arithmetic_downgrade_pass<op::v0::Multiply, op::v1::Multiply>();
}

TEST(opset_transform, opset1_multiply_upgrade_pass)
{
    test_opset1_arithmetic_upgrade_pass<op::v0::Multiply, op::v1::Multiply>();
}

TEST(opset_transform, opset0_not_equal_downgrade_pass)
{
    test_opset0_comparison_downgrade_pass<op::v0::NotEqual, op::v1::NotEqual>();
}

TEST(opset_transform, opset1_not_equal_upgrade_pass)
{
    test_opset1_comparison_upgrade_pass<op::v0::NotEqual, op::v1::NotEqual>();
}

TEST(opset_transform, opset0_power_downgrade_pass)
{
    test_opset0_arithmetic_downgrade_pass<op::v0::Power, op::v1::Power>();
}

TEST(opset_transform, opset1_power_upgrade_pass)
{
    test_opset1_arithmetic_upgrade_pass<op::v0::Power, op::v1::Power>();
}
