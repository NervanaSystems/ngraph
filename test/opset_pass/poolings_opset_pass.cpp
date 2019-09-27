#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "ngraph/ngraph.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/opset1_upgrade.hpp"
#include "util/test_control.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(upgrade_pass, opset1_avgpool_pass)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{1, 3, 6, 9});
    Shape pads_begin{0, 0};
    Shape pads_end{0, 0};
    Strides strides{1, 1};
    Shape kernel_shape{3, 3};
    bool include_pad = true;
    bool ceil_mode = false;
    op::PadType pad_mode = op::PadType::EXPLICIT;

    auto avgpool_v0 = make_shared<op::v0::AvgPool>(
        arg, kernel_shape, strides, pads_begin, pads_end, include_pad, pad_mode, ceil_mode);
    auto result = make_shared<op::Result>(avgpool_v0);
    auto f = make_shared<Function>(ResultVector{result}, ParameterVector{arg});

    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<pass::Opset1Upgrade>();
    pass_manager.run_passes(f);

    auto avgpool_s1_result = f->get_results().at(0);
    auto node = avgpool_s1_result->input(0).get_source_output().get_node_shared_ptr();
    auto avg_pool_v1_node = static_pointer_cast<op::v1::AvgPool>(node);

    EXPECT_EQ(avg_pool_v1_node->description(), "AvgPool");
    EXPECT_EQ(avg_pool_v1_node->get_version(), 1);

    EXPECT_EQ(avg_pool_v1_node->get_pads_begin(), pads_begin);
    EXPECT_EQ(avg_pool_v1_node->get_pads_end(), pads_end);
    EXPECT_EQ(avg_pool_v1_node->get_strides(), strides);
    EXPECT_EQ(avg_pool_v1_node->get_kernel(), kernel_shape);
    EXPECT_EQ(avg_pool_v1_node->get_rounding_type(), op::RoundingType::FLOOR);
    EXPECT_EQ(avg_pool_v1_node->get_exclude_pad(), !include_pad);
    EXPECT_EQ(avg_pool_v1_node->get_auto_pad(), pad_mode);
}

TEST(upgrade_pass, opset1_maxpool_pass)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{1, 3, 6, 9});
    Shape pads_begin{0, 0};
    Shape pads_end{0, 0};
    Strides strides{1, 1};
    Shape kernel_shape{3, 3};
    bool ceil_mode = false;
    op::PadType pad_mode = op::PadType::EXPLICIT;

    auto maxpool_v0 = make_shared<op::v0::MaxPool>(
        arg, kernel_shape, strides, pads_begin, pads_end, pad_mode, ceil_mode);
    auto result = make_shared<op::Result>(maxpool_v0);
    auto f = make_shared<Function>(ResultVector{result}, ParameterVector{arg});

    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<pass::Opset1Upgrade>();
    pass_manager.run_passes(f);

    auto maxpool_s1_result = f->get_results().at(0);
    auto node = maxpool_s1_result->input(0).get_source_output().get_node_shared_ptr();
    auto max_pool_v1_node = static_pointer_cast<op::v1::MaxPool>(node);

    EXPECT_EQ(max_pool_v1_node->description(), "MaxPool");
    EXPECT_EQ(max_pool_v1_node->get_version(), 1);

    EXPECT_EQ(max_pool_v1_node->get_pads_begin(), pads_begin);
    EXPECT_EQ(max_pool_v1_node->get_pads_end(), pads_end);
    EXPECT_EQ(max_pool_v1_node->get_strides(), strides);
    EXPECT_EQ(max_pool_v1_node->get_kernel(), kernel_shape);
    EXPECT_EQ(max_pool_v1_node->get_rounding_type(), op::RoundingType::FLOOR);
    EXPECT_EQ(max_pool_v1_node->get_auto_pad(), pad_mode);
}
