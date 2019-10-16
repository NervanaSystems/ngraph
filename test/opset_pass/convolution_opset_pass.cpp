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

TEST(opset_transform, opset1_convolution_upgrade_pass)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{1, 3, 6, 9});
    auto filters = make_shared<op::Parameter>(element::f32, Shape{1, 3, 3, 3});
    CoordinateDiff pads_begin{0, 0};
    CoordinateDiff pads_end{0, 0};
    Strides strides{1, 1};
    Strides dilations{1, 1};
    Strides data_dilations_strides{1, 1};
    op::PadType pad_type = op::PadType::EXPLICIT;

    auto convolution_v0 = make_shared<op::v0::Convolution>(
        data, filters, strides, dilations, pads_begin, pads_end, data_dilations_strides, pad_type);
    auto result = make_shared<op::Result>(convolution_v0);
    auto f = make_shared<Function>(ResultVector{result}, ParameterVector{data, filters});

    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<pass::Opset1Upgrade>();
    pass_manager.run_passes(f);

    auto convolution_s1_result = f->get_results().at(0);
    auto node = convolution_s1_result->input(0).get_source_output().get_node_shared_ptr();
    auto convolution_v1_node = static_pointer_cast<op::v1::Convolution>(node);

    EXPECT_EQ(convolution_v1_node->description(), "Convolution");
    EXPECT_EQ(convolution_v1_node->get_version(), 1);

    EXPECT_EQ(convolution_v1_node->get_pads_begin(), pads_begin);
    EXPECT_EQ(convolution_v1_node->get_pads_end(), pads_end);
    EXPECT_EQ(convolution_v1_node->get_strides(), strides);
    EXPECT_EQ(convolution_v1_node->get_auto_pad(), pad_type);
    EXPECT_EQ(convolution_v1_node->get_dilations(), dilations);
}

TEST(opset_transform, opset1_convolution_bprop_data_downgrade_pass)
{
    auto data_batch_shape = op::Constant::create<int64_t>(element::i64, Shape{3}, {64, 3, 100});
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 10});  // filters
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{64, 128, 48}); // output delta
    auto move_strides = Strides{2};
    auto dilation_strides = Strides{1};
    auto padding_below = CoordinateDiff{2};
    auto padding_above = CoordinateDiff{3};
    auto conv = make_shared<op::v1::ConvolutionBackpropData>(param0,
                                                             param1,
                                                             data_batch_shape,
                                                             move_strides,
                                                             dilation_strides,
                                                             padding_below,
                                                             padding_above);

    auto f = make_shared<Function>(NodeVector{conv}, ParameterVector{param0, param1});

    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<pass::Opset0Downgrade>();
    pass_manager.run_passes(f);

    auto conv_v0 = static_pointer_cast<op::v0::ConvolutionBackpropData>(
        f->get_results().at(0)->input_value(0).get_node_shared_ptr());

    EXPECT_EQ(conv_v0->description(), "ConvolutionBackpropData");
    EXPECT_EQ(conv_v0->get_version(), 0);
    EXPECT_EQ(conv_v0->get_data_batch_shape(), (Shape{64, 3, 100}));
}

TEST(opset_transform, opset1_convolution_bprop_filters_downgrade_pass)
{
    auto filters_shape = op::Constant::create<int64_t>(element::i64, Shape{3}, {128, 3, 10});
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{64, 3, 100});  // data batch
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{64, 128, 91}); // output delta
    auto move_strides = Strides{1};
    auto dilate_strides = Strides{1};
    auto padding_below = CoordinateDiff{0};
    auto padding_above = CoordinateDiff{0};
    auto conv = make_shared<op::v1::ConvolutionBackpropFilters>(param0,

                                                                param1,
                                                                filters_shape,
                                                                move_strides,
                                                                dilate_strides,
                                                                padding_below,
                                                                padding_above);
    auto f = make_shared<Function>(NodeVector{conv}, ParameterVector{param0, param1});

    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<pass::Opset0Downgrade>();
    pass_manager.run_passes(f);

    auto conv_v0 = static_pointer_cast<op::v0::ConvolutionBackpropFilters>(
        f->get_results().at(0)->input_value(0).get_node_shared_ptr());

    EXPECT_EQ(conv_v0->description(), "ConvolutionBackpropFilters");
    EXPECT_EQ(conv_v0->get_version(), 0);
    EXPECT_EQ(conv_v0->get_filters_shape(), (Shape{128, 3, 10}));
}
