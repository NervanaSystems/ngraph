#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "ngraph/ngraph.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/opset1_upgrade.hpp"
#include "util/test_control.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(upgrade_pass, opset1_convolution_pass)
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
