#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "ngraph/ngraph.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/opset1_upgrade.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(serialize, opset1_pad_pass)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{5, 6});
    auto arg_pad_value = make_shared<op::Parameter>(element::f32, Shape{});
    CoordinateDiff padding_below{1, 2};
    CoordinateDiff padding_above{3, 4};
    auto pad_mode = op::PadMode::EDGE;

    auto pad_v0 =
        make_shared<op::v0::Pad>(arg, arg_pad_value, padding_below, padding_above, pad_mode);
    auto result = make_shared<op::Result>(pad_v0);
    auto f = make_shared<Function>(ResultVector{result}, ParameterVector{arg, arg_pad_value});

    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<pass::Opset1Upgrade>();
    pass_manager.run_passes(f);

    auto pad_s1_result = f->get_results().at(0);
    auto node = pad_s1_result->input(0).get_source_output().get_node_shared_ptr();
    auto pad_v1_node = static_pointer_cast<op::v1::Pad>(node);

    EXPECT_EQ(pad_v1_node->description(), "Pad");
    EXPECT_EQ(pad_v1_node->get_version(), 1);
    EXPECT_EQ(pad_v1_node->get_pad_mode(), pad_mode);

    EXPECT_EQ(pad_v1_node->get_pads_begin(), padding_below);
    EXPECT_EQ(pad_v1_node->get_pads_end(), padding_above);
}
