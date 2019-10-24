#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "ngraph/ngraph.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/opset0_downgrade.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(opset_transform, opset1_avgpool_downgrade_pass)
{
    Shape window_shape{2, 2};
    auto window_movement_strides = Strides{1, 1};
    Shape padding_below{0, 0};
    Shape padding_above{0, 0};
    auto delta = make_shared<op::Parameter>(element::f32, Shape{2, 2, 2, 2});
    auto forward_arg_shape = op::Constant::create<int64_t>(element::i64, Shape{4}, {2, 2, 3, 3});

    auto avgpool_v1 = make_shared<op::v1::AvgPoolBackprop>(delta,
                                                           forward_arg_shape,
                                                           window_movement_strides,
                                                           padding_below,
                                                           padding_above,
                                                           window_shape,
                                                           false);
    auto f = make_shared<Function>(NodeVector{avgpool_v1}, ParameterVector{delta});

    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<pass::Opset0Downgrade>();
    pass_manager.run_passes(f);

    auto avgpool_v0 = static_pointer_cast<op::v0::AvgPoolBackprop>(
        f->get_results().at(0)->input_value(0).get_node_shared_ptr());

    EXPECT_EQ(avgpool_v0->description(), "AvgPoolBackprop");
    EXPECT_EQ(avgpool_v0->get_version(), 0);
    EXPECT_EQ(avgpool_v0->get_forward_arg_shape(), (Shape{2, 2, 3, 3}));
}
