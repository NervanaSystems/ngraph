// ----------------------------------------------------------------------------
// Copyright 2018 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <iterator>
#include <list>
#include <memory>
#include <sstream>

#include "gtest/gtest.h"
#include "ngraph/file_util.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/ops/sum.hpp"
#include "ngraph/pass/graph_rewrite.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pattern/matcher.hpp"
#include "ngraph/pattern/op/any.hpp"
#include "ngraph/pattern/op/label.hpp"
#include "ngraph/serializer.hpp"
#include "util/test_tools.hpp"

//
#include "ngraph/pass/visualize_tree.hpp"

using namespace ngraph;
using namespace std;

class XLAFusion : public ngraph::pass::GraphRewrite
{
public:
    void construct_avgpool()
    {
        auto rw_constant = op::Constant::create(element::f32, Shape{}, {0.f});
        auto input = std::make_shared<pattern::op::Label>(element::f32, Shape{2, 28, 28, 32});

        auto RA = make_shared<op::Parameter>(element::f32, Shape{});
        auto RB = make_shared<op::Parameter>(element::f32, Shape{});
        auto rf = make_shared<Function>(RA + RB, op::Parameters{RA, RB});

        auto window_shape = Shape{1, 8, 8, 1};
        auto window_movement_strides = Strides{1, 1, 1, 1};
        auto rw = make_shared<op::ReduceWindow>(
            input, rw_constant, rf, window_shape, window_movement_strides);

        auto div_constant = op::Constant::create(element::f32, Shape{}, {64.f});

        auto broadcast_pred = [](std::shared_ptr<Node> n) {
            return static_cast<bool>(std::dynamic_pointer_cast<op::Broadcast>(n));
        };

        auto divisor = std::make_shared<pattern::op::Any>(
            div_constant, element::f32, Shape{2, 21, 21, 32}, broadcast_pred);

        auto division = rw / divisor;

        ngraph::pattern::gr_callback_fn callback = [input](pattern::Matcher& m) {

            NGRAPH_DEBUG << "In a callback for construct_avgpool against "
                         << m.match_root()->get_name();

            //check if window and stride and input are all 4D we don't care about other cases much
            auto mrw = std::dynamic_pointer_cast<op::ReduceWindow>(m.match_root()->get_input_op(0));

            std::shared_ptr<ngraph::Node> nn;

            if (mrw->get_shape().size() != 4)
            {
                NGRAPH_DEBUG << "Input isn't 4D tensor";
                return nn;
            }

            auto add = mrw->get_functions().at(0)->get_result();

            auto f_a = make_shared<op::Parameter>(element::f32, Shape{});
            auto f_b = make_shared<op::Parameter>(element::f32, Shape{});

            ngraph::pattern::Matcher f_m(f_a + f_b);
            if (!f_m.match(add))
            {
                NGRAPH_DEBUG << "Reduce function didn't match to parm1 + parm2";
                return nn;
            }

            if (add->get_input_op(0) == add->get_input_op(1))
            {
                return nn;
            }

            Shape ap_shape;
            Strides ap_strides;

            AxisVector number_channel;
            AxisVector image_dims;
            for (size_t i = 0; i < mrw->get_shape().size(); i++)
            {
                if (mrw->get_window_shape().at(i) != 1)
                {
                    ap_shape.push_back(mrw->get_window_shape().at(i));
                    ap_strides.push_back(mrw->get_window_movement_strides().at(i));
                    image_dims.push_back(i);
                }
                else
                {
                    number_channel.push_back(i);
                }
            }

            if (ap_shape.size() != 2)
            {
                return nn;
            }

            /*
			if (!is_equal_to_const_value(to_string(shape_size(ap_shape)).c_str(), m.match_root()->get_input_op(1)))
			{
                NGRAPH_DEBUG << "ap_shape = " << vector_to_string(ap_shape);
                NGRAPH_DEBUG << "constant = " << m.match_root()->get_input_op(1)->get_name();
				return nn;
			}
            */

            auto pattern_map = m.get_pattern_map();
            std::copy(image_dims.begin(), image_dims.end(), back_inserter(number_channel));

            const auto& input_shape = pattern_map[input]->get_shape();
            std::vector<std::size_t> reshape_shape;
            std::transform(number_channel.begin(),
                           number_channel.end(),
                           std::back_inserter(reshape_shape),
                           [&input_shape](size_t i) { return input_shape.at(i); });

            auto reshape =
                make_shared<op::Reshape>(pattern_map[input], number_channel, reshape_shape);

            auto ap = std::shared_ptr<Node>(new op::AvgPool(reshape, ap_shape, ap_strides));

            //compute inverse axis order to revert layout to what it used to be
            AxisVector inverse_reshape_axis_order;
            for (size_t i = 0; i < reshape_shape.size(); i++)
            {
                auto pos = std::find(begin(number_channel), end(number_channel), i);
                inverse_reshape_axis_order.push_back(std::distance(begin(number_channel), pos));
            }

            auto inverse_output_shape =
                ngraph::apply_permutation(ap->get_shape(), inverse_reshape_axis_order);
            auto inverse_reshape = std::shared_ptr<Node>(
                new op::Reshape(ap, inverse_reshape_axis_order, inverse_output_shape));

            return inverse_reshape;
        };

        auto m = make_shared<ngraph::pattern::Matcher>(division, callback);
        this->add_matcher(m);
    }

    XLAFusion()
        : GraphRewrite()
    {
        construct_avgpool();
    }
};

TEST(xla_fusion, avgpool)
{
    pass::Manager pass_manager;
    pass_manager.register_pass<pass::VisualizeTree>("avgpool_before.pdf");
    pass_manager.register_pass<XLAFusion>();
    pass_manager.register_pass<pass::VisualizeTree>("avgpool_after.pdf");
    const string json_path = file_util::path_join(SERIALIZED_ZOO, "tf/avgpool.json");
    const string json_string = file_util::read_file_to_string(json_path);
    stringstream ss(json_string);
    shared_ptr<Function> func = ngraph::deserialize(ss);
    pass_manager.run_passes(func);
    ASSERT_GT(count_ops_of_type<op::AvgPool>(func), 0);
}
