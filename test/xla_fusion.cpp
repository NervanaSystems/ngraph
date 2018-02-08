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
#include <iostream>
#include <list>
#include <memory>

#include "gtest/gtest.h"
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/ops/sum.hpp"
#include "ngraph/pass/graph_rewrite.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pattern/matcher.hpp"
#include "ngraph/pattern/op/any.hpp"
#include "ngraph/pattern/op/label.hpp"
#include "util/test_tools.hpp"

//
#include <fstream>
#include <sstream>
#include "ngraph/file_util.hpp"
#include "ngraph/pass/visualize_tree.hpp"
#include "ngraph/serializer.hpp"

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

        //auto divisor = make_shared<op::Broadcast>(div_constant, Shape{ 2, 21, 21, 32 }, AxisSet{ 0, 1, 2, 3 });
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

            for (size_t i = 0; i < mrw->get_shape().size(); i++)
            {
                if (mrw->get_window_shape().at(i) != 1)
                {
                    ap_shape.push_back(mrw->get_window_shape().at(i));
                    ap_strides.push_back(mrw->get_window_movement_strides().at(i));
                }
            }

            if (!ap_shape.size())
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
            auto ap =
                std::shared_ptr<Node>(new op::AvgPool(pattern_map[input], ap_shape, ap_strides));
            NGRAPH_DEBUG << "Created ap = " << ap->get_name();
            return ap;
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
    const string json_path = file_util::path_join(SERIALIZED_ZOO, "mxnet/avgpool.json");
    const string json_string = file_util::read_file_to_string(json_path);
    stringstream ss(json_string);
    shared_ptr<Function> func = ngraph::deserialize(ss);

    string js = serialize(func, 4);
    {
        ofstream f("avgpool.js");
        f << js;
    }

    pass_manager.run_passes(func);
    ASSERT_GT(count_ops_of_type<op::AvgPool>(func), 0);
}