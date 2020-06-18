//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#include "reshape_elimination_v1.hpp"
#include <algorithm>
#include <iostream>
#include <numeric>
#include <unordered_set>

#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/pattern/matcher.hpp"
#include "ngraph/pattern/op/label.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

void pass::ReshapeEliminationV1::construct_identity_reshape_pattern()
{
    auto op = make_shared<pattern::op::Label>(element::f32, Shape{1, 3});
    auto rpattern = make_shared<pattern::op::Label>(element::i64, Shape{1});
    auto reshape1 = make_shared<op::v1::Reshape>(op, rpattern, false);

    auto callback = [op](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for construct_identity_reshape_pattern against node = "
                     << m.get_match_root()->get_name();
        auto pvm = m.get_pattern_value_map();
        auto gop = pvm[op];

        auto r1 = m.get_match_root_as<op::v1::Reshape>();
        NGRAPH_CHECK(r1, "match root node ", *m.get_match_root(), " not of type `op::v1::Reshape`");

        if (gop.get_partial_shape().is_dynamic() || r1->get_output_partial_shape(0).is_dynamic() ||
            gop.get_shape() != r1->get_output_shape(0))
        {
            NGRAPH_DEBUG << "Not a no-op; Shapes are different!";
            return false;
        }

        gop.get_node()->set_friendly_name(r1->get_friendly_name());
        r1->output(0).replace(gop);
        return true;
    };

    auto m = make_shared<pattern::Matcher>(reshape1);
    this->add_matcher(m, callback, PassProperty::REQUIRE_STATIC_SHAPE);
}
