/*******************************************************************************
* Copyright 2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "reshape_elimination.hpp"
#include <algorithm>
#include <iostream>
#include <numeric>
#include <unordered_set>

#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/ops/add.hpp"
#include "ngraph/ops/broadcast.hpp"
#include "ngraph/ops/broadcast.hpp"
#include "ngraph/ops/dot.hpp"
#include "ngraph/ops/parameter.hpp"
#include "ngraph/ops/reshape.hpp"
#include "ngraph/pattern/matcher.hpp"
#include "ngraph/pattern/op/any.hpp"
#include "ngraph/pattern/op/label.hpp"
#include "ngraph/util.hpp"

extern template std::vector<size_t> ngraph::apply_permutation(std::vector<size_t> input,
                                                              ngraph::AxisVector order);

void ngraph::pass::ReshapeElimination::construct_identity_reshape_pattern()
{
    Shape shape_op{3};
    Shape shape_r1{1, 3};

    auto op = std::make_shared<pattern::op::Label>(element::f32, shape_op);
    auto reshape1 = std::make_shared<op::Reshape>(op, AxisVector{0}, shape_r1);

    auto callback = [op](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for construct_identity_reshape_pattern against node = "
                     << m.match_root()->get_name();
        auto pattern_map = m.get_pattern_map();

        std::shared_ptr<ngraph::Node> nn;
        auto gop = pattern_map[op];

        auto r1 = std::dynamic_pointer_cast<op::Reshape>(m.match_root());

        if (r1->get_shape() != gop->get_shape())
        {
            NGRAPH_DEBUG << "Not a no-op; Shapes are different!";
            return nn;
        }

        Shape do_r1(r1->get_shape().size());
        std::iota(begin(do_r1), end(do_r1), 0);

        if (do_r1 != r1->get_input_order())
        {
            NGRAPH_DEBUG << "Not a no-op; Not in default input order!";
            return nn;
        }

        return gop;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(reshape1, callback);
    this->add_matcher(m);
}

void ngraph::pass::ReshapeElimination::construct_reshapex2_pattern()
{
    Shape shape_op{3};
    Shape shape_r1{1, 3};

    auto op = std::make_shared<pattern::op::Label>(element::f32, shape_op);
    auto reshape1 = std::make_shared<op::Reshape>(op, AxisVector{0}, shape_r1);
    auto reshape2 = std::make_shared<op::Reshape>(reshape1, AxisVector{0, 1}, shape_op);

    auto callback = [op](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for construct_reshapex2_pattern against node = "
                     << m.match_root()->get_name();
        auto pattern_map = m.get_pattern_map();

        std::shared_ptr<ngraph::Node> nn;
        auto gop = pattern_map[op];

        if (gop->get_shape() != m.match_root()->get_shape())
        {
            NGRAPH_DEBUG << "Operand shape doesn't match the shape of the second reshape!";
            NGRAPH_DEBUG << "gop " << gop->get_name()
                         << "shape = " << vector_to_string(gop->get_shape());
            NGRAPH_DEBUG << "match_root " << m.match_root()->get_name()
                         << "shape = " << vector_to_string(m.match_root()->get_shape());
            return nn;
        }

        auto r2 = std::dynamic_pointer_cast<op::Reshape>(m.match_root());
        auto r1 = std::dynamic_pointer_cast<op::Reshape>(r2->get_input_op(0));

        Shape do_r2(r1->get_shape().size());
        std::iota(begin(do_r2), end(do_r2), 0);
        Shape do_r1(gop->get_shape().size());
        std::iota(begin(do_r1), end(do_r1), 0);

        NGRAPH_DEBUG << "r1's i/o = " << vector_to_string(r1->get_input_order())
                     << "do_r1 = " << vector_to_string(do_r1);
        NGRAPH_DEBUG << "r2's i/o = " << vector_to_string(r2->get_input_order())
                     << "do_r2 = " << vector_to_string(do_r2);

        if (r1->get_input_order() == do_r1 && r2->get_input_order() == do_r2)
        {
            NGRAPH_DEBUG << "Two reshapes were removed!";
            return gop;
        }

        auto perm1 = ngraph::apply_permutation(do_r1, r1->get_input_order());
        auto perm2 = ngraph::apply_permutation(perm1, r2->get_input_order());
        if (perm2 == do_r1)
        {
            NGRAPH_DEBUG << "Two transposes were removed!";
            return gop;
        }

        return nn;
    };
    auto m = std::make_shared<ngraph::pattern::Matcher>(reshape2, callback);
    this->add_matcher(m);
}
