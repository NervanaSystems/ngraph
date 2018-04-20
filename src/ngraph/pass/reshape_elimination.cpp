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
#include "ngraph/op/add.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/dot.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/pattern/matcher.hpp"
#include "ngraph/pattern/op/any.hpp"
#include "ngraph/pattern/op/label.hpp"
#include "ngraph/util.hpp"

template <typename T>
static std::vector<T> apply_permutation(std::vector<T> input, ngraph::AxisVector order)
{
    if (input.size() != order.size())
    {
        throw "input and order sizes don't match!";
    }

    std::vector<T> output(input.size());

    for (size_t i = 0; i < order.size(); i++)
    {
        output[i] = input.at(order.at(i));
    }

    return output;
}

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
        auto gop = pattern_map[op];

        auto r1 = std::dynamic_pointer_cast<op::Reshape>(m.match_root());

        if (r1->get_shape() != gop->get_shape())
        {
            NGRAPH_DEBUG << "Not a no-op; Shapes are different!";
            return false;
        }

        Shape do_r1(r1->get_shape().size());
        std::iota(begin(do_r1), end(do_r1), 0);

        if (do_r1 != r1->get_input_order())
        {
            NGRAPH_DEBUG << "Not a no-op; Not in default input order!";
            return false;
        }

        ngraph::replace_node(m.match_root(), gop);
        return true;
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

        auto gop = pattern_map[op];

        if (gop->get_shape() != m.match_root()->get_shape())
        {
            NGRAPH_DEBUG << "Operand shape doesn't match the shape of the second reshape!";
            NGRAPH_DEBUG << "gop " << gop->get_name()
                         << "shape = " << vector_to_string(gop->get_shape());
            NGRAPH_DEBUG << "match_root " << m.match_root()->get_name()
                         << "shape = " << vector_to_string(m.match_root()->get_shape());
            return false;
        }

        auto r2 = std::dynamic_pointer_cast<op::Reshape>(m.match_root());
        auto r1 = std::dynamic_pointer_cast<op::Reshape>(r2->get_argument(0));

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
            ngraph::replace_node(m.match_root(), gop);
            return true;
        }

        auto perm1 = apply_permutation(do_r1, r1->get_input_order());
        auto perm2 = apply_permutation(perm1, r2->get_input_order());
        if (perm2 == do_r1)
        {
            NGRAPH_DEBUG << "Two transposes were removed!";
            ngraph::replace_node(m.match_root(), gop);
            return true;
        }

        return false;
    };
    auto m = std::make_shared<ngraph::pattern::Matcher>(reshape2, callback);
    this->add_matcher(m);
}

void ngraph::pass::ReshapeElimination::construct_dot_transpose_pattern()
{
    //dot(A,B).T = dot (B.T, A.T)
    auto dot_pred = [](std::shared_ptr<Node> n) {
        return static_cast<bool>(std::dynamic_pointer_cast<op::Dot>(n));
    };

    auto pdot = std::make_shared<pattern::op::Label>(element::f32, Shape{2, 1}, dot_pred);
    auto preshape = std::make_shared<op::Reshape>(pdot, AxisVector{1, 0}, Shape{1, 2});

    ngraph::pattern::graph_rewrite_callback callback = [](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for construct_dot_transpose_pattern against node = "
                     << m.match_root()->get_name();

        auto mtranspose = std::dynamic_pointer_cast<op::Reshape>(m.match_root());
        //this also checks the rank
        if (mtranspose->get_input_order() != AxisVector{1, 0})
        {
            NGRAPH_DEBUG << "Reshape isn't transpose. "
                         << vector_to_string(mtranspose->get_input_order());
            return false;
        }

        auto mdot = mtranspose->get_argument(0);
        if (mdot->get_shape().size() != 2)
        {
            NGRAPH_DEBUG << "Dot has the wrong shape. " << vector_to_string(mdot->get_shape());
            return false;
        }

        auto arg0 = mdot->get_argument(0);
        auto reshape0_shape = Shape{arg0->get_shape().at(1), arg0->get_shape().at(0)};
        auto reshape0 = std::make_shared<op::Reshape>(arg0, AxisVector{1, 0}, reshape0_shape);

        auto arg1 = mdot->get_argument(1);
        auto reshape1_shape = Shape{arg1->get_shape().at(1), arg1->get_shape().at(0)};
        auto reshape1 = std::make_shared<op::Reshape>(arg1, AxisVector{1, 0}, reshape1_shape);

        auto tdot = std::shared_ptr<Node>(new op::Dot(reshape1, reshape0));
        ngraph::replace_node(m.match_root(), tdot);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(preshape, callback);
    this->add_matcher(m);
}
