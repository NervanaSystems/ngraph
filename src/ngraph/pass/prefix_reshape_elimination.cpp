//*****************************************************************************
// Copyright 2017-2019 Intel Corporation
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

#include "ngraph/pass/prefix_reshape_elimination.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/util/binary_elementwise_arithmetic.hpp"
#include "ngraph/op/util/unary_elementwise_arithmetic.hpp"
#include "ngraph/pattern/matcher.hpp"
#include "ngraph/pattern/op/any.hpp"
#include "ngraph/pattern/op/any_of.hpp"
#include "ngraph/pattern/op/label.hpp"

using namespace std;
using namespace ngraph;

pass::PrefixReshapeElimination::PrefixReshapeElimination()
{
    auto src_op = make_shared<pattern::op::Label>(
        element::i8, Shape{}, [](shared_ptr<Node>) { return true; });
    auto reshape_op = make_shared<pattern::op::Any>(
        element::i8,
        Shape{},
        [](shared_ptr<Node> node) {
            op::Reshape* reshape = dynamic_cast<op::Reshape*>(node.get());
            if (!reshape)
            {
                return false;
            }

            // Validate that this isn't a reordering-reshape.
            if (reshape->get_is_transpose())
            {
                return false;
            }

            // Make sure that logical dimension sizes match.
            const Shape& src_shape = reshape->get_input_shape(0);
            for (size_t idx = 0; idx < reshape->get_output_shape().size(); ++idx)
            {
                size_t src_size = 1;
                if (idx < src_shape.size())
                {
                    src_size = src_shape.at(src_shape.size() - 1 - idx);
                }
                size_t dest_size =
                    reshape->get_output_shape().at(reshape->get_output_shape().size() - 1 - idx);
                if (dest_size != src_size)
                {
                    return false;
                }
            }

            return true;
        },
        NodeVector{src_op});
    auto target_op = make_shared<pattern::op::AnyOf>(
        element::i8,
        Shape{},
        [](shared_ptr<Node> node) {
            return pattern::has_class<op::util::UnaryElementwiseArithmetic>()(node) ||
                   pattern::has_class<op::util::BinaryElementwiseArithmetic>()(node);
        },
        NodeVector{reshape_op});

    auto callback = [](pattern::Matcher& m) {
        replace_node(m.get_matched_nodes().at(1), m.get_matched_nodes().at(2));
        return true;
    };
    add_matcher(make_shared<pattern::Matcher>(target_op, "PrefixReshapeElimination"),
                callback,
                PassProperty::REQUIRE_STATIC_SHAPE);
}
