//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
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

#include "ngraph/runtime/plaidml/plaidml_pass_implicit_broadcast.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/util/binary_elementwise_arithmetic.hpp"
#include "ngraph/op/util/unary_elementwise_arithmetic.hpp"
#include "ngraph/pattern/matcher.hpp"
#include "ngraph/pattern/op/any_of.hpp"
#include "ngraph/pattern/op/label.hpp"

ngraph::runtime::plaidml::pass::ImplicitBroadcast::ImplicitBroadcast()
{
    auto src_op = std::make_shared<pattern::op::Label>(
        element::i8, Shape{}, [](std::shared_ptr<Node>) { return true; });
    auto broadcast_op = std::make_shared<op::Broadcast>(src_op, Shape{}, AxisSet{});

    auto target_op = std::make_shared<pattern::op::AnyOf>(
        element::i8,
        Shape{},
        [](std::shared_ptr<Node> node) {
            return pattern::has_class<op::util::UnaryElementwiseArithmetic>()(node) ||
                   pattern::has_class<op::util::BinaryElementwiseArithmetic>()(node);
        },
        NodeVector{broadcast_op});

    auto callback = [](pattern::Matcher& m) {
        // Since the broadcast is going to an elementwise operation, we
        // can always replace it with an equivalent reshape that uses ones
        // for the broadcast axes.
        auto src = m.get_matched_nodes().at(2);
        Shape src_shape = src->get_shape();
        auto broadcast = std::static_pointer_cast<op::Broadcast>(m.get_matched_nodes().at(1));

        AxisVector reshape_order;
        Shape reshape_shape;
        std::size_t input_dim = 0;
        std::size_t didx_limit = broadcast->get_broadcast_shape().size();
        for (std::size_t didx = 0; didx < didx_limit; ++didx)
        {
            if (broadcast->get_broadcast_axes().count(didx))
            {
                reshape_shape.emplace_back(1);
            }
            else
            {
                reshape_order.emplace_back(input_dim);
                reshape_shape.emplace_back(src_shape.at(input_dim++));
            }
        }

        auto reshape = std::make_shared<op::Reshape>(src, reshape_order, reshape_shape);

        replace_node(broadcast, reshape);

        return true;
    };
    add_matcher(std::make_shared<pattern::Matcher>(target_op), callback);
}
