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

#include "ngraph/runtime/plaidml/plaidml_pass_implicit_broadcast.hpp"
#include "ngraph/check.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/util/binary_elementwise_arithmetic.hpp"
#include "ngraph/op/util/unary_elementwise_arithmetic.hpp"
#include "ngraph/pattern/matcher.hpp"
#include "ngraph/pattern/op/any_of.hpp"
#include "ngraph/pattern/op/label.hpp"
#include "ngraph/runtime/plaidml/plaidml_ops_implicit_broadcast.hpp"

ngraph::runtime::plaidml::pass::ImplicitBroadcast::ImplicitBroadcast()
{
    auto src_op = std::make_shared<pattern::op::Label>(
        element::i8, Shape{}, [](std::shared_ptr<Node>) { return true; });
    auto broadcast_op = std::make_shared<ngraph::op::Broadcast>(src_op, Shape{}, AxisSet{});

    auto target_op = std::make_shared<pattern::op::AnyOf>(
        element::i8,
        Shape{},
        [](std::shared_ptr<Node> node) {
            return pattern::has_class<ngraph::op::util::UnaryElementwiseArithmetic>()(node) ||
                   pattern::has_class<ngraph::op::util::BinaryElementwiseArithmetic>()(node);
        },
        NodeVector{broadcast_op});

    auto callback = [](pattern::Matcher& m) {
        // Since the broadcast is going to an elementwise operation, we
        // can always replace it with an equivalent reshape that uses ones
        // for the broadcast axes, followed by a fake op that fixes up the
        // shape.
        auto src = m.get_matched_nodes().at(2);
        Shape src_shape = src->get_shape();
        auto broadcast =
            std::static_pointer_cast<ngraph::op::Broadcast>(m.get_matched_nodes().at(1));

        if (src_shape.size())
        {
            // Create a reshape operation to get the right target broadcast shape.  (Note that a
            // zero-D tensor or constant can be passed directly into the ImplicitBroadcast op).
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
            src = std::make_shared<ngraph::op::Reshape>(src, reshape_order, reshape_shape);
        }

        auto implicit_broadcast =
            std::make_shared<plaidml::op::ImplicitBroadcast>(src, broadcast->get_shape());

        // N.B. We don't use replace_node() here, since it's important to only replace the broadcast
        // with an implicit broadcast when the consuming operation is an elementwise operation,
        // since PlaidML contractions don't provide implicit broadcast semantics.
        bool result = false;
        for (size_t i = 0; i < broadcast->get_output_size(); ++i)
        {
            for (auto& input : broadcast->output(i).get_target_inputs())
            {
                Node* node = input.get_node();
                if (dynamic_cast<ngraph::op::util::UnaryElementwiseArithmetic*>(node) ||
                    dynamic_cast<ngraph::op::util::BinaryElementwiseArithmetic*>(node))
                {
                    input.replace_source_output(implicit_broadcast->output(i));
                    result = true;
                }
            }
        }

        NGRAPH_CHECK(result,
                     "Expected at least one elementwise consumer in the PlaidML implicit broadcast "
                     "rewrite graph pass");
        return result;
    };
    add_matcher(std::make_shared<pattern::Matcher>(target_op), callback);
}
