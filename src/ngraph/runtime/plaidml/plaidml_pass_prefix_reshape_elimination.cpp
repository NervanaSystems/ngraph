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

#include "ngraph/runtime/plaidml/plaidml_pass_prefix_reshape_elimination.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/util/binary_elementwise_arithmetic.hpp"
#include "ngraph/op/util/unary_elementwise_arithmetic.hpp"
#include "ngraph/pattern/matcher.hpp"
#include "ngraph/pattern/op/any.hpp"
#include "ngraph/pattern/op/any_of.hpp"
#include "ngraph/pattern/op/label.hpp"
#include "ngraph/runtime/plaidml/plaidml_ops_implicit_broadcast.hpp"

using namespace std;
using namespace ngraph;

runtime::plaidml::pass::PrefixReshapeElimination::PrefixReshapeElimination()
{
    auto src_op = make_shared<pattern::op::Label>(
        element::i8, Shape{}, [](shared_ptr<Node>) { return true; });
    auto reshape_op = make_shared<pattern::op::Any>(
        element::i8,
        Shape{},
        [](shared_ptr<Node> node) {
            ngraph::op::Reshape* reshape = dynamic_cast<ngraph::op::Reshape*>(node.get());
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
            return pattern::has_class<ngraph::op::util::UnaryElementwiseArithmetic>()(node) ||
                   pattern::has_class<ngraph::op::util::BinaryElementwiseArithmetic>()(node);
        },
        NodeVector{reshape_op});

    auto callback = [](pattern::Matcher& m) {
        auto src = m.get_matched_nodes().at(2);
        auto prefix_reshape =
            std::static_pointer_cast<ngraph::op::Reshape>(m.get_matched_nodes().at(1));
        auto implicit_broadcast =
            std::make_shared<op::ImplicitBroadcast>(src, prefix_reshape->get_shape());

        // N.B. We don't use replace_node() here, since it's important to only replace the prefix reshape with
        // an implicit broadcast when the consuming operation is an elementwise operation, since PlaidML
        // contractions don't provide implicit broadcast semantics.
        bool result = false;
        for (size_t i = 0; i < prefix_reshape->get_output_size(); ++i)
        {
            for (auto& input : prefix_reshape->output(i).get_target_inputs())
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
    add_matcher(make_shared<pattern::Matcher>(target_op, "PrefixReshapeElimination"),
                callback,
                ngraph::pass::PassProperty::REQUIRE_STATIC_SHAPE);
}
