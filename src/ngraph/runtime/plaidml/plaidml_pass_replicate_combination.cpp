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

#include "ngraph/runtime/plaidml/plaidml_pass_replicate_combination.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/pattern/matcher.hpp"
#include "ngraph/pattern/op/label.hpp"
#include "ngraph/runtime/plaidml/plaidml_ops_replicate.hpp"

ngraph::runtime::plaidml::pass::ReplicateCombination::ReplicateCombination()
{
    auto upper_replicate_op =
        std::make_shared<pattern::op::Label>(element::i8, Shape{}, [](std::shared_ptr<Node> node) {
            return pattern::has_class<plaidml::op::Replicate>()(node);
        });

    auto lower_replicate_op = std::make_shared<pattern::op::Any>(
        element::i8,
        Shape{},
        [](std::shared_ptr<Node> node) {
            return pattern::has_class<plaidml::op::Replicate>()(node);
        },
        NodeVector{upper_replicate_op});

    auto callback = [](pattern::Matcher& m) {
        auto nodes = m.get_matched_nodes();
        auto lower = std::static_pointer_cast<plaidml::op::Replicate>(nodes.at(0));
        auto upper = std::static_pointer_cast<plaidml::op::Replicate>(nodes.at(1));
        std::vector<size_t> axes = lower->get_replication_axes();
        const std::vector<size_t>& upper_axes = upper->get_replication_axes();
        auto uit = upper_axes.begin();
        for (auto ait = axes.begin(); ait != axes.end(); ++ait, ++uit)
        {
            *ait *= *uit;
        }

        replace_node(
            lower,
            std::make_shared<plaidml::op::Replicate>(upper->get_argument(0), std::move(axes)));

        return true;
    };

    add_matcher(std::make_shared<pattern::Matcher>(lower_replicate_op), callback);
}
