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

#include "ngraph/runtime/plaidml/plaidml_pass_replicate_elision.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/op/util/binary_elementwise_arithmetic.hpp"
#include "ngraph/op/util/unary_elementwise_arithmetic.hpp"
#include "ngraph/pattern/matcher.hpp"
#include "ngraph/pattern/op/any_of.hpp"
#include "ngraph/pattern/op/label.hpp"
#include "ngraph/pattern/op/skip.hpp"
#include "ngraph/runtime/plaidml/plaidml_ops_replicate.hpp"

ngraph::runtime::plaidml::pass::ReplicateElision::ReplicateElision()
{
    auto replicate_op =
        std::make_shared<pattern::op::Label>(element::i8, Shape{}, [](std::shared_ptr<Node> node) {
            return pattern::has_class<plaidml::op::Replicate>()(node);
        });
    auto skip_op =
        std::make_shared<pattern::op::Skip>(replicate_op, [](std::shared_ptr<Node> node) {
            return pattern::has_class<plaidml::op::Replicate>()(node);
        });
    auto target_op =
        std::make_shared<pattern::op::AnyOf>(element::i8,
                                             Shape{},
                                             [](std::shared_ptr<Node> node) {
                                                 return node->is_unary_elementwise_arithmetic() ||
                                                        node->is_binary_elementwise_arithmetic();
                                             },
                                             NodeVector{skip_op});

    auto callback = [](pattern::Matcher& m) {
        bool replaced_any = false;
        auto nodes = m.get_matched_nodes();
        std::size_t dim_limit = nodes.at(1)->get_shape().size();
        std::vector<bool> broadcast_axes(dim_limit, true);

        for (auto nit = nodes.begin() + 1; nit != nodes.end(); ++nit)
        {
            auto replicate = std::static_pointer_cast<plaidml::op::Replicate>(*nit);
            const auto& replicate_axes = replicate->get_replication_axes();
            bool elidable = true;
            for (std::size_t idx = 0; idx < dim_limit; ++idx)
            {
                if (replicate_axes.at(idx) == 1)
                {
                    continue;
                }
                if (!broadcast_axes.at(idx))
                {
                    elidable = false;
                    continue;
                }
                if (replicate->get_input_shape(0).at(idx) != 1)
                {
                    broadcast_axes.at(idx) = false;
                    elidable = false;
                }
            }
            if (elidable)
            {
                replaced_any = true;
                replace_node(replicate, replicate->get_argument(0));
            }
        }

        return replaced_any;
    };

    add_matcher(std::make_shared<pattern::Matcher>(target_op), callback);
}
