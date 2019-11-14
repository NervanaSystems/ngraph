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

#include <typeindex>

#include "ngraph/graph_util.hpp"
#include "ngraph/op/and.hpp"
#include "ngraph/op/equal.hpp"
#include "ngraph/op/greater.hpp"
#include "ngraph/op/greater_eq.hpp"
#include "ngraph/op/less.hpp"
#include "ngraph/op/less_eq.hpp"
#include "ngraph/op/not.hpp"
#include "ngraph/op/not_equal.hpp"
#include "ngraph/op/or.hpp"
#include "ngraph/op/passthrough.hpp"
#include "ngraph/op/xor.hpp"
#include "ngraph/pattern/matcher.hpp"
#include "ngraph/pattern/op/any.hpp"
#include "ngraph/pattern/op/any_of.hpp"
#include "ngraph/pattern/op/label.hpp"
#include "ngraph/runtime/plaidml/plaidml_pass_explicit_logicals.hpp"

void ngraph::runtime::plaidml::pass::ExplicitLogicals::construct_logical_to_data()
{
    auto producer_op =
        std::make_shared<pattern::op::Label>(element::i8, Shape{}, [](std::shared_ptr<Node> node) {
            static const std::unordered_set<std::type_index> logical_producers{
                std::type_index{typeid(ngraph::op::And)},
                std::type_index{typeid(ngraph::op::Equal)},
                std::type_index{typeid(ngraph::op::Greater)},
                std::type_index{typeid(ngraph::op::GreaterEq)},
                std::type_index{typeid(ngraph::op::Less)},
                std::type_index{typeid(ngraph::op::LessEq)},
                std::type_index{typeid(ngraph::op::Not)},
                std::type_index{typeid(ngraph::op::NotEqual)},
                std::type_index{typeid(ngraph::op::Or)},
                std::type_index{typeid(ngraph::op::Xor)}};

            const ngraph::Node* node_ptr = node.get();

            // True iff this node produces a logical output.
            return logical_producers.count(std::type_index(typeid(*node_ptr))) != 0;
        });
    auto data_consumer_op = std::make_shared<pattern::op::Any>(
        element::i8,
        Shape{},
        [](std::shared_ptr<Node> node) {
            static const std::unordered_set<std::type_index> logical_consumers{
                std::type_index{typeid(ngraph::op::And)},
                std::type_index{typeid(ngraph::op::Equal)},
                std::type_index{typeid(ngraph::op::Not)},
                std::type_index{typeid(ngraph::op::NotEqual)},
                std::type_index{typeid(ngraph::op::Or)},
                std::type_index{typeid(ngraph::op::Xor)}};

            const ngraph::Node* node_ptr = node.get();

            // True iff this node should not be presented with a logical output.
            return logical_consumers.count(std::type_index(typeid(*node_ptr))) == 0;
        },
        NodeVector{producer_op});

    auto callback = [producer_op](pattern::Matcher& m) {
        auto consumer = m.get_match_root();
        auto producer = m.get_pattern_map()[producer_op];
        NGRAPH_DEBUG << "Adding conversion for " << producer->description() << " -> "
                     << consumer->description();
        ngraph::insert_new_node_between(
            producer,
            consumer,
            std::make_shared<op::Passthrough>(
                "ConvertLogicalToData",
                "Tile",
                "function (I) -> (O) { O = as_int(I ? 1 : 0, 8);}",
                NodeVector{producer},
                std::vector<std::tuple<element::Type, PartialShape>>{{std::make_tuple(
                    element::boolean, PartialShape{producer->get_output_shape(0)})}}));
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(data_consumer_op);
    add_matcher(m, callback);
}
