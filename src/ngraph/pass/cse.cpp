/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
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

#include <memory>
#include <set>
#include <typeinfo>
#include <unordered_map>

#include "cse.hpp"
#include "ngraph/axis_vector.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/op/abs.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/product.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/pattern/matcher.hpp"

using namespace ngraph;

#define TI(x) std::type_index(typeid(x))

static bool cse_unarywise(std::shared_ptr<Node> a, std::shared_ptr<Node> b)
{
    NGRAPH_DEBUG << "In cse_unarywise for " << a->get_name() << " and " << b->get_name();

    return a->get_argument(0) == b->get_argument(0);
}

static bool cse_binarywise(std::shared_ptr<Node> a, std::shared_ptr<Node> b)
{
    NGRAPH_DEBUG << "In cse_binary for " << a->get_name() << " and " << b->get_name();

    return (a->get_argument(0) == b->get_argument(0) && a->get_argument(1) == b->get_argument(1)) ||
           (a->get_argument(1) == b->get_argument(0) && a->get_argument(0) == b->get_argument(1));
}

static std::unordered_map<std::type_index,
                          std::function<bool(std::shared_ptr<Node>, std::shared_ptr<Node>)>>
    initialize_ops_to_cse_handlers()
{
    return std::unordered_map<std::type_index,
                              std::function<bool(std::shared_ptr<Node>, std::shared_ptr<Node>)>>({
        {TI(op::Abs), cse_unarywise}, {TI(op::Add), cse_binarywise},
    });
}

static std::unordered_map<std::type_index,
                          std::function<bool(std::shared_ptr<Node>, std::shared_ptr<Node>)>>
    ops_to_cse_handlers = initialize_ops_to_cse_handlers();

class NodeKey
{
public:
    NodeKey(std::shared_ptr<Node> n)
        : m_node(n)
    {
    }

    std::shared_ptr<Node> get_node() const { return m_node; }
    bool operator==(const NodeKey& other) const
    {
        Node& p_this = *m_node.get();
        Node& p_other = *other.get_node().get();

        if (TI(p_this) != TI(p_other))
        {
            return false;
        }

        auto eh = ops_to_cse_handlers.find(TI(p_this));
        if (eh == ops_to_cse_handlers.end())
        {
            return false;
        }

        return eh->second(m_node, other.get_node());
    }

private:
    std::shared_ptr<Node> m_node;
};

namespace std
{
    template <>
    struct hash<NodeKey>
    {
        std::size_t operator()(const NodeKey& k) const
        {
            Node& p_this = *k.get_node().get();
            auto ti = TI(p_this);

            std::hash<std::type_index> type_hash_compute{};
            auto type_hash = type_hash_compute(ti);

            std::vector<size_t> arg_ids;

            arg_ids.push_back(type_hash);

            auto cargs = k.get_node()->get_arguments();

            //TODO: Do we need another map, so we could
            //specify how to compute hash for each op?
            if (p_this.is_commutative())
            {
                std::sort(begin(cargs), end(cargs));
            }

            for (auto arg : cargs)
            {
                arg_ids.push_back(arg->get_instance_id());
            }

            auto hashc = ngraph::hash_combine(arg_ids);
            return hashc;
        }
    };
}

bool ngraph::pass::CommonSubexpressionElimination::run_on_function(
    std::shared_ptr<ngraph::Function> f)
{
    bool replaced = false;
    std::unordered_map<NodeKey, std::shared_ptr<Node>> expressions{};

    for (auto n : f->get_ordered_ops())
    {
        if (n->is_output() || n->is_parameter() ||
            n->is_constant() /*we could CSE constants as well*/)
        {
            continue;
        }

        NodeKey n_key{n};
        if (expressions.count(n_key))
        {
            ngraph::replace_node(n, expressions.at(n_key));
            replaced = true;
        }
        else
        {
            expressions.insert(std::make_pair(n_key, n));
        }
    }

    return replaced;
}
