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
#include "ngraph/op/abs.hpp"
#include "ngraph/op/acos.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/asin.hpp"
#include "ngraph/op/atan.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/ceiling.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/cos.hpp"
#include "ngraph/op/cosh.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/exp.hpp"
#include "ngraph/op/floor.hpp"
#include "ngraph/op/log.hpp"
#include "ngraph/op/maximum.hpp"
#include "ngraph/op/minimum.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/negative.hpp"
#include "ngraph/op/power.hpp"
#include "ngraph/op/product.hpp"
#include "ngraph/op/relu.hpp"
#include "ngraph/op/remainder.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/sigmoid.hpp"
#include "ngraph/op/sign.hpp"
#include "ngraph/op/sin.hpp"
#include "ngraph/op/sinh.hpp"
#include "ngraph/op/softmax.hpp"
#include "ngraph/op/sqrt.hpp"
#include "ngraph/op/subtract.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/op/tan.hpp"
#include "ngraph/op/tanh.hpp"
#include "ngraph/pattern/matcher.hpp"

using namespace ngraph;

#define TI(x) std::type_index(typeid(x))

static bool cse_constant(std::shared_ptr<Node> a, std::shared_ptr<Node> b)
{
    NGRAPH_DEBUG << "In cse_constant for " << a->get_name() << " and " << b->get_name();

    if (a->get_shape() != b->get_shape() || a->get_element_type() != b->get_element_type())
    {
        return false;
    }

    auto ca = std::dynamic_pointer_cast<op::Constant>(a);
    auto cb = std::dynamic_pointer_cast<op::Constant>(b);

    size_t size = shape_size(a->get_shape()) * a->get_element_type().size();

    return !memcmp(ca->get_data_ptr(), cb->get_data_ptr(), size);
}

static bool cse_reshape(std::shared_ptr<Node> a, std::shared_ptr<Node> b)
{
    NGRAPH_DEBUG << "In cse_reshape for " << a->get_name() << " and " << b->get_name();

    auto reshape_a = std::dynamic_pointer_cast<ngraph::op::Reshape>(a);
    auto reshape_b = std::dynamic_pointer_cast<ngraph::op::Reshape>(b);

    return (a->get_argument(0) == b->get_argument(0)) &&
           (reshape_a->get_input_order() == reshape_b->get_input_order()) &&
           (reshape_a->get_output_shape() == reshape_b->get_output_shape());
}
static bool cse_broadcast(std::shared_ptr<Node> a, std::shared_ptr<Node> b)
{
    NGRAPH_DEBUG << "In cse_broadcast for " << a->get_name() << " and " << b->get_name();

    auto broadcast_a = std::dynamic_pointer_cast<ngraph::op::Broadcast>(a);
    auto broadcast_b = std::dynamic_pointer_cast<ngraph::op::Broadcast>(b);

    return (a->get_argument(0) == b->get_argument(0)) &&
           (broadcast_a->get_broadcast_axes() == broadcast_b->get_broadcast_axes()) &&
           (broadcast_a->get_broadcast_shape() == broadcast_b->get_broadcast_shape());
}
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

static bool cse_reduction(std::shared_ptr<Node> a, std::shared_ptr<Node> b)
{
    NGRAPH_DEBUG << "In cse_reduction for " << a->get_name() << " and " << b->get_name();

    auto ar_a = std::dynamic_pointer_cast<op::util::ArithmeticReduction>(a);
    auto ar_b = std::dynamic_pointer_cast<op::util::ArithmeticReduction>(b);

    return ar_a->get_argument(0) == ar_b->get_argument(0) &&
           ar_a->get_reduction_axes() == ar_b->get_reduction_axes();
}

static std::unordered_map<std::type_index,
                          std::function<bool(std::shared_ptr<Node>, std::shared_ptr<Node>)>>
    initialize_ops_to_cse_handlers()
{
    return std::unordered_map<std::type_index,
                              std::function<bool(std::shared_ptr<Node>, std::shared_ptr<Node>)>>(
        {{TI(op::Abs), cse_unarywise},
         {TI(op::Acos), cse_unarywise},
         {TI(op::Asin), cse_unarywise},
         {TI(op::Atan), cse_unarywise},
         {TI(op::Ceiling), cse_unarywise},
         {TI(op::Constant), cse_constant},
         {TI(op::Cos), cse_unarywise},
         {TI(op::Cosh), cse_unarywise},
         {TI(op::Exp), cse_unarywise},
         {TI(op::Floor), cse_unarywise},
         {TI(op::Log), cse_unarywise},
         {TI(op::Negative), cse_unarywise},
         {TI(op::Relu), cse_unarywise},
         {TI(op::Sigmoid), cse_unarywise},
         {TI(op::Sign), cse_unarywise},
         {TI(op::Sin), cse_unarywise},
         {TI(op::Sinh), cse_unarywise},
         //{TI(op::Softmax), cse_unarywise},
         {TI(op::Sqrt), cse_unarywise},
         {TI(op::Tan), cse_unarywise},
         {TI(op::Tanh), cse_unarywise},
         {TI(op::Add), cse_binarywise},
         {TI(op::Divide), cse_binarywise},
         {TI(op::Maximum), cse_binarywise},
         {TI(op::Minimum), cse_binarywise},
         {TI(op::Multiply), cse_binarywise},
         {TI(op::Power), cse_binarywise},
         //{TI(op::Remainder), cse_binarywise},
         {TI(op::Subtract), cse_binarywise},
         {TI(op::Sum), cse_reduction},
         {TI(op::Product), cse_reduction},
         {TI(op::Reshape), cse_reshape},
         {TI(op::Broadcast), cse_broadcast}});
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

            // TODO: Do we need another map, so we could
            // specify how to compute hash for each op?
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
        if (n->is_output() || n->is_parameter())
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
