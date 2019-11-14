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
#include "ngraph/op/atan2.hpp"
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
#include "ngraph/op/one_hot.hpp"
#include "ngraph/op/power.hpp"
#include "ngraph/op/product.hpp"
#include "ngraph/op/relu.hpp"
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

using namespace std;
using namespace ngraph;

#define TI(x) type_index(typeid(x))

static bool cse_constant(shared_ptr<Node> a, shared_ptr<Node> b)
{
    NGRAPH_DEBUG << "In cse_constant for " << a->get_name() << " and " << b->get_name();

    if (a->get_shape() != b->get_shape() || a->get_element_type() != b->get_element_type())
    {
        return false;
    }

    auto ca = static_pointer_cast<op::Constant>(a);
    auto cb = static_pointer_cast<op::Constant>(b);

    size_t size = shape_size(a->get_shape()) * a->get_element_type().size();

    return !memcmp(ca->get_data_ptr(), cb->get_data_ptr(), size);
}

static bool cse_reshape(shared_ptr<Node> a, shared_ptr<Node> b)
{
    NGRAPH_DEBUG << "In cse_reshape for " << a->get_name() << " and " << b->get_name();

    auto reshape_a = static_pointer_cast<ngraph::op::Reshape>(a);
    auto reshape_b = static_pointer_cast<ngraph::op::Reshape>(b);

    return (a->input(0).get_source_output() == b->input(0).get_source_output()) &&
           (reshape_a->get_input_order() == reshape_b->get_input_order()) &&
           (reshape_a->get_output_shape() == reshape_b->get_output_shape());
}
static bool cse_broadcast(shared_ptr<Node> a, shared_ptr<Node> b)
{
    NGRAPH_DEBUG << "In cse_broadcast for " << a->get_name() << " and " << b->get_name();

    auto broadcast_a = static_pointer_cast<ngraph::op::Broadcast>(a);
    auto broadcast_b = static_pointer_cast<ngraph::op::Broadcast>(b);

    return (a->input(0).get_source_output() == b->input(0).get_source_output()) &&
           (broadcast_a->get_broadcast_axes() == broadcast_b->get_broadcast_axes()) &&
           (broadcast_a->get_broadcast_shape() == broadcast_b->get_broadcast_shape());
}
static bool cse_unarywise(shared_ptr<Node> a, shared_ptr<Node> b)
{
    NGRAPH_DEBUG << "In cse_unarywise for " << a->get_name() << " and " << b->get_name();

    return a->input(0).get_source_output() == b->input(0).get_source_output();
}

static bool cse_binarywise(shared_ptr<Node> a, shared_ptr<Node> b)
{
    NGRAPH_DEBUG << "In cse_binary for " << a->get_name() << " and " << b->get_name();

    return (a->input(0).get_source_output() == b->input(0).get_source_output() &&
            a->input(1).get_source_output() == b->input(1).get_source_output()) ||
           (a->input(1).get_source_output() == b->input(0).get_source_output() &&
            a->input(0).get_source_output() == b->input(1).get_source_output());
}

static bool cse_reduction(shared_ptr<Node> a, shared_ptr<Node> b)
{
    NGRAPH_DEBUG << "In cse_reduction for " << a->get_name() << " and " << b->get_name();

    auto ar_a = static_pointer_cast<op::util::ArithmeticReduction>(a);
    auto ar_b = static_pointer_cast<op::util::ArithmeticReduction>(b);

    return ar_a->input(0).get_source_output() == ar_b->input(0).get_source_output() &&
           ar_a->get_reduction_axes() == ar_b->get_reduction_axes();
}

static bool cse_one_hot(shared_ptr<Node> a, shared_ptr<Node> b)
{
    NGRAPH_DEBUG << "In cse_one_hot for " << a->get_name() << " and " << b->get_name();

    auto one_hot_a = static_pointer_cast<ngraph::op::OneHot>(a);
    auto one_hot_b = static_pointer_cast<ngraph::op::OneHot>(b);

    return (a->input(0).get_source_output() == b->input(0).get_source_output()) &&
           (one_hot_a->get_one_hot_axis() == one_hot_b->get_one_hot_axis()) &&
           (a->get_shape() == b->get_shape());
}
static unordered_map<type_index, function<bool(shared_ptr<Node>, shared_ptr<Node>)>>
    initialize_ops_to_cse_handlers()
{
    return unordered_map<type_index, function<bool(shared_ptr<Node>, shared_ptr<Node>)>>(
        {{TI(op::Abs), cse_unarywise},
         {TI(op::Acos), cse_unarywise},
         {TI(op::Asin), cse_unarywise},
         {TI(op::Atan), cse_unarywise},
         {TI(op::Atan2), cse_binarywise},
         {TI(op::Ceiling), cse_unarywise},
         {TI(op::Constant), cse_constant},
         {TI(op::Cos), cse_unarywise},
         {TI(op::Cosh), cse_unarywise},
         {TI(op::Exp), cse_unarywise},
         {TI(op::Floor), cse_unarywise},
         {TI(op::Log), cse_unarywise},
         {TI(op::Negative), cse_unarywise},
         {TI(op::OneHot), cse_one_hot},
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
         {TI(op::Subtract), cse_binarywise},
         {TI(op::Sum), cse_reduction},
         {TI(op::Product), cse_reduction},
         {TI(op::Reshape), cse_reshape},
         {TI(op::Broadcast), cse_broadcast}});
}

static unordered_map<type_index, function<bool(shared_ptr<Node>, shared_ptr<Node>)>>
    ops_to_cse_handlers = initialize_ops_to_cse_handlers();

class NodeKey
{
public:
    NodeKey(shared_ptr<Node> n,
            unordered_map<type_index, function<bool(shared_ptr<Node>, shared_ptr<Node>)>>&
                backend_handlers)
        : m_node(n)
        , m_backend_handlers(backend_handlers)
    {
    }

    shared_ptr<Node> get_node() const { return m_node; }
    bool operator==(const NodeKey& other) const
    {
        Node& p_this = *m_node.get();
        Node& p_other = *other.get_node().get();

        if (TI(p_this) != TI(p_other))
        {
            return false;
        }

        {
            auto eh = ops_to_cse_handlers.find(TI(p_this));
            if (eh != ops_to_cse_handlers.end())
            {
                return eh->second(m_node, other.get_node());
            }
        }

        {
            auto eh = m_backend_handlers.find(TI(p_this));
            if (eh != m_backend_handlers.end())
            {
                return eh->second(m_node, other.get_node());
            }
        }

        return false;
    }

private:
    shared_ptr<Node> m_node;
    unordered_map<type_index, function<bool(shared_ptr<Node>, shared_ptr<Node>)>>&
        m_backend_handlers;
};

namespace std
{
    template <>
    struct hash<NodeKey>
    {
        size_t operator()(const NodeKey& k) const
        {
            Node& p_this = *k.get_node().get();
            auto ti = TI(p_this);

            hash<type_index> type_hash_compute{};
            auto type_hash = type_hash_compute(ti);

            vector<size_t> arg_ids;

            arg_ids.push_back(type_hash);

            std::vector<Output<Node>> cargs;
            for (auto input : k.get_node()->inputs())
            {
                cargs.push_back(input.get_source_output());
            }

            // TODO: Do we need another map, so we could
            // specify how to compute hash for each op?
            if (p_this.is_commutative())
            {
                sort(begin(cargs), end(cargs));
            }

            for (auto arg : cargs)
            {
                arg_ids.push_back(arg.get_node_shared_ptr()->get_instance_id());
                arg_ids.push_back(arg.get_index());
            }

            auto hashc = ngraph::hash_combine(arg_ids);
            return hashc;
        }
    };
}

bool ngraph::pass::CommonSubexpressionElimination::run_on_function(shared_ptr<ngraph::Function> f)
{
    bool replaced = false;
    unordered_map<NodeKey, shared_ptr<Node>> expressions{};

    for (auto n : f->get_ordered_ops())
    {
        if (n->is_output() || n->is_parameter())
        {
            continue;
        }

        NodeKey n_key(n, m_backend_cse_handlers);
        if (expressions.count(n_key))
        {
            ngraph::replace_node(n, expressions.at(n_key));
            replaced = true;
        }
        else
        {
            expressions.insert(make_pair(n_key, n));
        }
    }

    return replaced;
}
