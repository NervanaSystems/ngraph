/*******************************************************************************
* Copyright 2018 Intel Corporation
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
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <map>
#include <stack>

#include "ngraph/ops/dot.hpp"
#include "ngraph/ops/reshape.hpp"
#include "ngraph/ops/slice.hpp"
#include "ngraph/ops/parameter.hpp"
#include "cpu_rnn_mat_fusion.hpp"

using namespace ngraph;

static NodeVector get_users(Node& node)
{
    NodeVector result;

    for (size_t i = 0; i < node.get_output_size(); ++i)
    {
        for (auto input : node.get_output_inputs(i))
        {
            result.push_back(input->get_node());
        }
    }

    return result;
}

#define TI(x) std::type_index(typeid(x))

struct DotOps
{
    std::shared_ptr<Node> dot;
    NodeVector path;
};

void FindValidPathDFS(std::shared_ptr<Node> n, NodeVector path, std::vector<NodeVector>& path_bundle)
{
    Node& node = *n;
    std::cout << "visiting: " << node.get_friendly_name() << std::endl;
    if (TI(node) == TI(ngraph::op::Dot)) {
        path.push_back(n);
        path_bundle.push_back(path);
        return;
    }
    if (TI(node) == TI(ngraph::op::Slice) || TI(node) == TI(ngraph::op::Reshape)) {
        path.push_back(n);
        const auto outputs = get_users(node);
        for (const auto& out : outputs) {
            FindValidPathDFS(out, path, path_bundle);
        }
    }
}

struct OrderedParams
{
public:
    OrderedParams(const std::shared_ptr<Node>& n1, const std::shared_ptr<Node>& n2) {
        if (n1 < n2) {
            m_params.first = n1;
            m_params.second = n2;
        }
        else {
            m_params.first = n2;
            m_params.second = n1;
        }
    }
    std::shared_ptr<Node> first() const { return m_params.first; }
    std::shared_ptr<Node> second() const { return m_params.second; }
private:
    friend bool operator< (const OrderedParams& n1, const OrderedParams& n2);
    std::pair<std::shared_ptr<Node>, std::shared_ptr<Node>> m_params;
};

bool operator< (const OrderedParams& n1, const OrderedParams& n2)
{
    return n1.m_params < n2.m_params;
}

bool ngraph::runtime::cpu::pass::CPURnnMatFusion::run_on_function(
    std::shared_ptr<ngraph::Function> function)
{
    bool clobbered = false;
    std::cout << "Slice: " << TI(ngraph::op::Slice).hash_code() << std::endl;
    std::cout << "Reshape: " << TI(ngraph::op::Reshape).hash_code() << std::endl;
    std::cout << "Dot: " << TI(ngraph::op::Dot).hash_code() << std::endl;

    std::list<std::shared_ptr<Node>> param_nodes;
    for (auto& n : function->get_ordered_ops())
    {
        // Work around a warning [-Wpotentially-evaluated-expression]
        Node& node = *n;
        std::cout << "instance id: " << node.get_instance_id() << std::endl;
        std::string type = "other";
        if (TI(node) == TI(ngraph::op::Parameter)) {
            param_nodes.push_back(n);
        }
        if (TI(node) == TI(ngraph::op::Slice)) {
            type = "Slice";
        }
        if (TI(node) == TI(ngraph::op::Reshape)) {
            type = "Reshape";
        }
        if (TI(node) == TI(ngraph::op::Dot)) {
            type = "Dot";
        }
        std::cout << "node (" << type << "): " << node.get_friendly_name() << std::endl;
        for (const auto& in : node.get_input_ops()) {
            std::cout << "    in:  " << in->get_friendly_name() << std::endl;
        }
        auto outputs = get_users(node);
        for (const auto& out : outputs) {
            std::cout << "    out: " << out->get_friendly_name() << std::endl;
        }
    }
    std::cout << "find all dots" << std::endl;
    // iterate all parameters and find path to dot op
    std::map<std::shared_ptr<Node>, std::vector<NodeVector>> dot_ops;
    std::vector<NodeVector> path_bundle;
    for (auto& p : param_nodes) {
        std::cout << "param: " << p->get_friendly_name() << std::endl;
        const auto outputs = get_users(*p);
        for (const auto& out : outputs) {
            std::cout << "    out: " << out->get_friendly_name() << std::endl;
        }
        NodeVector path;
        path.push_back(p);
        for (const auto& out : outputs) {
            FindValidPathDFS(out, path, path_bundle);
        }
    }
    for (const auto& nv : path_bundle) {
        dot_ops[nv.back()].push_back(nv);
    }
    for (auto& d : dot_ops) {
        std::cout << "dot: " << d.first->get_friendly_name() << std::endl;
        for (auto& vn : d.second) {
            std::cout << "  path: " << std::endl;
            for (auto& n : vn) {
                std::cout << "   node: " << n->get_friendly_name() << std::endl;
            }
        }
    }
    std::cout << "remove dots with single path" << std::endl;
    // remove dot ops with single path
    for (auto dot_it = dot_ops.cbegin(); dot_it != dot_ops.cend();) {
        if (dot_it->second.size() < 2) {
            dot_it = dot_ops.erase(dot_it);
        }
        else {
            ++dot_it;
        }
    }
    for (auto& d : dot_ops) {
        std::cout << "dot: " << d.first->get_friendly_name() << std::endl;
        for (auto& vn : d.second) {
            std::cout << "  path: " << std::endl;
            for (auto& n : vn) {
                std::cout << "   node: " << n->get_friendly_name() << std::endl;
            }
        }
    }

    std::cout << "find all pairs" << std::endl;
    std::map<OrderedParams, NodeVector> params;
    for (auto& d : dot_ops) {
        //assert(d.second.size() == 2);
        OrderedParams p(d.second[0].front(), d.second[1].front());
        params[p].push_back(d.first);
    }
    for (auto& p : params) {
        std::cout << "pair: [" << p.first.first()->get_friendly_name() << ", "
                  << p.first.second()->get_friendly_name() << "]" << std::endl;
        for (auto& op : p.second) {
            std::cout << "op: " << op->get_friendly_name() << std::endl;
        }
    }
    // remove pairs with single op
    std::cout << "removed single op pairs" << std::endl;
    for (auto it = params.begin(); it != params.end();) {
        if (it->second.size() < 2) {
            it = params.erase(it);
        }
        else {
            ++it;
        }
    }
    for (auto& p : params) {
        std::cout << "pair: [" << p.first.first()->get_friendly_name() << ", "
                  << p.first.second()->get_friendly_name() << "]" << std::endl;
        for (auto& op : p.second) {
            std::cout << "op: " << op->get_friendly_name() << std::endl;
        }
    }
    // TODO: check consistency
    // check dot op matrix order
    // check shape size

    // create new ops
    for (auto& p : params) {

        auto dot_op = p.second[0];
        auto p1 = dot_op->get_input_op(0);//p.first.first();
        auto p2 = dot_op->get_input_op(1);//p.first.second();

        std::cout << "replacing " << p1->get_friendly_name() << " with " << p1->get_input_op(0)->get_friendly_name() << std::endl;
        std::cout << "replacing " << p2->get_friendly_name() << " with " << p2->get_input_op(0)->get_friendly_name() << std::endl;
        function->replace_node(p1, p1->get_input_op(0));
        function->replace_node(p2, p2->get_input_op(0));
//        for (auto v : p1->get_shape()) {
//            std::cout << v << " ";
//        }
//        std::cout << std::endl;
//        for (auto v : p2->get_shape()) {
//            std::cout << v << " ";
//        }
//        std::cout << std::endl;
        for (auto& n : function->get_ordered_ops())
        {
            // Work around a warning [-Wpotentially-evaluated-expression]
            Node& node = *n;
            std::cout << "instance id: " << node.get_instance_id() << std::endl;
            std::string type = "other";
            if (TI(node) == TI(ngraph::op::Parameter)) {
                param_nodes.push_back(n);
            }
            if (TI(node) == TI(ngraph::op::Slice)) {
                type = "Slice";
            }
            if (TI(node) == TI(ngraph::op::Reshape)) {
                type = "Reshape";
            }
            if (TI(node) == TI(ngraph::op::Dot)) {
                type = "Dot";
            }
            std::cout << "node (" << type << "): " << node.get_friendly_name() << std::endl;
            for (const auto& in : node.get_input_ops()) {
                std::cout << "    in:  " << in->get_friendly_name() << std::endl;
            }
            auto outputs = get_users(node);
            for (const auto& out : outputs) {
                std::cout << "    out: " << out->get_friendly_name() << std::endl;
            }
        }

//        std::shared_ptr<op::Reshape> n1_reshape = std::make_shared(op::Reshape(A, AxisVector{1, 0}, shape_r)));
    }
    return clobbered;
}
