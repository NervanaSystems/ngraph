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
#include <numeric>

#include "ngraph/ops/dot.hpp"
#include "ngraph/ops/reshape.hpp"
#include "ngraph/ops/slice.hpp"
#include "ngraph/ops/parameter.hpp"
#include "cpu_rnn_mat_fusion.hpp"

using namespace ngraph;

static NodeVector get_users(const Node& node)
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


struct NodeSegment : public NodeVector
{
    enum Type {
      DATA = 0,
      WEIGHTS,
      UNDEFINED
    };
    Type type {UNDEFINED};
};
typedef std::pair<NodeSegment::Type, std::vector<std::type_index>> NodeTypeSequence;
typedef std::list<NodeTypeSequence> NodeTypeSequenceSet;

// precondition: all valid sequences must be unique
void FindValidPathDFS(const std::shared_ptr<Node>& n,
                      NodeSegment segment,
                      std::vector<NodeSegment>& segment_bundle,
                      NodeTypeSequenceSet valid_sequence_list,
                      int depth)
{
    const Node& node = *n;
//    std::cout << "visiting: " << node.get_friendly_name() << std::endl;

    // base case, we have one valid segment left, and depth is exeeding the valid seqence length
    // we found a match
    if (valid_sequence_list.size() == 1 && depth >= valid_sequence_list.front().second.size()) {
        segment.type = valid_sequence_list.front().first;
        segment_bundle.push_back(segment);
//        std::cout << "adding " << node.get_friendly_name() << std::endl;
        return;
    }
    for (auto seq_it = valid_sequence_list.begin(); seq_it != valid_sequence_list.end();) {
        const auto& valid_seq = seq_it->second;
        // skip and remove sequences which are longer or doesn't match current node at depth index
        if (depth >= valid_seq.size() || TI(node) != valid_seq[depth]) {
            seq_it = valid_sequence_list.erase(seq_it);
            continue;
        }
        else {
            ++seq_it;
        }
    }
    // postcondition: valid_sequnce_list.size() > 0 : there's still valid sequence to match, continue recursion
    if (valid_sequence_list.size() > 0) {
        segment.push_back(n);
        const auto outputs = get_users(node);
        for (const auto& out_node : outputs) {
            FindValidPathDFS(out_node, segment, segment_bundle, valid_sequence_list, depth+1);
        }
    }
}

typedef std::pair<std::shared_ptr<Node>, std::shared_ptr<Node>> OrderedParams;
//{
//public:
//    OrderedParams(const std::shared_ptr<Node>& data, const std::shared_ptr<Node>& weights) {
//        m_params.first = data;
//        m_params.second = weights;
//    }
//    std::shared_ptr<Node> first() const { return m_params.first; }
//    std::shared_ptr<Node> second() const { return m_params.second; }
//private:
//    friend bool operator< (const OrderedParams& n1, const OrderedParams& n2);
//    std::pair<std::shared_ptr<Node>, std::shared_ptr<Node>> m_params;
//};

//bool operator<(const OrderedParams& n1, const OrderedParams& n2)
//{
//    return n1.m_params < n2.m_params;
//}

bool ngraph::runtime::cpu::pass::CPURnnMatFusion::run_on_function(
    std::shared_ptr<ngraph::Function> function)
{
    bool clobbered = false;

    NodeTypeSequenceSet valid_sequences {
        {NodeSegment::DATA,
         {TI(ngraph::op::Parameter),
          TI(ngraph::op::Slice),
          TI(ngraph::op::Reshape),
          TI(ngraph::op::Dot)}},
        {NodeSegment::WEIGHTS,
         {TI(ngraph::op::Parameter),
          TI(ngraph::op::Reshape),
          TI(ngraph::op::Dot)}}
    };

#if 0
    std::list<std::shared_ptr<Node>> param_nodes;
    for (auto& n : function->get_ordered_ops())
    {
        Node& node = *n;
        if (TI(node) == TI(ngraph::op::Parameter)) {
            param_nodes.push_back(n);
        }

        // debug
//        std::cout << "instance id: " << node.get_instance_id() << std::endl;
//        std::string type = "other";
//        if (TI(node) == TI(ngraph::op::Slice)) {
//            type = "Slice";
//        }
//        if (TI(node) == TI(ngraph::op::Reshape)) {
//            type = "Reshape";
//        }
//        if (TI(node) == TI(ngraph::op::Dot)) {
//            type = "Dot";
//        }
//        std::cout << "node (" << type << "): " << node.get_friendly_name() << std::endl;
//        for (const auto& in : node.get_input_ops()) {
//            std::cout << "    in:  " << in->get_friendly_name() << std::endl;
//        }
//        auto outputs = get_users(node);
//        for (const auto& out : outputs) {
//            std::cout << "    out: " << out->get_friendly_name() << std::endl;
//        }
    }
#endif

    std::cout << "find all dots" << std::endl;
    // iterate all parameters and find path to dot op
    std::vector<NodeSegment> segment_bundle;
    for (auto& n : function->get_ordered_ops()) {
#if 0
        std::cout << "param: " << p->get_friendly_name() << std::endl;
        const auto outputs = get_users(*p);
        for (const auto& out : outputs) {
            std::cout << "    out: " << out->get_friendly_name() << std::endl;
        }
#endif
        NodeSegment segment;
        FindValidPathDFS(n, segment, segment_bundle, valid_sequences, 0);
    }

    // combined all segments by last operator
    std::map<std::shared_ptr<Node>, std::vector<NodeSegment>> op_seg_map;
    for (const auto& segment : segment_bundle) {
        op_seg_map[segment.back()].push_back(segment);
        std::cout << "segment: \n";
        for (auto& n : segment) {
            std::cout << "  " << n->get_friendly_name() << std::endl;
        }
    }
//    for (auto& op : op_seg_map) {
//        std::cout << "op: " << op.first->get_friendly_name() << std::endl;
//        for (auto& vn : op.second) {
//            std::cout << "  segment (" << static_cast<int>(vn.type) << "): " << std::endl;
//            for (auto& n : vn.nodes) {
//                std::cout << "   node: " << n->get_friendly_name() << std::endl;
//            }
//        }
//    }

    std::cout << "remove single segments" << std::endl;
    // remove dot ops with single path
    for (auto op_it = op_seg_map.cbegin(); op_it != op_seg_map.cend();) {
        if (op_it->second.size() < 2) {
            op_it = op_seg_map.erase(op_it);
        }
        else {
            ++op_it;
        }
    }
    for (auto& op : op_seg_map) {
        std::cout << "op: " << op.first->get_friendly_name() << std::endl;
        for (auto& segment : op.second) {
            std::cout << "  segment (" << static_cast<int>(segment.type) << "): " << std::endl;
            for (auto& n : segment) {
                std::cout << "   node: " << n->get_friendly_name() << std::endl;
            }
        }
    }

    std::cout << "find all param pairs" << std::endl;
    std::map<OrderedParams, NodeVector> param_list;
    for (auto& op_seg : op_seg_map) {
        std::vector<NodeSegment>& segments = op_seg.second;
        OrderedParams p;
        // make first element data, second weights
        if (segments[0].type != NodeSegment::DATA) {
            segments[0].swap(segments[1]);
        }
        p.first = segments[NodeSegment::DATA][0];
        p.second = segments[NodeSegment::WEIGHTS][0];
        param_list[p].push_back(op_seg.first);
    }
    // remove pairs with single op
    std::cout << "removed single op pairs" << std::endl;
    for (auto it = param_list.begin(); it != param_list.end();) {
        if (it->second.size() < 2) {
            it = param_list.erase(it);
        }
        else {
            ++it;
        }
    }



    for (auto& p : param_list) {
        OrderedParams params = p.first;
        std::cout << "pair: [" << params.first->get_friendly_name() << ", "
                  << params.second->get_friendly_name() << "]" << std::endl;
        for (auto& op : p.second) {
            std::cout << "op: " << op->get_friendly_name() << std::endl;
        }
    }

    // TODO: check consistency
    // check dot op matrix order
    // check shape size

    // create new ops
    // for each parameter pair
    // create
    // for each dot op
    std::cout << "replacing nodes" << std::endl;
    for (auto& p : param_list) {
        OrderedParams params = p.first;
        auto data_node = params.first;
        auto weights_node = params.second;

        NodeVector &op_nodes = p.second;
        // get the first combo op
        auto first_op = op_nodes[0];
        auto data_segment = op_seg_map[first_op][NodeSegment::DATA];
        auto weights_segment = op_seg_map[first_op][NodeSegment::WEIGHTS];

        // construct new op nodes
        ngraph::AxisVector data_order(data_node->get_shape().size());
        std::iota(begin(data_order), end(data_order), 0);
        const auto& data_shape = data_node->get_shape();
        auto data_reshape_node = std::make_shared<op::Reshape>(data_node, data_order, Shape{data_shape[0]*data_shape[1], data_shape[2]});
        auto weights_reshape_node = weights_segment[1]->copy_with_new_args({weights_node});
        auto dot_node = data_segment.back()->copy_with_new_args({data_reshape_node, weights_reshape_node});
        // create a slice for each user of the dot op matching the original slice op's lower/upper bound
        NodeVector slice_ops;
        NodeVector reshape_ops;
        const auto &dot_shape = dot_node->get_shape();
        std::cout << dot_shape[0] << " " << dot_shape[1] << " " << dot_shape[2] << std::endl;
        for (auto op : op_nodes) {
            auto cur_data_segment = op_seg_map[op][NodeSegment::DATA];
            auto old_slice = std::dynamic_pointer_cast<ngraph::op::Slice>(cur_data_segment[1]);
            const auto& lb = old_slice->get_lower_bounds();
            const auto& ub = old_slice->get_upper_bounds();
            std::cout << lb[0] << " " << lb[1] << " " << lb[2] << std::endl;
            std::cout << ub[0] << " " << ub[1] << " " << ub[2] << std::endl;
            ngraph::Coordinate lower_bounds({ub[0]*lb[1], 0});
            ngraph::Coordinate upper_bounds({ub[0]*ub[1], dot_shape[1]});
            std::cout << lower_bounds[0] << " " << lower_bounds[1] << " " << lower_bounds[2] << std::endl;
            std::cout << upper_bounds[0] << " " << upper_bounds[1] << " " << upper_bounds[2] << std::endl;
            auto slice_node =
                std::make_shared<ngraph::op::Slice>(dot_node, lower_bounds, upper_bounds);

//            ngraph::AxisVector order(slice_node->get_shape().size());
//            std::iota(begin(order), end(order), 0);
//            auto reshape_node =
//                std::make_shared<op::Reshape>(slice_node, order, Shape{dot_shape[0], dot_shape[2]});

            // replace old nodes
            function->replace_node(op, slice_node);
        }
    }


#if 0
//        p1->
//        function->replace_node(nv[i], p);
//
//        for (auto& dot : dot_ops) {
//            for (auto& nv : dot.second) {
//                const auto& param_node = nv[0];
//                for (size_t i = 1; i < nv.size()-1; ++i) {
//                    std::cout << "replacing " << nv[i]->get_friendly_name() << " with "
//                              << param_node->get_friendly_name() << std::endl;
//                    function->replace_node(nv[i], p);
//                }
//            }
//        }
//        for (auto v : p1->get_shape()) {
//            std::cout << v << " ";
//        }
//        std::cout << std::endl;
//        for (auto v : p2->get_shape()) {
//            std::cout << v << " ";
//        }
//        std::cout << std::endl;
//    for (auto& n : function->get_ordered_ops())
//    {
//        // Work around a warning [-Wpotentially-evaluated-expression]
//        Node& node = *n;
//        std::cout << "instance id: " << node.get_instance_id() << std::endl;
//        std::string type = "other";
//        if (TI(node) == TI(ngraph::op::Parameter)) {
//            param_nodes.push_back(n);
//        }
//        if (TI(node) == TI(ngraph::op::Slice)) {
//            type = "Slice";
//        }
//        if (TI(node) == TI(ngraph::op::Reshape)) {
//            type = "Reshape";
//        }
//        if (TI(node) == TI(ngraph::op::Dot)) {
//            type = "Dot";
//        }
//        std::cout << "node (" << type << "): " << node.get_friendly_name() << std::endl;
//        for (const auto& in : node.get_input_ops()) {
//            std::cout << "    in:  " << in->get_friendly_name() << std::endl;
//        }
//        auto outputs = get_users(node);
//        for (const auto& out : outputs) {
//            std::cout << "    out: " << out->get_friendly_name() << std::endl;
//        }
//    }

//        std::shared_ptr<op::Reshape> n1_reshape = std::make_shared(op::Reshape(A, AxisVector{1, 0}, shape_r)));
#endif
    return clobbered;
}
