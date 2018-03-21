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
#include <fstream>
#include <sys/time.h>

#include "ngraph/ops/add.hpp"
#include "ngraph/ops/broadcast.hpp"
#include "ngraph/ops/dot.hpp"
#include "ngraph/ops/reshape.hpp"
#include "ngraph/ops/slice.hpp"
#include "ngraph/ops/parameter.hpp"
#include "cpu_rnn_mat_fusion.hpp"
#include "ngraph/serializer.hpp"

using namespace ngraph;

typedef std::shared_ptr<Node> NodePtr;

static double dtime()
{
    double tseconds = 0.0;
    struct timeval mytime;
    gettimeofday(&mytime,(struct timezone*)0);
    tseconds = (double)(mytime.tv_sec +
            mytime.tv_usec*1.0e-6);
    return( tseconds );
}
int runtime::cpu::pass::CPURnnMatFusion::counter = 0;
#define TI(x) std::type_index(typeid(x))

// a sequence of nodes, identified with a segment type for the input parameter type
struct NodeSegment : public NodeVector
{
    enum Type {
      DATA = 0,
      WEIGHTS,
      BIAS,
      UNDEFINED
    };
    Type type {UNDEFINED};
};
typedef std::pair<NodeSegment::Type, std::vector<std::type_index>> NodeTypeSequence;
typedef std::list<NodeTypeSequence> NodeTypeSequenceSet;

// Preorder traversal to collect all valid segments in the graph
// precondition: all valid sequences must be unique
// [a, b, c] and [a, c, b] are different, for valid sequences like [a, b, c] and [a, b], the
// longest sequence will be matched.
void FindValidSegments(const NodePtr &node,
                       NodeSegment segment,
                       std::vector<NodeSegment> &segment_bundle,
                       NodeTypeSequenceSet valid_sequence_list,
                       int depth)
{
//    std::cout << "visiting: " << node->get_friendly_name() << std::endl;
    const Node& node_ref = *node;
    // check current node against all valid sequences at current depth level. Remove sequences
    // which does not match current node type
    for (auto seq_it = valid_sequence_list.begin(); seq_it != valid_sequence_list.end();) {
        const auto& valid_seq = seq_it->second;
        // remove sequences which are too short or doesn't match current node type at depth index
        if (depth >= valid_seq.size() || TI(node_ref) != valid_seq[depth]) {
            seq_it = valid_sequence_list.erase(seq_it);
            continue;
        }
        else {
            ++seq_it;
        }
    }
    // postconditions:
    // valid_sequnce_list.size() > 0 : there's still valid sequences to match
    // otherwise : terminate
    if (valid_sequence_list.size() > 0) {
        segment.push_back(node);
        // base case, we have one valid segment left (since valid sequences are expected to be
        // unique), and current depth matches (sequence-length - 1) (i.e. last node)
        // we found a match
        if (valid_sequence_list.size() == 1 && depth == (valid_sequence_list.front().second.size()-1)) {
            segment.type = valid_sequence_list.front().first;
            segment_bundle.push_back(segment);
//            std::cout << "### added " << segment.back()->get_friendly_name() << " (" << segment.type <<
//                                                                                                   ")"
//                                                                      << std::endl;
            return;
        }
        // still have more than one sequences to check, continue traversal
        else {
            const auto outputs = node->get_users();
            for (const auto& out_node : outputs) {
                FindValidSegments(out_node, segment, segment_bundle, valid_sequence_list, depth + 1);
            }
        }
    }
}

struct OrderedParams
{
public:
    OrderedParams()
    : m_params{nullptr, nullptr, nullptr}
    {}

    bool exist(const NodeSegment::Type type) {
        return m_params[type] != nullptr;
    }

    bool valid() {
        return std::none_of(m_params.cbegin(), m_params.cend(),
                            [](const NodePtr& n) { return n == nullptr; });
    }

    void set(const NodeSegment::Type type, const NodePtr& node) {
        m_params[type] = node;
    }

    NodePtr get(const NodeSegment::Type type) const {
        return m_params.at(type);
    }

    friend bool operator<(const OrderedParams& a, const OrderedParams& b);
private:
    // order based on NodeSegment::Type
    // <data, weights, bias>
    std::array<NodePtr, 3> m_params;
};

bool operator<(const OrderedParams& a, const OrderedParams& b) {
    return a.m_params < b.m_params;
}

bool runtime::cpu::pass::CPURnnMatFusion::run_on_function(std::shared_ptr<Function> function)
{
    double tstart = dtime();
    std::cout << "##### Running CPURnnMatFusion" << std::endl;
    {
#if 0
        const std::string file_string = "rnn-" + std::to_string(counter) + "-before.json";
        std::string json_data = ngraph::serialize(function, 4, false);
        std::cout << "serializing: " << file_string << std::endl;
        std::ofstream write;
        write.open(file_string.c_str(), std::ios::out);
        write << json_data;
        write.close();
#endif
    }

    bool modified = false;

    const NodeTypeSequenceSet valid_sequences {
        {NodeSegment::DATA,
         {TI(op::Parameter),
          TI(op::Slice),
          TI(op::Reshape),
          TI(op::Dot),
          TI(op::Add)}},
        {NodeSegment::WEIGHTS,
         {TI(op::Parameter),
          TI(op::Reshape),
          TI(op::Dot),
          TI(op::Add)}},
        {NodeSegment::BIAS,
         {TI(op::Parameter),
          TI(op::Broadcast),
          TI(op::Add)}}
    };
    // this is the expected sequence count for fusing
    const size_t sequence_count = 3;

    // find all parameter nodes
    std::vector<NodePtr> param_nodes;
    for (auto& node : function->get_ordered_ops()) {
        const Node& node_ref = *node;
        if (TI(node_ref) == TI(op::Parameter)) {
            param_nodes.push_back(node);
        }
    }

    // iterate all parameters and find all valid segments
    std::vector<NodeSegment> segment_bundle;
    for (auto& node : param_nodes) {
        std::cout << "param: " << node->get_friendly_name() << std::endl;
        const auto outputs = node->get_users();
        for (const auto& out : outputs) {
            std::cout << "    out: " << out->get_friendly_name() << std::endl;
        }
        NodeSegment segment;
        FindValidSegments(node, segment, segment_bundle, valid_sequences, 0);
    }

    std::cout << "###### valid segments" << std::endl;
    // combined all segments by last operator
    std::map<NodePtr, std::vector<NodeSegment>> op_seg_map;
    for (const auto& segment : segment_bundle) {
        std::cout << "segment: \n";
        for (auto& n : segment) {
            std::cout << "  " << n->get_friendly_name() << std::endl;
        }
        op_seg_map[segment.back()].push_back(segment);
    }
    std::cout << "###### op seg map" << std::endl;
    for (auto& op : op_seg_map) {
        std::cout << "op: " << op.first->get_friendly_name() << std::endl;
        for (auto& vn : op.second) {
            std::cout << "  segment (" << static_cast<int>(vn.type) << "): " << std::endl;
            for (auto& n : vn) {
                std::cout << "   node: " << n->get_friendly_name() << std::endl;
            }
        }
    }

    // remove ops with single segment
    for (auto op_it = op_seg_map.cbegin(); op_it != op_seg_map.cend();) {
        // remove ops with less than expected segements
        if (op_it->second.size() < sequence_count) {
            op_it = op_seg_map.erase(op_it);
        }
        else {
            ++op_it;
        }
    }

    // create a lookup map for each unique pair of parameters
    std::map<OrderedParams, NodeVector> param_list;
    for (auto& op_seg : op_seg_map) {
        std::vector<NodeSegment>& segments = op_seg.second;
        OrderedParams p;
        for (auto& seg : segments) {
            p.set(seg.type, seg[0]);
        }
        if (p.valid()) {
            param_list[p].push_back(op_seg.first);
        }
    }

    // remove params with single combo op (meaning no need to combine slicing)
    for (auto it = param_list.cbegin(); it != param_list.cend();) {
        if (it->second.size() < 2) {
            it = param_list.erase(it);
        }
        else {
            ++it;
        }
    }

    for (auto& p : param_list) {
        std::cout << "params: ["
                  << p.first.get(NodeSegment::DATA)->get_friendly_name() << ", "
                  << p.first.get(NodeSegment::WEIGHTS)->get_friendly_name() << ", "
                  << p.first.get(NodeSegment::BIAS)->get_friendly_name() << "]" << std::endl;
        std::cout << "  op: " << std::endl;
        for (auto& n : p.second) {
            std::cout << "   node: " << n->get_friendly_name() << std::endl;
        }
    }

#if 1
    // Expecting in put data shape D=[x, y, z], weights W=[u, v]
    // where y is the time step. We are computing R=dot(D,W)=[x,y,v]. We can reshape D to D'=[x*y, z], then we have dot(D',W), result
    // in R=[x*y, v], then we need to slice the result by strided by time steps.
    // iterate each unique pair of parameters, replace dot operations
    for (auto& p : param_list) {
        OrderedParams params = p.first;
        NodeVector &op_nodes = p.second;

        auto data_node = params.get(NodeSegment::DATA);
        auto weights_node = params.get(NodeSegment::WEIGHTS);
        auto bias_node = params.get(NodeSegment::BIAS);

        const auto& data_shape = data_node->get_shape();
        const auto& weights_shape = weights_node->get_shape();
        const auto& bias_shape = bias_node->get_shape();

        // get the first combo op
        auto first_op = op_nodes[0];
        auto first_weights_segment = op_seg_map[first_op][NodeSegment::WEIGHTS];

        // construct new op nodes
        AxisVector data_order(data_node->get_shape().size());
        std::iota(begin(data_order), end(data_order), 0);
        auto data_reshape_node = std::make_shared<op::Reshape>(data_node, data_order, Shape{data_shape[0]*data_shape[1], data_shape[2]});
        auto weights_reshape_node = first_weights_segment[1]->copy_with_new_args({weights_node});
        auto dot_node = std::make_shared<op::Dot>(data_reshape_node, weights_reshape_node);
        const auto& dot_shape = dot_node->get_shape();

        auto bias_broadcast_node = std::make_shared<op::Broadcast>(bias_node, dot_shape, AxisSet{0});
        auto add_node = std::make_shared<op::Add>(dot_node, bias_broadcast_node);
        const auto& add_shape = add_node->get_shape();

        // create a slice for each user of the dot op matching the original dot op's output
        for (auto op : op_nodes) {
            const auto& cur_data_segment = op_seg_map[op][NodeSegment::DATA];
            const auto old_slice = std::dynamic_pointer_cast<op::Slice>(cur_data_segment[1]);
            const auto& lb = old_slice->get_lower_bounds();
            // lower bound matching the current time step
            const Coordinate lower_bounds{lb[1], 0};
            // striding by the number of time steps
            const Strides strides{data_shape[1],1};
//            auto slice_node = std::make_shared<op::Slice>(add_node, lower_bounds, add_shape, strides);
//
//            // replace old nodes
//            function->replace_node(op, slice_node);
        }
        modified = true;
    }
#endif
#if 0
    if (modified) {
        const std::string file_string = "rnn-" + std::to_string(counter) + "-after.json";
//        ngraph::serialize(file_string, function, 4);
        std::string json_data = ngraph::serialize(function, 4, false);
        std::ofstream write;
        std::cout << "serializing: " << file_string << std::endl;
        write.open(file_string.c_str(), std::ios::out);
        write << json_data;
        write.close();
    }
    ++counter;
#endif
    std::cout << "##### Finished CPURnnMatFusion: " << modified << std::endl;
    double ttime = dtime() - tstart;
    std::cout << "rnn fusion time: " << ttime << std::endl;
    return modified;
}
