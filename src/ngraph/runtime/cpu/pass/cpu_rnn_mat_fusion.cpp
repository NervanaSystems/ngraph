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

#define TI(x) std::type_index(typeid(x))

// a sequence of nodes, identified with a segment type for the input parameter type
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
void FindValidSegments(const std::shared_ptr<Node> &node,
                       NodeSegment segment,
                       std::vector<NodeSegment> &segment_bundle,
                       NodeTypeSequenceSet valid_sequence_list,
                       int depth)
{
    // base case, we have one valid segment left, and depth is exeeding the valid seqence length
    // we found a match
    if (valid_sequence_list.size() == 1 && depth >= valid_sequence_list.front().second.size()) {
        segment.type = valid_sequence_list.front().first;
        segment_bundle.push_back(segment);
        return;
    }
    const Node& node_ref = *node;
    for (auto seq_it = valid_sequence_list.begin(); seq_it != valid_sequence_list.end();) {
        const auto& valid_seq = seq_it->second;
        // skip and remove sequences which are longer or doesn't match current node at depth index
        if (depth >= valid_seq.size() || TI(node_ref) != valid_seq[depth]) {
            seq_it = valid_sequence_list.erase(seq_it);
            continue;
        }
        else {
            ++seq_it;
        }
    }
    // postconditions:
    // valid_sequnce_list.size() > 0 : there's still valid sequence to match, continue recursion
    // valid_sequnce_list.size() = 0 : terminate
    if (valid_sequence_list.size() > 0) {
        segment.push_back(node);
        const auto outputs = node->get_users();
        for (const auto& out_node : outputs) {
            FindValidSegments(out_node, segment, segment_bundle, valid_sequence_list, depth + 1);
        }
    }
}


bool runtime::cpu::pass::CPURnnMatFusion::run_on_function(std::shared_ptr<Function> function)
{
    std::cout << "##### Running CPURnnMatFusion" << std::endl;
    bool modified = false;

    const NodeTypeSequenceSet valid_sequences {
        {NodeSegment::DATA,
         {TI(op::Parameter),
          TI(op::Slice),
          TI(op::Reshape),
          TI(op::Dot)}},
        {NodeSegment::WEIGHTS,
         {TI(op::Parameter),
          TI(op::Reshape),
          TI(op::Dot)}}
    };

    // iterate all parameters and find path to dot op
    std::vector<NodeSegment> segment_bundle;
    for (auto& node : function->get_ordered_ops()) {
        NodeSegment segment;
        FindValidSegments(node, segment, segment_bundle, valid_sequences, 0);
    }

    // combined all segments by last operator
    std::map<std::shared_ptr<Node>, std::vector<NodeSegment>> op_seg_map;
    for (const auto& segment : segment_bundle) {
        op_seg_map[segment.back()].push_back(segment);
    }

    // remove ops with single segment
    for (auto op_it = op_seg_map.cbegin(); op_it != op_seg_map.cend();) {
        if (op_it->second.size() < 2) {
            op_it = op_seg_map.erase(op_it);
        }
        else {
            ++op_it;
        }
    }

    // create a lookup map for each unique pair of parameters
    typedef std::pair<std::shared_ptr<Node>, std::shared_ptr<Node>> OrderedParams;
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
    for (auto it = param_list.cbegin(); it != param_list.cend();) {
        if (it->second.size() < 2) {
            it = param_list.erase(it);
        }
        else {
            ++it;
        }
    }

    // Expecting in put data shape D=[x, y, z], weights W=[u, v]
    // where y is the time step. We are computing R=dot(D,W)=[x,y,v]. We can reshape D to D'=[x*y, z], then we have dot(D',W), result
    // in R=[x*y, v], then we need to slice the result by strided by time steps.
    // iterate each unique pair of parameters, replace dot operations
    for (auto& p : param_list) {
        OrderedParams params = p.first;
        NodeVector &op_nodes = p.second;

        auto data_node = params.first;
        auto weights_node = params.second;

        // get the first combo op
        auto first_op = op_nodes[0];
        auto first_weights_segment = op_seg_map[first_op][NodeSegment::WEIGHTS];

        // construct new op nodes
        AxisVector data_order(data_node->get_shape().size());
        std::iota(begin(data_order), end(data_order), 0);
        const auto& data_shape = data_node->get_shape();
        auto data_reshape_node = std::make_shared<op::Reshape>(data_node, data_order, Shape{data_shape[0]*data_shape[1], data_shape[2]});
        auto weights_reshape_node = first_weights_segment[1]->copy_with_new_args({weights_node});
        auto dot_node = std::make_shared<op::Dot>(data_reshape_node, weights_reshape_node);
        const auto& dot_shape = dot_node->get_shape();

        // create a slice for each user of the dot op matching the original dot op's output
        for (auto op : op_nodes) {
            const auto& cur_data_segment = op_seg_map[op][NodeSegment::DATA];
            const auto old_slice = std::dynamic_pointer_cast<op::Slice>(cur_data_segment[1]);
            const auto& lb = old_slice->get_lower_bounds();
            // lower bound matching the current time step
            const Coordinate lower_bounds{lb[1], 0};
            // striding by the number of time steps
            const Strides strides{data_shape[1],1};
            auto slice_node = std::make_shared<op::Slice>(dot_node, lower_bounds, dot_shape, strides);

            // replace old nodes
            function->replace_node(op, slice_node);
        }
        modified = true;
    }
    std::cout << "##### Finished CPURnnMatFusion: " << modified << std::endl;
    return modified;
}
