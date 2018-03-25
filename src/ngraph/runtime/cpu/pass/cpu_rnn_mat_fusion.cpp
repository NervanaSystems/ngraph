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

#include <array>
#include <map>
#include <memory>
#include <numeric>
#include <stack>
#include <typeindex>
#include <unordered_map>

#include "cpu_rnn_mat_fusion.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/dot.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/slice.hpp"

#include "ngraph/pattern/matcher.hpp"
#include "ngraph/pattern/op/label.hpp"


using namespace ngraph;

typedef std::shared_ptr<Node> NodePtr;

#define TI(x) std::type_index(typeid(x))

// a sequence of nodes, identified with a segment type for the input parameter type
struct NodeSegment : public NodeVector
{
    enum Type
    {
        DATA = 0,
        WEIGHTS,
        BIAS,
        UNDEFINED
    };
    Type type{UNDEFINED};
};
typedef std::pair<NodeSegment::Type, std::vector<std::type_index>> NodeTypeSequence;
typedef std::list<NodeTypeSequence> NodeTypeSequenceList;


//{TI(op::Parameter), TI(op::Slice), TI(op::Reshape), TI(op::Dot), TI(op::Add)}},
static std::shared_ptr<Node> construct_data_pattern(std::shared_ptr<pattern::op::Label> DATA_SLICE)
{	
	//auto slice = std::make_shared<op::Slice>(DATA, Coordinate{ 0, 0, 0 }, Coordinate{ 1, 2, 4 }, Strides{ 1, 1, 1 });
	//auto reshape_slice = std::make_shared<op::Reshape>(slice, AxisVector{ 0, 1, 2 }, Shape{ 2, 4 });
	auto reshape_slice = std::make_shared<op::Reshape>(DATA_SLICE, AxisVector{ 0, 1, 2 }, Shape{ 2, 4 });
	auto W = std::make_shared<pattern::op::Label>(element::f32, Shape{ 4, 1 });
	auto dot = std::make_shared<op::Dot>(reshape_slice, W);
	auto broadcast = std::make_shared<pattern::op::Label>(element::f32, dot->get_shape());
	return dot + broadcast;
}

//{NodeSegment::WEIGHTS, { TI(op::Parameter), TI(op::Reshape), TI(op::Dot), TI(op::Add) }},
static std::shared_ptr<Node> construct_weights_pattern(std::shared_ptr<pattern::op::Label> WEIGHTS_RESHAPE)
{
	//auto reshape_weights = std::make_shared<op::Reshape>(WEIGHTS, AxisVector{ 0 }, Shape { 4, 1 });
	auto X = std::make_shared<pattern::op::Label>(element::f32, Shape{ 2, 4 });
	//auto dot = std::make_shared<op::Dot>(X, reshape_weights);
	auto dot = std::make_shared<op::Dot>(X, WEIGHTS_RESHAPE);
	auto broadcast = std::make_shared<pattern::op::Label>(element::f32, dot->get_shape());
	return dot + broadcast;
}

//{NodeSegment::BIAS, {TI(op::Parameter), TI(op::Broadcast), TI(op::Add)}}};
static std::shared_ptr<Node> construct_bias_pattern(std::shared_ptr<pattern::op::Label> BIAS_BROADCAST)
{
	auto dot_label = std::make_shared<pattern::op::Label>(element::i32, Shape { 2, 1 });
	//auto broadcast = std::make_shared<op::Broadcast>(BIAS, dot_label->get_shape(), AxisSet{ 0 });
	return dot_label + BIAS_BROADCAST;
}

// Preorder traversal to collect all valid segments in the graph
// precondition: all valid sequences must be unique
// [a, b, c] and [a, c, b] are different, for valid sequences like [a, b, c] and [a, b], the
// longest sequence will be matched.
void FindValidSegments(const NodePtr& node,
                       NodeSegment segment,
                       std::vector<NodeSegment>& segment_bundle,
                       NodeTypeSequenceList valid_sequence_list,
                       int depth)
{
    const Node& node_ref = *node;
    // check current node against all valid sequences at current depth level. Remove sequences
    // which does not match current node type
    for (auto seq_it = valid_sequence_list.begin(); seq_it != valid_sequence_list.end();)
    {
        const auto& valid_seq = seq_it->second;
        // remove sequences which are too short or doesn't match current node type at depth index
        if (depth >= valid_seq.size() || TI(node_ref) != valid_seq[depth])
        {
            seq_it = valid_sequence_list.erase(seq_it);
        }
        else
        {
            ++seq_it;
        }
    }
    // postconditions:
    // valid_sequnce_list.size() > 0 : there's still valid sequences to match
    // otherwise : terminate
    if (valid_sequence_list.size() > 0)
    {
        segment.push_back(node);
        // base case, we have one valid segment left (since valid sequences are expected to be
        // unique), and current depth matches (sequence-length - 1) (i.e. last node)
        // we found a match
        if (valid_sequence_list.size() == 1 &&
            depth == (valid_sequence_list.front().second.size() - 1))
        {
            segment.type = valid_sequence_list.front().first;
            segment_bundle.push_back(segment);
            return;
        }
        // still have more than one sequences to check, continue traversal
        else
        {
            const auto outputs = node->get_users();
            for (const auto& out_node : outputs)
            {
                FindValidSegments(
                    out_node, segment, segment_bundle, valid_sequence_list, depth + 1);
            }
        }
    }
}

// this is the expected sequence count for fusing
const size_t SEGMENT_COUNT = 3;

struct OrderedParams
{
public:
    OrderedParams()
        : m_params{{nullptr, nullptr, nullptr}}
    {
    }

    bool valid()
    {
        return std::none_of(m_params.cbegin(), m_params.cend(), [](const NodePtr& n) -> bool {
            return n == nullptr;
        });
    }

    void set(const NodeSegment::Type type, const NodePtr& node) { m_params[type] = node; }
    NodePtr get(const NodeSegment::Type type) const { return m_params.at(type); }
    friend bool operator<(const OrderedParams& a, const OrderedParams& b);

private:
    // order based on NodeSegment::Type
    // <data, weights, bias>
    std::array<NodePtr, SEGMENT_COUNT> m_params;
};

bool operator<(const OrderedParams& a, const OrderedParams& b)
{
    return a.m_params < b.m_params;
}

bool runtime::cpu::pass::CPURnnMatFusion::run_on_function(std::shared_ptr<Function> function)
{
    bool modified = false;

    const NodeTypeSequenceList valid_sequences{
        {NodeSegment::DATA,
         {TI(op::Parameter), TI(op::Slice), TI(op::Reshape), TI(op::Dot), TI(op::Add)}},
        {NodeSegment::WEIGHTS, {TI(op::Parameter), TI(op::Reshape), TI(op::Dot), TI(op::Add)}},
        {NodeSegment::BIAS, {TI(op::Parameter), TI(op::Broadcast), TI(op::Add)}}};

    // find all parameter nodes
    std::vector<NodePtr> param_nodes;
    for (auto& node : function->get_ordered_ops())
    {
        if (node->is_parameter())
        {
            param_nodes.push_back(node);
        }
    }

    // iterate all parameters and find all valid segments
    std::vector<NodeSegment> segment_bundle;
    for (auto& node : param_nodes)
    {
        NodeSegment segment;
        FindValidSegments(node, segment, segment_bundle, valid_sequences, 0);
    }

    // combined all segments by last operator
    std::map<NodePtr, std::vector<NodeSegment>> op_seg_map;
    for (const auto& segment : segment_bundle)
    {
        auto op_it = op_seg_map.find(segment.back());
        if (op_it == op_seg_map.end())
        {
            auto insert_result = op_seg_map.insert(
                std::make_pair(segment.back(), std::vector<NodeSegment>(SEGMENT_COUNT)));
            op_it = insert_result.first;
        }
        (op_it->second)[segment.type] = segment;
    }

    // remove ops with less than SEGMENT_COUNT number of segments
    for (auto op_it = op_seg_map.cbegin(); op_it != op_seg_map.cend();)
    {
        // remove ops with less than expected segements
        bool valid = true;
        for (auto& seg : op_it->second)
        {
            if (seg.empty())
            {
                valid = false;
                break;
            }
        }
        if (!valid)
        {
            op_it = op_seg_map.erase(op_it);
        }
        else
        {
            ++op_it;
        }
    }

    // create a lookup map for each unique set of parameters
    std::map<OrderedParams, NodeVector> param_list;
    for (auto& op_seg : op_seg_map)
    {
        std::vector<NodeSegment>& segments = op_seg.second;
        OrderedParams p;
        // put each segment's parameter in the OrderedParams by type
        for (auto& seg : segments)
        {
            p.set(seg.type, seg[0]);
        }
        // if any of them is missing, p will be invalid
        // this can happen for example, when two of them are both
        // weights
        if (p.valid())
        {
            param_list[p].push_back(op_seg.first);
        }
    }

    // remove params with single combo op (meaning no need to combine slicing)
    for (auto it = param_list.cbegin(); it != param_list.cend();)
    {
        if (it->second.size() < 2)
        {
            it = param_list.erase(it);
        }
        else
        {
            ++it;
        }
    }

	//============================== REFACTORED VERSION =========================================

	//auto DATA = std::make_shared<pattern::op::Label>(element::f32, Shape{ 2, 2, 4 });
	auto data_pred = [](std::shared_ptr<Node> n)
	{
		return std::dynamic_pointer_cast<op::Slice>(n) != nullptr;
	};
	auto DATA_SLICE = std::make_shared<pattern::op::Label>(element::f32, Shape{ 1, 2, 4 }, data_pred);
	auto data_pattern = construct_data_pattern(DATA_SLICE);

	//auto WEIGHTS = std::make_shared<pattern::op::Label>(element::f32, Shape{ 4 });
	auto weights_pred = [](std::shared_ptr<Node> n)
	{
		return std::dynamic_pointer_cast<op::Reshape>(n) != nullptr;
	};
	auto WEIGHTS_RESHAPE = std::make_shared<pattern::op::Label>(element::f32, Shape{ 4, 1 }, weights_pred);
	auto weights_pattern = construct_weights_pattern(WEIGHTS_RESHAPE);

	//auto BIAS = std::make_shared<pattern::op::Label>(element::f32, Shape{ 1 });
	//we don't really need a broadcast node but 
	//labelling a Broadcast allows us to extract
	//parms from all 3 labels in the same fashion
	//(i.e. via get_input_op(0))
	auto broadcast_pred = [](std::shared_ptr<Node> n)
	{
		return std::dynamic_pointer_cast<op::Broadcast>(n) != nullptr;
	};
	auto BIAS_BROADCAST = std::make_shared<pattern::op::Label>(element::f32, Shape{ 1 }, broadcast_pred);
	auto bias_pattern = construct_bias_pattern(BIAS_BROADCAST);

	const size_t NUM_MMB_ARGS = 3;
	std::shared_ptr<pattern::op::Label> labels[] = { DATA_SLICE, WEIGHTS_RESHAPE, BIAS_BROADCAST };
	//Matchers' ordering is important! Don't change!
	std::shared_ptr<pattern::Matcher> matchers[] =
	{ 
		std::make_shared<pattern::Matcher>(data_pattern),
		std::make_shared<pattern::Matcher>(weights_pattern),
		std::make_shared<pattern::Matcher>(bias_pattern)
	};

	std::map<std::shared_ptr<Node>, NodeVector> op_seg_map2; //add to list of parms
	std::map<NodeVector, NodeVector> param_list2;
	for (auto n : function->get_ordered_ops())
	{
		NodeVector parms;
		NodeVector matched_nodes;
		for (size_t i = 0; i < NUM_MMB_ARGS; i++)
		{
			auto matcher = matchers[i];
			if (matcher->match(n))
			{
				//if we get all 3 matches they will all fall 
				//in the right spots (e.g. DATA, WEIGHTS, BIAS) since matchers are ordered
				//if we have less than 3 matches we skip this node anyways
				auto matched = matcher->get_pattern_map()[labels[i]];
				parms.push_back(matched->get_input_op(0));
				matched_nodes.push_back(matched);
			}

			if (parms.size() != NUM_MMB_ARGS)
			{
				continue;
			}

			//we have a full set for the current Add (n) i.e. data, weights, bias
			op_seg_map2.insert(std::make_pair(n, matched_nodes));
			param_list2[parms].push_back(n);
		}
	}

	// remove params with single combo op (meaning no need to combine slicing)
	for (auto it = param_list2.cbegin(); it != param_list2.cend();)
	{
		if (it->second.size() < 2)
		{
			it = param_list2.erase(it);
		}
		else
		{
			++it;
		}
	}

    // Expecting input data shape D=[x, y, z], weights W=[u, v], bias B = [w]
    // where y is the time step. We are computing R=dot(D,W)=[x,y,v]. We can reshape D to D'=[x*y, z], then we have dot(D',W), result
    // in R=[x*y, v], then add(R,B). We need to slice the result by strided by time steps.
    // iterate each unique set of parameters, replace original operations
    for (auto& p : param_list2)
    {
        NodeVector params = p.first;
        NodeVector& op_nodes = p.second;

        auto data_node = params.at(NodeSegment::DATA);
		auto weights_node = params.at(NodeSegment::WEIGHTS);
        auto bias_node = params.at(NodeSegment::BIAS);

        const auto& data_shape = data_node->get_shape();
        // construct new op nodes
        AxisVector data_order(data_node->get_shape().size());
        std::iota(begin(data_order), end(data_order), 0);
        auto data_reshape_node = std::make_shared<op::Reshape>(
            data_node, data_order, Shape{data_shape[0] * data_shape[1], data_shape[2]});

		auto old_weights_reshape_node = op_seg_map2.at(op_nodes.at(0)).at(NodeSegment::WEIGHTS);
        auto weights_reshape_node = old_weights_reshape_node->copy_with_new_args({weights_node});
        auto dot_node = std::make_shared<op::Dot>(data_reshape_node, weights_reshape_node);
        const auto& dot_shape = dot_node->get_shape();

        auto bias_broadcast_node =
            std::make_shared<op::Broadcast>(bias_node, dot_shape, AxisSet{0});
        auto add_node = std::make_shared<op::Add>(dot_node, bias_broadcast_node);
        const auto& add_shape = add_node->get_shape();

        // create a slice for each user of the dot op matching the original dot op's output
        for (auto op : op_nodes)
        {
            const auto old_slice = std::dynamic_pointer_cast<op::Slice>(op_seg_map2[op].at(NodeSegment::DATA));
            const auto& old_lower_bounds = old_slice->get_lower_bounds();
            // lower bound matching the current time step
            const Coordinate lower_bounds{old_lower_bounds[1], 0};
            // striding by the number of data
            const Strides strides{data_shape[1], 1};
            auto slice_node =
                std::make_shared<op::Slice>(add_node, lower_bounds, add_shape, strides);

            // replace old nodes
            function->replace_node(op, slice_node);
        }
        modified = true;
    }
    return modified;
}
