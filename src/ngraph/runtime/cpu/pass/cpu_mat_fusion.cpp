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

#include <array>
#include <map>
#include <memory>
#include <numeric>
#include <stack>
#include <typeindex>
#include <unordered_map>

#include "cpu_mat_fusion.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/dot.hpp"
#include "ngraph/op/fused/batch_mat_mul_transpose.hpp"
#include "ngraph/op/fused/group_conv.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/pattern/matcher.hpp"
#include "ngraph/pattern/op/label.hpp"
#include "ngraph/util.hpp"

using namespace ngraph;

struct Type
{
    enum
    {
        DATA = 0,
        WEIGHTS,
        BIAS,
    };
};

// constructs (x*W + bias)
static std::shared_ptr<pattern::Matcher>
    construct_rnn_input_linear_transformation(std::shared_ptr<pattern::op::Label> labels[])
{
    auto skip =
        std::make_shared<pattern::op::Skip>(labels[Type::DATA], pattern::has_class<op::Reshape>());
    auto dot = std::make_shared<op::Dot>(skip, labels[Type::WEIGHTS]);
    auto add_bias = std::make_shared<op::Add>(dot, labels[Type::BIAS]);
    return std::make_shared<pattern::Matcher>(add_bias);
}

static std::shared_ptr<Node> construct_data_pattern(std::shared_ptr<pattern::op::Label> data_slice)
{
    auto reshape_slice =
        std::make_shared<op::Reshape>(data_slice, AxisVector{0, 1, 2}, Shape{2, 4});
    auto W = std::make_shared<pattern::op::Label>(element::f32, Shape{4, 1});
    auto dot = std::make_shared<op::Dot>(reshape_slice, W);
    auto broadcast = std::make_shared<pattern::op::Label>(element::f32, dot->get_shape());
    return dot + broadcast;
}

static std::shared_ptr<Node>
    construct_weights_pattern(std::shared_ptr<pattern::op::Label> weights_reshape)
{
    auto X = std::make_shared<pattern::op::Label>(element::f32, Shape{2, 4});
    auto dot = std::make_shared<op::Dot>(X, weights_reshape);
    auto broadcast = std::make_shared<pattern::op::Label>(element::f32, dot->get_shape());
    return dot + broadcast;
}

static std::shared_ptr<Node>
    construct_bias_pattern(std::shared_ptr<pattern::op::Label> bias_broadcast)
{
    auto dot_label = std::make_shared<pattern::op::Label>(element::f32, Shape{2, 1});
    return dot_label + bias_broadcast;
}

bool runtime::cpu::pass::CPURnnMatFusion::run_on_function(std::shared_ptr<Function> function)
{
    bool modify_graph = false;

    //--------------------------------------------------------
    // Construct pattern version_1 for RNN input linear transformation
    auto data_slice = std::make_shared<pattern::op::Label>(
        element::f32, Shape{1, 2, 4}, pattern::has_class<op::Slice>());
    auto data_pattern = construct_data_pattern(data_slice);

    auto weights_reshape = std::make_shared<pattern::op::Label>(
        element::f32, Shape{4, 1}, pattern::has_class<op::Reshape>());
    auto weights_pattern = construct_weights_pattern(weights_reshape);

    // we don't really need a broadcast node but
    // labelling a Broadcast allows us to extract
    // params from all 3 labels in the same fashion
    //(i.e. via get_argument(0))
    auto bias_broadcast = std::make_shared<pattern::op::Label>(
        element::f32, Shape{2, 1}, pattern::has_class<op::Broadcast>());
    auto bias_pattern = construct_bias_pattern(bias_broadcast);

    const size_t NUM_MMB_ARGS = 3;
    std::shared_ptr<pattern::op::Label> labels_v1[] = {data_slice, weights_reshape, bias_broadcast};
    // Matchers' ordering is important! Don't change!
    std::shared_ptr<pattern::Matcher> matchers_v1[] = {
        std::make_shared<pattern::Matcher>(data_pattern),
        std::make_shared<pattern::Matcher>(weights_pattern),
        std::make_shared<pattern::Matcher>(bias_pattern)};

    // this DS will be used to hold the matched attributes from matchers_v1
    std::map<std::shared_ptr<Node>, NodeVector> op_seg_map; // add to list of params
    std::map<NodeVector, NodeVector> param_list;
    //--------------------------------------------------------

    //--------------------------------------------------------
    // Construct pattern version_2 for RNN input linear transformation
    auto input_data = std::make_shared<pattern::op::Label>(
        element::f32, Shape{10, 50}, pattern::has_class<op::Parameter>());
    auto W = std::make_shared<pattern::op::Label>(
        element::f32, Shape{50, 400}, pattern::has_class<op::Reshape>());
    auto b = std::make_shared<pattern::op::Label>(
        element::f32, Shape{10, 400}, pattern::has_class<op::Broadcast>());
    std::shared_ptr<pattern::op::Label> labels_v2[] = {input_data, W, b};
    auto matcher_v2 = construct_rnn_input_linear_transformation(labels_v2);

    // this DS will be used to hold the matched attributes from matcher_v2
    std::map<std::shared_ptr<Node>, NodeVector> map_weights_to_pattern;
    std::map<std::pair<std::shared_ptr<Node>, std::shared_ptr<Node>>, NodeVector>
        map_weights_bias_to_data;
    //--------------------------------------------------------

    for (auto n : function->get_ordered_ops())
    {
        NodeVector params;
        NodeVector matched_nodes;

        // checks if the graph matches to pattern defined in the matcher_v2
        if (matcher_v2->match(n))
        {
            auto matched_weight = matcher_v2->get_pattern_map()[W]->get_argument(0);
            auto matched_data = matcher_v2->get_pattern_map()[input_data];
            auto matched_bias = matcher_v2->get_pattern_map()[b]->get_argument(0);
            std::vector<size_t> supported_ranks{2, 3};

            if (!ngraph::is_valid_rank(matcher_v2->get_match_root(), supported_ranks))
            {
                NGRAPH_DEBUG << "Add (mat_fusion_v2) " << matcher_v2->get_match_root()->get_name()
                             << " isn't 2D or 3D";
                continue;
            }
            if (!ngraph::is_valid_rank(matched_weight, supported_ranks))
            {
                NGRAPH_DEBUG << "Weights (mat_fusion_v2) " << matched_weight << " isn't 2D or 3D";
                continue;
            }

            if (!ngraph::is_valid_rank(matched_data, supported_ranks))
            {
                NGRAPH_DEBUG << "Data (mat_fusion_v2) " << matched_data << " isn't 2D or 3D";
                continue;
            }

            map_weights_to_pattern[matched_weight].push_back(matcher_v2->get_match_root());
            map_weights_bias_to_data[std::make_pair(matched_weight, matched_bias)].push_back(
                matched_data);
        }

        for (size_t i = 0; i < NUM_MMB_ARGS; i++)
        {
            auto matcher = matchers_v1[i];
            if (matcher->match(n))
            {
                // if we get all 3 matches they will all fall
                // in the right spots (e.g. DATA, WEIGHTS, BIAS) since matchers are ordered
                // if we have less than 3 matches we skip this node anyways
                auto matched = matcher->get_pattern_map()[labels_v1[i]];
                params.push_back(matched->get_argument(0));
                matched_nodes.push_back(matched);
            }

            if (params.size() != NUM_MMB_ARGS)
            {
                continue;
            }

            // we have a full set for the current Add (n) i.e. data, weights, bias
            op_seg_map.insert(std::make_pair(n, matched_nodes));
            param_list[params].push_back(n);
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

    auto callback_matcher_v2 = [&]() -> void {
        // fuse the input vector to a matrix
        for (auto& it : map_weights_bias_to_data)
        {
            auto weights = it.first.first;
            auto bias = it.first.second;

            // if there's just one data node skip the optimization
            if (it.second.size() < 2)
            {
                return;
            }

            if (map_weights_to_pattern[weights].size() !=
                map_weights_bias_to_data[std::make_pair(weights, bias)].size())
            {
                NGRAPH_DEBUG << "number of input data param's doesnt match the number of matched "
                                "pattern root "
                             << "nodes";
                return;
            }
            auto& w_shape = weights->get_shape();
            if (w_shape.size() != 2)
            {
                NGRAPH_DEBUG << "weights shape for linear transformation of input is not 2D";
                return;
            }

            auto& data_param_nodes = it.second;
            // we will not fuse if the batch_size are not same across all inputs of time step
            for (auto& node : data_param_nodes)
            {
                if (shape_size(data_param_nodes[0]->get_shape()) != shape_size(node->get_shape()))
                {
                    return;
                }
            }
            // now concat the parameter hashed to the same weights
            auto concated_data = std::make_shared<op::Concat>(data_param_nodes, 0);

            auto& data_shape = concated_data->get_shape();
            auto data_order = ngraph::get_default_order(concated_data->get_shape());

            // insert reshape on the concated data to make it 2D, if its 3D
            std::shared_ptr<Node> input_reshape_node = nullptr;
            if (data_shape.size() == 3)
            {
                input_reshape_node = std::make_shared<op::Reshape>(
                    concated_data, data_order, Shape{data_shape[0] * data_shape[1], data_shape[2]});
            }
            auto new_input_node = data_shape.size() == 2 ? concated_data : input_reshape_node;
            NGRAPH_CHECK(new_input_node);
            auto w_reshape_node = std::make_shared<op::Reshape>(
                weights, AxisVector{1, 0}, Shape{w_shape[1], w_shape[0]});
            auto new_dot = std::make_shared<op::Dot>(new_input_node, w_reshape_node);
            auto bias_broadcast_node =
                std::make_shared<op::Broadcast>(bias, new_dot->get_shape(), AxisSet{0});
            auto new_add_bias = std::make_shared<op::Add>(new_dot, bias_broadcast_node);

            // now slice the new_add and feed the corrosponding root nodes
            auto batch_size = new_add_bias->get_shape()[0] / data_param_nodes.size();
            auto shape_axis_1 = new_add_bias->get_shape()[1];
            size_t start_index = 0;
            size_t end_index = batch_size;
            for (auto& matched_root_node : map_weights_to_pattern[weights])
            {
                std::shared_ptr<Node> slice_node = std::make_shared<op::Slice>(
                    new_add_bias, Coordinate{start_index, 0}, Coordinate{end_index, shape_axis_1});

                if (matched_root_node->get_shape().size() != 2)
                {
                    NGRAPH_CHECK(matched_root_node->get_shape().size() == 3);
                    slice_node = std::make_shared<op::Reshape>(
                        slice_node, AxisVector{0, 1}, matched_root_node->get_shape());
                }
                start_index += batch_size;
                end_index += batch_size;
                NGRAPH_DEBUG << "Replacing op " << matched_root_node->get_name() << " with "
                             << slice_node->get_name() << std::endl;
                function->replace_node(matched_root_node, slice_node);
            }
            modify_graph = true;
        }
    };

    auto callback_matcher_v1 = [&]() -> void {

        // Expecting input data shape D=[x, y, z], weights W=[u, v], bias B = [w]
        // where y is the time step. We are computing R=dot(D,W)=[x,y,v]. We can reshape D to
        // D'=[x*y, z], then we have dot(D',W), result
        // in R=[x*y, v], then add(R,B). We need to slice the result by strided by time steps.
        // iterate each unique set of parameters, replace original operations
        for (auto& p : param_list)
        {
            NodeVector params = p.first;
            NodeVector& op_nodes = p.second;

            // we will sort the captured Add(Dot(X, W) + B) as per the slice ordering of X
            // this will simplify the replace_node logic
            auto compare_slices = [&](const std::shared_ptr<Node> node1,
                                      const std::shared_ptr<Node> node2) {
                const auto node1_slice =
                    std::static_pointer_cast<op::Slice>(op_seg_map[node1].at(Type::DATA));

                const auto node2_slice =
                    std::static_pointer_cast<op::Slice>(op_seg_map[node2].at(Type::DATA));

                return (node1_slice->get_lower_bounds() < node2_slice->get_lower_bounds() &&
                        node1_slice->get_upper_bounds() < node2_slice->get_upper_bounds());
            };
            std::sort(op_nodes.begin(), op_nodes.end(), compare_slices);

            // we fuse all the data slices captured in the pattern to make bigger GEMM call
            auto fuse_data_slices = [&]() {
                NodeVector data_slices;
                for (auto& op : op_nodes)
                {
                    auto data_node = op_seg_map.at(op).at(Type::DATA);
                    data_slices.push_back(data_node);
                }
                return std::make_shared<op::Concat>(data_slices, 0);
            };
            auto data_node = op_nodes.size() > 1 ? fuse_data_slices() : params.at(Type::DATA);
            auto weights_node = params.at(Type::WEIGHTS);
            auto bias_node = params.at(Type::BIAS);
            auto& data_shape = data_node->get_shape();

            // construct new op nodes
            auto data_reshape_node =
                std::make_shared<op::Reshape>(data_node,
                                              AxisVector{0, 1, 2},
                                              Shape{data_shape[0] * data_shape[1], data_shape[2]});

            auto old_weights_reshape_node = op_seg_map.at(op_nodes.at(0)).at(Type::WEIGHTS);
            auto weights_reshape_node =
                old_weights_reshape_node->copy_with_new_inputs({weights_node});
            auto dot_node = std::make_shared<op::Dot>(data_reshape_node, weights_reshape_node);
            const auto& dot_shape = dot_node->get_shape();

            auto bias_broadcast_node =
                std::make_shared<op::Broadcast>(bias_node, dot_shape, AxisSet{0});
            auto add_node = std::make_shared<op::Add>(dot_node, bias_broadcast_node);
            const auto& add_shape = add_node->get_shape();

            size_t num_timesteps = op_nodes.size();
            size_t batch_size = add_shape[0] / num_timesteps;
            size_t feature_size = add_shape[1];
            // create a slice for each user of the dot op matching the original dot op's output
            for (size_t i = 0, start_index = 0; i < op_nodes.size(); i++, start_index += batch_size)
            {
                // calculate the lower and upper bounds for the slice of the new fused node
                // ((<x0 | x1..|xt>*W)+b), which will used to replace the nodes matched in the
                // pattern
                const Coordinate lower_bounds{start_index, 0};
                const Coordinate upper_bounds{start_index + batch_size, feature_size};

                auto slice_node = std::make_shared<op::Slice>(add_node, lower_bounds, upper_bounds);

                // replace old nodes
                function->replace_node(op_nodes[i], slice_node);
            }
            modify_graph = true;
        }
    };

    // Based the matched pattern, this callback's fuse the input across time steps and replaces with
    // single DOT operation <X0|X1|X2|..... Xt>*W
    callback_matcher_v2();
    callback_matcher_v1();
    return modify_graph;
}

#define TI(x) std::type_index(typeid(x))

// Moved set_or_check_if_same and fuse_group_convolution, fuse_batch_mat_mul_transpose
// to core pass batch_fusion
