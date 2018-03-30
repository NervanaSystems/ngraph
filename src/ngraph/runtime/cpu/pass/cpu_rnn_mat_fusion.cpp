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

struct Type
{
    enum
    {
        DATA = 0,
        WEIGHTS,
        BIAS,
    };
};

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
    bool modified = false;

    auto data_pred = [](std::shared_ptr<Node> n) {
        return std::dynamic_pointer_cast<op::Slice>(n) != nullptr;
    };
    auto data_slice = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 2, 4}, data_pred);
    auto data_pattern = construct_data_pattern(data_slice);

    auto weights_pred = [](std::shared_ptr<Node> n) {
        return std::dynamic_pointer_cast<op::Reshape>(n) != nullptr;
    };
    auto weights_reshape =
        std::make_shared<pattern::op::Label>(element::f32, Shape{4, 1}, weights_pred);
    auto weights_pattern = construct_weights_pattern(weights_reshape);

    //we don't really need a broadcast node but
    //labelling a Broadcast allows us to extract
    //params from all 3 labels in the same fashion
    //(i.e. via get_input_op(0))
    auto broadcast_pred = [](std::shared_ptr<Node> n) {
        return std::dynamic_pointer_cast<op::Broadcast>(n) != nullptr;
    };
    auto bias_broadcast =
        std::make_shared<pattern::op::Label>(element::f32, Shape{2, 1}, broadcast_pred);
    auto bias_pattern = construct_bias_pattern(bias_broadcast);

    const size_t NUM_MMB_ARGS = 3;
    std::shared_ptr<pattern::op::Label> labels[] = {data_slice, weights_reshape, bias_broadcast};
    //Matchers' ordering is important! Don't change!
    std::shared_ptr<pattern::Matcher> matchers[] = {
        std::make_shared<pattern::Matcher>(data_pattern),
        std::make_shared<pattern::Matcher>(weights_pattern),
        std::make_shared<pattern::Matcher>(bias_pattern)};

    std::map<std::shared_ptr<Node>, NodeVector> op_seg_map; //add to list of params
    std::map<NodeVector, NodeVector> param_list;
    for (auto n : function->get_ordered_ops())
    {
        NodeVector params;
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
                params.push_back(matched->get_input_op(0));
                matched_nodes.push_back(matched);
            }

            if (params.size() != NUM_MMB_ARGS)
            {
                continue;
            }

            //we have a full set for the current Add (n) i.e. data, weights, bias
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

    // Expecting input data shape D=[x, y, z], weights W=[u, v], bias B = [w]
    // where y is the time step. We are computing R=dot(D,W)=[x,y,v]. We can reshape D to D'=[x*y, z], then we have dot(D',W), result
    // in R=[x*y, v], then add(R,B). We need to slice the result by strided by time steps.
    // iterate each unique set of parameters, replace original operations
    for (auto& p : param_list)
    {
        NodeVector params = p.first;
        NodeVector& op_nodes = p.second;

        auto data_node = params.at(Type::DATA);
        auto weights_node = params.at(Type::WEIGHTS);
        auto bias_node = params.at(Type::BIAS);

        const auto& data_shape = data_node->get_shape();
        // construct new op nodes
        AxisVector data_order(data_node->get_shape().size());
        std::iota(begin(data_order), end(data_order), 0);
        auto data_reshape_node = std::make_shared<op::Reshape>(
            data_node, data_order, Shape{data_shape[0] * data_shape[1], data_shape[2]});

        auto old_weights_reshape_node = op_seg_map.at(op_nodes.at(0)).at(Type::WEIGHTS);
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
            const auto old_slice =
                std::dynamic_pointer_cast<op::Slice>(op_seg_map[op].at(Type::DATA));
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
