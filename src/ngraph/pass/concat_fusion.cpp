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

#include "concat_fusion.hpp"
#include <algorithm>
#include <iostream>
#include <numeric>
#include <unordered_set>

#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/pattern/matcher.hpp"
#include "ngraph/pattern/op/label.hpp"
#include "ngraph/pattern/op/skip.hpp"
#include "ngraph/util.hpp"

using namespace ngraph;

bool check_self_concat_op(const std::shared_ptr<Node>& op)
{
    bool is_self_concat = true;
    auto arg = op->get_argument(0);
    for (size_t i = 1; i < op->get_input_size(); i++)
    {
        if (op->get_argument(i) != arg)
        {
            is_self_concat = false;
            break;
        }
    }

    return is_self_concat;
}

bool check_concat_axis_dim_value(const std::shared_ptr<Node>& concat_op)
{
    auto input_shape = concat_op->get_input_shape(0);
    size_t concat_axis = std::static_pointer_cast<op::Concat>(concat_op)->get_concatenation_axis();

    return (input_shape[concat_axis] == 1);
}

bool check_concat_has_no_fan_out(const std::shared_ptr<Node>& op)
{
    auto users = op->get_users();
    std::set<std::shared_ptr<Node>> user_set(users.begin(), users.end());
    // std::cout << op->get_name() << " : " << users.size() << " : " << user_set.size() << std::endl;
    size_t num_unique_users = user_set.size();
    return (num_unique_users == 1);
}

bool valid_self_concat(const std::shared_ptr<Node>& Op)
{
    if (!(std::dynamic_pointer_cast<op::Concat>(Op)))
    {
        return false;
    }

    if (!check_self_concat_op(Op))
    {
        NGRAPH_DEBUG << "Not a self concat";                         //TODO
        std::cout << "NGRAPH_DEBUG: Not a self concat" << std::endl; //TODO
        return false;
    }

    if (!check_concat_axis_dim_value(Op))
    {
        NGRAPH_DEBUG << "Not a self concat with dim 1";                         // TODO
        std::cout << "NGRAPH_DEBUG: Not a self concat with dim 1" << std::endl; // TODO
        return false;
    }

    if (!check_concat_has_no_fan_out(Op))
    {
        NGRAPH_DEBUG << "Self concat has fan out";                         // TODO
        std::cout << "NGRAPH_DEBUG: Self concat has fan out" << std::endl; // TODO
        return false;
    }

    return true;
}

bool check_source_is_concat(const std::shared_ptr<Node>& concat_op,
                            const std::shared_ptr<Node>& source_of_concat_op)
{
    bool is_source_concat = true;
    for (size_t i = 0; i < concat_op->get_input_size(); i++)
    {
        if (concat_op->get_argument(i) != source_of_concat_op)
        {
            is_source_concat = false;
            std::cout << "NGRAPH_DEBUG inside function: source is not concat "
                      << concat_op->get_name() << " trying to match source "
                      << source_of_concat_op->get_name() << std::endl;
            break;
        }
    }

    return is_source_concat;
}

void pass::ConcatElimination::construct_concat_elimination()
{
    auto op_label = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 3});
    auto concat = std::make_shared<op::Concat>(NodeVector{op_label}, 0);
    auto concat_label = std::make_shared<pattern::op::Label>(concat, nullptr, NodeVector{concat});

    auto callback = [op_label](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for construct_concat_elimination against node = "
                     << m.get_match_root()->get_name();
        auto pattern_map = m.get_pattern_map();
        auto op = pattern_map[op_label];

        auto root = std::dynamic_pointer_cast<op::Concat>(m.get_match_root());
        // std::cout << "Input Shape: " << join(root->get_input_shape(0))
        //           << " Output Shape: " << join(root->get_output_shape(0)) << std::endl;
        if (root && (root->get_input_shape(0) == root->get_output_shape(0)))
        {
            replace_node(m.get_match_root(), op);
            return true;
        }

        return false;
    };

    auto m = std::make_shared<pattern::Matcher>(concat_label, callback);
    this->add_matcher(m);
}

bool ngraph::pass::SelfConcatFusion::run_on_function(std::shared_ptr<Function> function)
{
    bool modify_graph = false;
    auto has_multiple_inputs = [](std::shared_ptr<Node> n) {
        auto input_size = n->get_input_size();
        auto root = std::dynamic_pointer_cast<op::Concat>(n);
        return (root && input_size > 1);
    };

    auto print_state_of_bounded_vectors = [this]() {
        std::cout << "STATE of BOUNDED VECTORS: " << std::endl;
        std::cout << "------------------------" << std::endl;
        std::cout << "Number of vectors: " << this->m_concat_pattern_vectors.size() << std::endl;
        size_t c = 0;
        for (auto iter : this->m_concat_pattern_vectors)
        {
            std::cout << "For vector " << c << std::endl;
            auto iter_node_vec = iter.first;
            auto iter_concat_axis = iter.second;
            for (auto it : iter_node_vec)
            {
                std::cout << it->get_name() << " ";
            }
            std::cout << std::endl;
            for (auto it : iter_concat_axis)
            {
                std::cout << it << ", ";
            }
            std::cout << std::endl;
            c++;
        }
        std::cout << "------------------------" << std::endl;
    };

    auto concat_op_label =
        std::make_shared<pattern::op::Label>(element::f32, Shape{1, 3}, has_multiple_inputs);
    auto matcher = std::make_shared<pattern::Matcher>(concat_op_label);
    for (auto n : function->get_ordered_ops())
    {
        if (matcher->match(n))
        {
            auto concat_op = matcher->get_pattern_map()[concat_op_label];
            if (!std::dynamic_pointer_cast<op::Concat>(concat_op))
            {
                NGRAPH_DEBUG << "Incorrect Match";
                continue;
            }
            std::cout << concat_op->get_name() << std::endl;
            if (!valid_self_concat(concat_op))
            {
                std::cout << "NGRAPH_DEBUG: " << concat_op->get_name()
                          << " is not a valid self concat" << std::endl;
                continue;
            }
            else
            {
                std::cout << "NGRAPH_DEBUG: " << concat_op->get_name() << " is a VALID self concat"
                          << std::endl;
            }

            auto& concat_vectors = this->m_concat_pattern_vectors;
            size_t concat_axis =
                std::static_pointer_cast<op::Concat>(concat_op)->get_concatenation_axis();
            if (concat_vectors.empty())
            {
                concat_vectors.push_back(
                    make_pair(NodeVector{concat_op}, std::vector<size_t>{concat_axis}));
                print_state_of_bounded_vectors();
            }
            else
            {
                auto& last_concat = concat_vectors.back().first.back();
                std::cout << "NGRAPH_DEBUG: " << concat_op->get_name() << " trying to match source "
                          << last_concat->get_name() << std::endl;
                if (check_source_is_concat(concat_op, last_concat))
                {
                    std::cout << "MATCHED SOURCE" << concat_op->get_name() << " and "
                              << last_concat->get_name() << std::endl;
                    concat_vectors.back().first.push_back(concat_op);
                    concat_vectors.back().second.push_back(concat_axis);
                    print_state_of_bounded_vectors();
                }
                else
                {
                    std::cout << "COULD NOT MATCH SOURCE" << concat_op->get_name() << " and "
                              << last_concat->get_name() << std::endl;
                    concat_vectors.push_back(
                        make_pair(NodeVector{concat_op}, std::vector<size_t>{concat_axis}));
                    print_state_of_bounded_vectors();
                }
            }
        }
    }

    print_state_of_bounded_vectors();

    // Remove the elements of concat_vetors with size = 1; Only fuse concats when there are more than 1 self concats in a row
    auto scalarize_dim = [](std::vector<size_t> concat_axis_vector,
                            const Shape& input_shape) -> Shape {

        std::cout << "Concat_axis_vetor: " << join(concat_axis_vector) << std::endl;
        std::cout << "Input Shape: " << join(input_shape) << std::endl;
        Shape scalarized_shape;
        for (size_t i = 0; i < input_shape.size(); i++)
        {
            auto it = std::find(concat_axis_vector.begin(), concat_axis_vector.end(), i);
            if (it == concat_axis_vector.end())
            {
                scalarized_shape.push_back(input_shape[i]);
            }
        }
        std::cout << "scalarized_shape: " << join(scalarized_shape) << std::endl;
        return scalarized_shape;
    };
    for (auto concat_op_pair : this->m_concat_pattern_vectors)
    {
        auto bounded_concat_ops = concat_op_pair.first;
        auto concat_axis_vector = concat_op_pair.second;

        auto& first_bounded_concat = (*bounded_concat_ops.begin());
        auto driver_op = first_bounded_concat->get_argument(0);
        std::cout << driver_op->get_name() << std::endl;
        const Shape& input_shape = first_bounded_concat->get_input_shape(0);

        auto scalarized_shape = scalarize_dim(concat_axis_vector, input_shape);
        std::cout << "scalarized_shape outside: " << join(scalarized_shape) << std::endl;
        AxisVector axis_order = get_default_order(input_shape);
        std::cout << "Default axis order: " << join(axis_order) << std::endl;
        auto reshape = std::make_shared<op::Reshape>(driver_op, axis_order, scalarized_shape);
        auto last_bounded_concat_op = bounded_concat_ops.back();
        auto broadcast_out_shape = last_bounded_concat_op->get_shape();
        std::cout << "Broadcast out shape: " << join(broadcast_out_shape) << std::endl;
        auto broadcast =
            std::make_shared<op::Broadcast>(reshape, broadcast_out_shape, concat_axis_vector);

        replace_node(last_bounded_concat_op, broadcast);
        modify_graph = true;
    }

    return modify_graph;
}
