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

#include "ngraph/op/tensor_iterator.hpp"
#include "ngraph/graph_util.hpp"

using namespace std;
using namespace ngraph;

const string op::TensorIterator::type_name{"TensorIterator"};

void op::TensorIterator::validate_and_infer_types()
{
    NODE_VALIDATION_CHECK(this,
                          get_input_size() == m_body_parameters.size(),
                          "Number of inputs must be the same as number of body parameters");

    // The number of iterations is determined by the shortest sequence input
    size_t iteration_count{0};
    // If true, iteration count is dynamic
    bool iteration_count_dynamic{false};
    // true when we know something about the count
    bool iteration_count_valid{false};
    for (auto input : inputs())
    {
        size_t input_index = input.get_index();
        Output<Node> value = input.get_source_output();
        PartialShape sequence_shape = value.get_partial_shape();
        PartialShape iterator_shape = sequence_shape;
        Rank sequence_rank = sequence_shape.rank();

        if (sequence_rank.is_dynamic())
        {
            // Can't determine the sequence length
            iteration_count_dynamic = true;
        }
        else
        {
            NODE_VALIDATION_CHECK(this,
                                  static_cast<size_t>(sequence_shape.rank()) != 0,
                                  "Input ",
                                  input_index,
                                  " is specified to be a sequence but is scalar.");
            Dimension sequence_dim = sequence_shape[0];
            vector<Dimension> dimensions = static_cast<vector<Dimension>>(sequence_shape);
            dimensions.erase(dimensions.begin());
            iterator_shape = PartialShape(dimensions);

            if (sequence_dim.is_dynamic())
            {
                // Can't determine the sequence length
                iteration_count_dynamic = true;
            }
            else
            {
                size_t sequence_length = static_cast<size_t>(sequence_dim);
                if (!iteration_count_valid || (sequence_length < iteration_count))
                {
                    iteration_count = sequence_length;
                    iteration_count_valid = true;
                }
            }
        }

        NODE_VALIDATION_CHECK(
            this,
            iterator_shape.compatible(m_body_parameters.at(input_index)->get_partial_shape()),
            "Iterator body param is not compatible with value");
    }
    // The body may depend on the body parameters as well as values from outside the body
    // Body parameters depend on the loop initialization
    NodeVector body_result_nodes;
    for (auto& body_output : m_body_outputs)
    {
        body_result_nodes.push_back(body_output.get_node_shared_ptr());
    }
    std::list<std::shared_ptr<Node>> body_node_closure(topological_sort(body_result_nodes, true));
    std::set<Node*> bound_nodes;
    std::vector<Node*> free_nodes;
    for (auto& parameter : m_body_parameters)
    {
        std::cerr << *this << " Bound: " << *parameter << std::endl;
        bound_nodes.insert(parameter.get());
    }
    for (auto& node : body_node_closure)
    {
        if (bound_nodes.find(node.get()) == bound_nodes.end())
        {
            bool is_free = true;
            for (auto input : node->inputs())
            {
                auto input_node = input.get_source_output().get_node();
                if (bound_nodes.find(input_node) != bound_nodes.end())
                {
                    bound_nodes.insert(node.get());
                    is_free = false;
                    std::cerr << *this << " Bound: "
                              << " : " << *node << std::endl;
                    break;
                }
            }
            if (is_free)
            {
                free_nodes.push_back(node.get());
                std::cout << *this << " Free: " << *node << std::endl;
            }
        }
    }
}

std::shared_ptr<Node> op::TensorIterator::copy_with_new_args(const NodeVector& new_args) const
{
    // This would be used for cloning/splicing, so the new args are replacements for the
    // args that get set up during body/iteration specification.
}
