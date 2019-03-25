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

#include "ngraph/pass/shape_specialization.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/op/constant.hpp"

using namespace ngraph;

//
// The shape specialization pass transforms a function by replacing, wherever possible, all
// shape-relevant inputs with constants. For example,
//
//
//          _________
//         | Param   |
//         | 2x2x3   |
//         |_________|
//              |
//          ____|____
//         |    0    |
//         | ShapeOf |
//         |_________|
//      |     |
//    __|_____|___
//   |  0     1   |
//   | DynReshape |
//   |____________|
//         |
//
// (Where 0 is the data input and 1 is the shape input) would be replaced with:
//
//          _____________
//         |             |
//         |   Constant  |
//         | val=[2,2,3] |
//         |_____________|
//      |     |
//    __|_____|___
//   |  0     1   |
//   | DynReshape |
//   |____________|
//         |
//
// Note that replacement will only be attempted on shape-relevant inputs, and will only be
// successful if the input's value is entirely determined by nodes that can be converted with
// as_constants().
//
bool pass::ShapeSpecialization::run_on_function(std::shared_ptr<Function> f)
{
    // TODO(amprocte): We are probably reinventing the wheel with the graph traversal here; the
    // reason is that we need to cut the traversal short in cases where input values are
    // irrelevant. See if there is a way to reduce this duplication.

    // Set of nodes that must be evaluated to determine the value of shape-relevant inputs.
    std::set<Node*> shape_determinants;

    // Step 1: Find root nodes (these are nodes with an output connected to a shape-relevant
    // input).
    for (auto& n : f->get_ops())
    {
        for (auto& output : n->get_outputs())
        {
            for (auto& input : output.get_inputs())
            {
                if (input->get_is_relevant_to_shape())
                {
                    shape_determinants.insert(n.get());
                    break;
                }
            }
        }
    }

    // Step 2: Find all shape determinants. This is the transitive closure of R, where n1 R n2
    // iff there is a data flow edge from n2 to n1 and that data flow edge is not
    // value-irrelevant.
    {
        std::list<Node*> to_visit{shape_determinants.begin(), shape_determinants.end()};
        std::set<Node*> already_visited;

        while (!to_visit.empty())
        {
            auto node = to_visit.front();
            to_visit.pop_front();

            if (already_visited.count(node) > 0)
            {
                continue;
            }

            shape_determinants.insert(node);
            already_visited.insert(node);

            for (auto& input : node->get_inputs())
            {
                if (!input.get_is_relevant_to_value())
                {
                    continue;
                }
                auto source_node = input.get_output().get_node().get();
                if (already_visited.count(source_node) == 0)
                {
                    to_visit.push_front(source_node);
                }
            }
        }
    }

    // Step 3: For each shape determinant in topological order, try to replace the determinant
    // with constants.
    bool changes_made = false;

    for (auto n : f->get_ordered_ops())
    {
        if (shape_determinants.count(n.get()) > 0)
        {
            std::vector<std::shared_ptr<op::Constant>> replacement_constants = n->as_constants();
            if (replacement_constants.size() > 0)
            {
                NGRAPH_ASSERT(n->get_output_size() == replacement_constants.size());

                for (size_t i = 0; i < n->get_output_size(); i++)
                {
                    NGRAPH_ASSERT(n->get_output_partial_shape(i).relaxes(
                        replacement_constants[i]->get_output_partial_shape(0)));
                    NGRAPH_ASSERT(n->get_output_element_type(i).is_dynamic() ||
                                  n->get_output_element_type(i) ==
                                      replacement_constants[i]->get_output_element_type(0));

                    auto& replacement_output = replacement_constants.at(i)->get_outputs().at(0);
                    auto& output = n->get_outputs().at(i);
                    auto inputs_copy = output.get_inputs();
                    for (auto& input : inputs_copy)
                    {
                        input->replace_output(replacement_output);
                        changes_made = true;
                    }
                }
            }
        }
    }

    return changes_made;
}
