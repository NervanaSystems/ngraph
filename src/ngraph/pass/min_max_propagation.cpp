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

#include "ngraph/pass/min_max_propagation.hpp"
#include "ngraph/graph_util.hpp"

using namespace ngraph;

bool pass::MinMaxShapePropagation::run_on_function(std::shared_ptr<Function> f)
{
    bool is_max_output_shape_set = false;
    PartialShape max_output_shape;
    for (auto& node : f->get_ordered_ops())
    {
        if (node->is_parameter() || node->is_constant())
        {
            continue;
        }

        // Check if the output shape of the node is dynamic
        if (node->get_output_partial_shape(0).is_dynamic())
        {
            // if max shape is set for the output shape, propagate the max.
            if (!node->output(0).get_max_partial_shape().is_dynamic())
            {
                is_max_output_shape_set = true;
                max_output_shape = node->output(0).get_max_partial_shape();
                node->set_output_max_partial_shape(0, max_output_shape);
                // Propagate the shape
                f->validate_nodes_and_infer_types();
            }
        }
        else if (is_max_output_shape_set) // propagate the max ?
        {
            node->output(0).set_max_partial_shape(max_output_shape);
        }
    }
    return true;
}
