/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
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

#include "cpu_collapse_dims.hpp"
#include <algorithm>
#include <iostream>
#include <numeric>
#include <unordered_set>
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/util.hpp"

using namespace ngraph;

struct CollapsedDims
{
    std::vector<size_t> output_shape;
    std::vector<bool> is_operated_axis;
    std::vector<size_t> axis_set;
    std::vector<size_t> input_shape;
};

// Fold and collapse axes of output_shape.
// Contiguous axes that are not being operated on can be collapsed.
// Contiguous axes that are being operated on are collapsed optionally.
// Skip size 1 dimensions.
static void collapse_dims(std::vector<size_t>& output_shape,
                          std::set<size_t> operated_axes,
                          struct CollapsedDims& cdims,
                          bool collapse_operated_axes)
{
    size_t collapse_size = 1;
    bool operated_axes_run = false;
    bool collapsing = false;
    for (int output_idx = static_cast<int>(output_shape.size()) - 1; output_idx >= 0; output_idx--)
    {
        auto is_operated_axis = operated_axes.count(output_idx) == 1;
        auto end_run = (operated_axes_run != is_operated_axis) ||
                       (is_operated_axis && !collapse_operated_axes);
        if (collapsing && end_run)
        {
            if (collapse_size != 1)
            {
                cdims.output_shape.push_back(collapse_size);
                cdims.is_operated_axis.push_back(operated_axes_run);
                collapse_size = 1;
            }
        }

        collapse_size *= output_shape[output_idx];
        operated_axes_run = is_operated_axis;
        collapsing = true;
    }
    // Last run
    if (collapse_size != 1)
    {
        cdims.output_shape.push_back(collapse_size);
        cdims.is_operated_axis.push_back(operated_axes_run);
    }
    std::reverse(cdims.output_shape.begin(), cdims.output_shape.end());
    std::reverse(cdims.is_operated_axis.begin(), cdims.is_operated_axis.end());

    for (size_t i = 0; i < cdims.is_operated_axis.size(); i++)
    {
        if (cdims.is_operated_axis[i])
        {
            cdims.axis_set.push_back(i);
        }
        else
        {
            cdims.input_shape.push_back(cdims.output_shape[i]);
        }
    }
}

bool runtime::cpu::pass::CPUCollapseDims::run_on_function(std::shared_ptr<ngraph::Function> f)
{
    bool replaced = false;
    for (auto n : f->get_ordered_ops())
    {
        if (std::dynamic_pointer_cast<op::Broadcast>(n))
        {
            auto node = std::dynamic_pointer_cast<op::Broadcast>(n).get();
            auto input_shape = node->get_argument(0)->get_shape();
            auto output_shape = node->get_shape();
            auto operated_axes = node->get_broadcast_axes();

            struct CollapsedDims cdims;

            collapse_dims(output_shape, operated_axes, cdims, true);

            if (cdims.axis_set.size() == 0)
            {
                // Null broadcast operation, replace with reshape
                AxisVector axis_order = ngraph::get_default_order(input_shape);
                auto reshape = std::make_shared<op::Reshape>(
                    node->get_argument(0), axis_order, n->get_shape());
                ngraph::replace_node(n, reshape);
                replaced = true;
            }
            else if (output_shape.size() != cdims.output_shape.size())
            {
                // Reshape arg to collapsed input_shape
                AxisVector input_axis_order = ngraph::get_default_order(input_shape);
                auto reshape_input = std::make_shared<op::Reshape>(
                    node->get_argument(0), input_axis_order, Shape(cdims.input_shape));

                auto broadcast = std::make_shared<op::Broadcast>(
                    reshape_input, Shape(cdims.output_shape), AxisSet(cdims.axis_set));

                // Reshape collapsed output to original output_shape
                AxisVector output_axis_order = ngraph::get_default_order(cdims.output_shape);
                auto reshape_output =
                    std::make_shared<op::Reshape>(broadcast, output_axis_order, output_shape);
                ngraph::replace_node(n, reshape_output);
                replaced = true;
            }

            if (replaced)
            {
                NGRAPH_DEBUG << "CollapseDims: Replaced broadcast " << input_shape << " "
                             << operated_axes << " " << output_shape << " with "
                             << Shape(cdims.input_shape) << " " << AxisSet(cdims.axis_set) << " "
                             << Shape(cdims.output_shape);
            }
        }
    }

    return replaced;
}
