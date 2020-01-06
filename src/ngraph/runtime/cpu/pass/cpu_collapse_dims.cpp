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

#include "cpu_collapse_dims.hpp"
#include <algorithm>
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/dot.hpp"
#include "ngraph/op/max.hpp"
#include "ngraph/op/min.hpp"
#include "ngraph/op/product.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/util.hpp"

using namespace ngraph;

struct CollapsedShape
{
    Shape fshape;        // Collapsed shape with operated axes
    Shape rshape;        // Collapsed shape without operated axes
    AxisVector axis_set; // operated axis in fshape
};

// Fold and collapse axes of shape.
// Contiguous axes that are not being operated on can be collapsed.
// Contiguous axes that are being operated on are collapsed optionally.
// E.g.,
// Shape{3, 3, 2}, AxisSet{0, 1} -> Shape{9, 2}, AxisSet{0}
// Shape{2, 4, 6, 6}, AxisSet{2, 3} -> Shape{8, 36}, AxisSet{1}
static void collapse_dims(std::vector<size_t>& shape,
                          std::set<size_t> operated_axes,
                          struct CollapsedShape& cshape,
                          bool skip_unit_size = true)
{
    size_t collapse_size = 1;
    bool operated_axes_run = false;
    std::vector<bool> fshape_operated_axis;
    bool collapsing = false;
    for (int output_idx = static_cast<int>(shape.size()) - 1; output_idx >= 0; output_idx--)
    {
        auto is_operated_axis = operated_axes.count(output_idx) == 1;
        auto end_run = (operated_axes_run != is_operated_axis);
        if (collapsing && end_run)
        {
            if (collapse_size != 1 || !skip_unit_size)
            {
                cshape.fshape.push_back(collapse_size);
                fshape_operated_axis.push_back(operated_axes_run);
                collapse_size = 1;
            }
        }

        collapse_size *= shape[output_idx];
        operated_axes_run = is_operated_axis;
        collapsing = true;
    }
    // Last run
    if (collapse_size != 1 || !skip_unit_size)
    {
        cshape.fshape.push_back(collapse_size);
        fshape_operated_axis.push_back(operated_axes_run);
    }
    std::reverse(cshape.fshape.begin(), cshape.fshape.end());
    std::reverse(fshape_operated_axis.begin(), fshape_operated_axis.end());

    for (size_t i = 0; i < fshape_operated_axis.size(); i++)
    {
        if (fshape_operated_axis[i])
        {
            cshape.axis_set.push_back(i);
        }
        else
        {
            cshape.rshape.push_back(cshape.fshape[i]);
        }
    }
}

static bool collapse_broadcast(std::shared_ptr<Node> n)
{
    bool replaced = false;
    auto node = std::static_pointer_cast<op::Broadcast>(n).get();
    auto input_shape = node->get_argument(0)->get_shape();
    auto output_shape = node->get_shape();
    auto operated_axes = node->get_broadcast_axes();

    struct CollapsedShape cshape;

    collapse_dims(output_shape, operated_axes, cshape);

    if (cshape.axis_set.size() == 0)
    {
        // Null broadcast operation, replace with reshape
        AxisVector axis_order = ngraph::get_default_order(input_shape);
        auto reshape =
            std::make_shared<op::Reshape>(node->get_argument(0), axis_order, n->get_shape());
        ngraph::replace_node(n, reshape);
        replaced = true;
    }
    else if (output_shape.size() != cshape.fshape.size())
    {
        // Reshape arg to collapsed input_shape
        AxisVector input_axis_order = ngraph::get_default_order(input_shape);
        auto reshape_input = std::make_shared<op::Reshape>(
            node->get_argument(0), input_axis_order, Shape(cshape.rshape));

        auto broadcast = std::make_shared<op::Broadcast>(
            reshape_input, Shape(cshape.fshape), AxisSet(cshape.axis_set));

        // Reshape collapsed output to original output_shape
        AxisVector output_axis_order = ngraph::get_default_order(cshape.fshape);
        auto reshape_output =
            std::make_shared<op::Reshape>(broadcast, output_axis_order, output_shape);
        ngraph::replace_node(n, reshape_output);
        replaced = true;
    }

    if (replaced)
    {
        NGRAPH_DEBUG << "CollapseDims: Replaced broadcast " << input_shape << " " << operated_axes
                     << " " << output_shape << " with " << Shape(cshape.rshape) << " "
                     << AxisSet(cshape.axis_set) << " " << Shape(cshape.fshape);
    }
    return replaced;
}

template <typename T>
static bool collapse_reduction(std::shared_ptr<Node> n)
{
    bool replaced = false;
    auto node = std::static_pointer_cast<T>(n).get();
    auto input_shape = node->get_argument(0)->get_shape();
    auto output_shape = node->get_shape();
    auto operated_axes = node->get_reduction_axes();

    struct CollapsedShape cshape;

    collapse_dims(input_shape, operated_axes, cshape);

    if (cshape.axis_set.size() == 0)
    {
        // Null reduction operation
        AxisVector axis_order = ngraph::get_default_order(input_shape);
        auto reshape =
            std::make_shared<op::Reshape>(node->get_argument(0), axis_order, n->get_shape());
        ngraph::replace_node(n, reshape);
        replaced = true;
    }
    else if (input_shape.size() != cshape.fshape.size())
    {
        // Reshape arg to collapsed input_shape
        AxisVector input_axis_order = ngraph::get_default_order(input_shape);
        auto reshape_input = std::make_shared<op::Reshape>(
            node->get_argument(0), input_axis_order, Shape(cshape.fshape));

        auto reduction = std::make_shared<T>(reshape_input, AxisSet(cshape.axis_set));

        // Reshape collapsed output to original output_shape
        AxisVector output_axis_order = ngraph::get_default_order(cshape.rshape);
        auto reshape_output =
            std::make_shared<op::Reshape>(reduction, output_axis_order, output_shape);
        ngraph::replace_node(n, reshape_output);
        replaced = true;
    }

    if (replaced)
    {
        NGRAPH_DEBUG << "CollapseDims: Replaced arithmetic reduction " << input_shape << " "
                     << operated_axes << " " << output_shape << " with " << Shape(cshape.fshape)
                     << " " << AxisSet(cshape.axis_set) << " " << Shape(cshape.rshape);
    }
    return replaced;
}

template <typename T>
static bool collapse_dot(std::shared_ptr<Node> n)
{
    bool replaced = false;
    auto node = std::static_pointer_cast<T>(n).get();
    auto A_shape = node->get_argument(0)->get_shape();
    auto B_shape = node->get_argument(1)->get_shape();
    auto reduction_count = node->get_reduction_axes_count();

    AxisSet operated_axes_A, operated_axes_B;

    for (size_t i = 0; i < reduction_count; i++)
    {
        operated_axes_A.insert(A_shape.size() - i - 1);
        operated_axes_B.insert(i);
    }

    struct CollapsedShape cshape_A, cshape_B;
    collapse_dims(A_shape, operated_axes_A, cshape_A, false);
    collapse_dims(B_shape, operated_axes_B, cshape_B, false);

    if (A_shape != cshape_A.fshape || B_shape != cshape_B.fshape)
    {
        // Reshape A to cshape_A.fshape
        AxisVector A_axis_order = ngraph::get_default_order(A_shape);
        auto reshape_A = std::make_shared<op::Reshape>(
            node->get_argument(0), A_axis_order, Shape(cshape_A.fshape));

        // Reshape B to cshape_B.fshape
        AxisVector B_axis_order = ngraph::get_default_order(B_shape);
        auto reshape_B = std::make_shared<op::Reshape>(
            node->get_argument(1), B_axis_order, Shape(cshape_B.fshape));

        auto cdot =
            std::make_shared<op::Dot>(reshape_A, reshape_B, reduction_count ? 1 : reduction_count);
        auto reshape_output = std::make_shared<op::Reshape>(
            cdot, ngraph::get_default_order(cdot->get_shape()), node->get_shape());
        ngraph::replace_node(n, reshape_output);

        NGRAPH_DEBUG << "CollapseDims: Replaced dot " << A_shape << " . " << B_shape
                     << " reduction count: " << reduction_count << " with "
                     << Shape(cshape_A.fshape) << " . " << Shape(cshape_B.fshape);

        replaced = true;
    }

    return replaced;
}

bool runtime::cpu::pass::CPUCollapseDims::run_on_function(std::shared_ptr<ngraph::Function> f)
{
    bool replaced = false;
    for (auto n : f->get_ordered_ops())
    {
        if (is_type<op::Broadcast>(n))
        {
            replaced |= collapse_broadcast(n);
        }
        else if (is_type<op::Max>(n))
        {
            replaced |= collapse_reduction<op::Max>(n);
        }
        else if (is_type<op::Min>(n))
        {
            replaced |= collapse_reduction<op::Min>(n);
        }
        else if (is_type<op::Product>(n))
        {
            replaced |= collapse_reduction<op::Product>(n);
        }
        else if (is_type<op::Sum>(n))
        {
            replaced |= collapse_reduction<op::Sum>(n);
        }
        else if (is_type<op::Dot>(n))
        {
            replaced |= collapse_dot<op::Dot>(n);
        }
    }

    return replaced;
}
