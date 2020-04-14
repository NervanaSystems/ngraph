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

#include <functional>
#include <memory>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>

#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/convert.hpp"
#include "ngraph/op/experimental/shape_of.hpp"
#include "ngraph/op/fused/squeeze.hpp"
#include "ngraph/op/fused/unsqueeze.hpp"
#include "ngraph/op/non_zero.hpp"
#include "ngraph/op/pad.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/op/stop_gradient.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/util.hpp"
#include "nop_elimination.hpp"

using namespace std;
using namespace ngraph;

#define TI(x) x::type_info

static bool eliminate_pad(const std::shared_ptr<Node>& node)
{
    auto pad = as_type_ptr<op::v0::Pad>(node);
    // skip if shapes are dynamic
    if (pad->get_input_partial_shape(0).is_dynamic() ||
        pad->get_output_partial_shape(0).is_dynamic())
    {
        return false;
    }

    if (pad->get_input_shape(0) == pad->get_output_shape(0))
    {
        return remove_node_update_name(node, node->get_argument(0));
    }
    return false;
}

static bool eliminate_sum(const std::shared_ptr<Node>& node)
{
    auto sum = as_type_ptr<op::v0::Sum>(node);
    if (sum->get_reduction_axes().empty())
    {
        return remove_node_update_name(node, node->get_argument(0));
    }
    return false;
}
/****************************************
 *
 *   Convert elimination use cases:
 *   Let the candidate Convert node for elimination
 *   be Convert1 in the following sub-graph:
 *
//            [ Input_node ]
//            /      |      \
//           /       |       \
//          |        u3       |
//    [ Convert1 ]            u4
//    /   |     \
//   /    |      \
//  |     |       |
//  u1    |       |
//        u2      |
//                { r1 }
//
 *
 * Then the elimination logic covers following use cases:
 *     // Vertical elimination 1
 *     1. If all users of Convert1 are of type op::Convert &&
 *                    m_destination_type of all users are equal:
 *             replace Convert1 with one of the users and
 *             attach other users' users to it
 *     // Vertical elimination 2
 *     2. Else if Input_node::element_type == Convert1::m_destination_type:
 *             replace Convert1 with Input_node
 *     // Vertical elimination 3
 *     3. Else if all users of Convert1 are type_agnostic ( eg. op: Non_Zero)
 *             replace Convert with Input_node
 *     // Horizontal elimination 1
 *     4. Else if some users (same_users) of Input_node are of type op::Convert &&
 *                      m_destination_type of all users are equal:
 *             choose 1 of same_users as u1 and move users of all others in same_users under u1
 *     else return false
 *
 *****************************************/
static bool eliminate_convert(const std::shared_ptr<Node>& node)
{
    auto convert = as_type_ptr<op::v0::Convert>(node);
    auto destination_type = convert->get_destination_type();
    static const std::set<NodeTypeInfo> type_agnostic{TI(op::v3::NonZero)};
    //    size_t user_count = node->output(0).get_target_inputs().size();

    // vertical elimination 1
    // TBD

    // vertical elimination 2 & 3
    bool users_are_type_agnostic = true;
    for (auto& user : node->output(0).get_target_inputs())
    {
        if (!type_agnostic.count(user.get_node()->get_type_info()))
        {
            users_are_type_agnostic = false;
            break;
        }
    }
    if (destination_type == convert->get_argument(0)->get_element_type() || users_are_type_agnostic)
    {
        return remove_node_update_name(node, convert->get_argument(0));
    }
    return false;
}

static bool eliminate_slice(const std::shared_ptr<Node>& node)
{
    auto slice = as_type_ptr<op::v0::Slice>(node);
    // skip if shapes are dynamic
    if (slice->get_input_partial_shape(0).is_dynamic() ||
        slice->get_output_partial_shape(0).is_dynamic())
    {
        return false;
    }
    if (slice->get_input_shape(0) == slice->get_output_shape(0))
    {
        return remove_node_update_name(node, node->get_argument(0));
    }
    return false;
}

static bool eliminate_broadcast(const std::shared_ptr<Node>& node)
{
    auto broadcast = as_type_ptr<op::v0::Broadcast>(node);
    // skip if shapes are dynamic
    if (broadcast->get_input_partial_shape(0).is_dynamic() ||
        broadcast->get_output_partial_shape(0).is_dynamic())
    {
        return false;
    }
    if (broadcast->get_input_shape(0) == broadcast->get_output_shape(0))
    {
        return remove_node_update_name(node, node->get_argument(0));
    }
    return false;
}

static bool eliminate_concat(const std::shared_ptr<Node>& node)
{
    auto node_input = node->get_argument(0);

    // remove concat with single input
    if (node->get_input_size() == 1)
    {
        return remove_node_update_name(node, node_input);
    }
    return false;
}

static bool eliminate_reshape_v1(const std::shared_ptr<Node>& node)
{
    auto node_input = node->get_argument(0);
    // check if reshape is not identity op
    if (node_input->get_output_partial_shape(0).is_dynamic() ||
        node->get_output_partial_shape(0).is_dynamic() ||
        node_input->get_output_shape(0) != node->get_output_shape(0))
    {
        NGRAPH_DEBUG << "Not a no-op; Shapes are different!";
        return false;
    }
    return remove_node_update_name(node, node_input);
}

static bool eliminate_unsqueeze(const std::shared_ptr<Node>& node)
{
    auto unsqueeze = as_type_ptr<op::v0::Unsqueeze>(node);
    auto input = unsqueeze->get_argument(0);
    // eliminate redundant squeeze->unsqueeze
    if (auto squeeze = as_type_ptr<op::v0::Squeeze>(input))
    {
        if (!ngraph::compare_constants(squeeze->get_argument(1), unsqueeze->get_argument(1)))
        {
            NGRAPH_DEBUG << "squeeze->unsqueeze axes do not match";
            return false;
        }
        return remove_node_update_name(unsqueeze, squeeze->get_argument(0));
    }
    return false;
}

static bool eliminate_squeeze(const std::shared_ptr<Node>& node)
{
    auto squeeze = as_type_ptr<op::v0::Squeeze>(node);
    auto input = squeeze->get_argument(0);
    // eliminate redundant unsqueeze->squeeze
    if (auto unsqueeze = as_type_ptr<op::v0::Unsqueeze>(input))
    {
        if (!ngraph::compare_constants(unsqueeze->get_argument(1), squeeze->get_argument(1)))
        {
            NGRAPH_DEBUG << "unsqueeze->squeeze axes do not match";
            return false;
        }
        return remove_node_update_name(squeeze, unsqueeze->get_argument(0));
    }
    return false;
}

static bool eliminate_stop_gradient(const std::shared_ptr<Node>& node)
{
    remove_node_update_name(node, node->get_argument(0));
    return true;
}

static const std::unordered_map<NodeTypeInfo, std::function<bool(const std::shared_ptr<Node>&)>>
    dispatcher{{TI(op::v0::Pad), &eliminate_pad},
               {TI(op::v0::Sum), &eliminate_sum},
               {TI(op::v0::Convert), &eliminate_convert},
               {TI(op::v0::Slice), &eliminate_slice},
               {TI(op::v0::StopGradient), &eliminate_stop_gradient},
               {TI(op::v1::Reshape), &eliminate_reshape_v1},
               {TI(op::v0::Concat), &eliminate_concat},
               {TI(op::v0::Squeeze), &eliminate_squeeze},
               {TI(op::v0::Unsqueeze), &eliminate_unsqueeze},
               {TI(op::v0::Broadcast), &eliminate_broadcast}};

bool pass::NopElimination::run_on_function(std::shared_ptr<Function> function)
{
    bool clobbered = false;

    for (const auto& n : function->get_ops())
    {
        // Work around a warning [-Wpotentially-evaluated-expression]
        const Node& node = *n;
        auto handler = dispatcher.find(node.get_type_info());
        if (handler != dispatcher.end())
        {
            clobbered = handler->second(n) || clobbered;
        }
    }

    return clobbered;
}
