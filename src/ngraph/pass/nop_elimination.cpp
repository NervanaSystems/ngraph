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

#include "ngraph/log.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/convert.hpp"
#include "ngraph/op/experimental/shape_of.hpp"
#include "ngraph/op/pad.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/op/stop_gradient.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/util.hpp"
#include "nop_elimination.hpp"

using namespace std;
using namespace ngraph;

#define TI(x) std::type_index(typeid(x))

static bool remove_update_name(const std::shared_ptr<Node>& node,
                               const std::shared_ptr<Node>& node_input)
{
    bool has_result_output = false;
    for (auto& output : node->output(0).get_target_inputs())
    {
        if (dynamic_cast<op::Result*>(output.get_node()))
        {
            has_result_output = true;
        }
    }
    // ignore trivial elimination
    if (has_result_output && as_type_ptr<ngraph::op::Parameter>(node_input))
    {
        return false;
    }
    if (!has_result_output || node_input->get_users().size() == 1)
    {
        node_input->set_friendly_name(node->get_friendly_name());
        replace_node(node, node_input);
        return true;
    }
    return false;
}

static bool eliminate_pad(const std::shared_ptr<Node>& node)
{
    auto pad = std::static_pointer_cast<op::v0::Pad>(node);
    if (pad->get_input_shape(0) == pad->get_output_shape(0))
    {
        return remove_update_name(node, node->get_argument(0));
    }
    return false;
}

static bool eliminate_sum(const std::shared_ptr<Node>& node)
{
    auto sum = std::static_pointer_cast<op::Sum>(node);
    if (sum->get_reduction_axes().empty())
    {
        return remove_update_name(node, node->get_argument(0));
    }
    return false;
}

static bool eliminate_convert(const std::shared_ptr<Node>& node)
{
    auto convert = std::static_pointer_cast<op::Convert>(node);
    if (convert->get_convert_element_type() == convert->get_argument(0)->get_element_type())
    {
        return remove_update_name(node, node->get_argument(0));
    }
    return false;
}

static bool eliminate_slice(const std::shared_ptr<Node>& node)
{
    auto slice = std::static_pointer_cast<op::Slice>(node);
    if (slice->get_input_shape(0) == slice->get_output_shape(0))
    {
        return remove_update_name(node, node->get_argument(0));
    }
    return false;
}

static bool eliminate_broadcast(const std::shared_ptr<Node>& node)
{
    auto broadcast = std::static_pointer_cast<op::v0::Broadcast>(node);
    if (broadcast->get_input_shape(0) == broadcast->get_output_shape(0))
    {
        return remove_update_name(node, node->get_argument(0));
    }
    return false;
}

static bool eliminate_concat(const std::shared_ptr<Node>& node)
{
    auto node_input = node->get_argument(0);

    // remove concat with single input
    if (node->get_input_size() == 1)
    {
        return remove_update_name(node, node_input);
    }
    return false;
}

static bool eliminate_reshape_v1(const std::shared_ptr<Node>& node)
{
    auto node_input = node->get_argument(0);
    // check if reshape is not identity op
    if (node_input->get_output_partial_shape(0).is_dynamic() ||
        node->get_output_partial_shape(0).is_dynamic() ||
        node->get_output_shape(0) != node->get_output_shape(0))
    {
        NGRAPH_DEBUG << "Not a no-op; Shapes are different!";
        return false;
    }
    return remove_update_name(node, node_input);
}

static bool eliminate_stop_gradient(const std::shared_ptr<Node>& node)
{
    replace_node(node, node->get_argument(0));
    return true;
}

static const std::unordered_map<std::type_index, std::function<bool(const std::shared_ptr<Node>&)>>
    dispatcher{{TI(op::v0::Pad), &eliminate_pad},
               {TI(op::Sum), &eliminate_sum},
               {TI(op::Convert), &eliminate_convert},
               {TI(op::Slice), &eliminate_slice},
               {TI(op::StopGradient), &eliminate_stop_gradient},
               {TI(op::v1::Reshape), &eliminate_reshape_v1},
               {TI(op::v0::Concat), &eliminate_concat},
               {TI(op::v0::Broadcast), &eliminate_broadcast}};

bool pass::NopElimination::run_on_function(std::shared_ptr<Function> function)
{
    bool clobbered = false;

    for (const auto& n : function->get_ops())
    {
        // Work around a warning [-Wpotentially-evaluated-expression]
        const Node& node = *n;
        auto handler = dispatcher.find(TI(node));
        if (handler != dispatcher.end())
        {
            clobbered = handler->second(n) || clobbered;
        }
    }

    return clobbered;
}
