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

#include <functional>
#include <memory>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>

#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/convert.hpp"
#include "ngraph/op/pad.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/op/stop_gradient.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/util.hpp"
#include "nop_elimination.hpp"

using namespace std;
using namespace ngraph;

#define TI(x) std::type_index(typeid(x))

static bool eliminate_pad(const std::shared_ptr<Node>& node)
{
    auto pad = std::static_pointer_cast<op::Pad>(node);
    if (pad->get_input_shape(0) == pad->get_output_shape(0))
    {
        replace_node(node, node->get_argument(0));
        return true;
    }
    return false;
}

static bool eliminate_sum(const std::shared_ptr<Node>& node)
{
    auto sum = std::static_pointer_cast<op::Sum>(node);
    if (sum->get_reduction_axes().empty())
    {
        replace_node(node, node->get_argument(0));
        return true;
    }
    return false;
}

static bool eliminate_convert(const std::shared_ptr<Node>& node)
{
    auto convert = std::static_pointer_cast<op::Convert>(node);
    if (convert->get_convert_element_type() == convert->get_argument(0)->get_element_type())
    {
        replace_node(node, node->get_argument(0));
        return true;
    }
    return false;
}

static bool eliminate_slice(const std::shared_ptr<Node>& node)
{
    auto slice = std::static_pointer_cast<op::Slice>(node);
    if (slice->get_input_shape(0) == slice->get_output_shape(0))
    {
        replace_node(node, node->get_argument(0));
        return true;
    }
    return false;
}

static bool replace_broadcast_like(const std::shared_ptr<Node>& node)
{
    // Replace a broadcast like with the broadcast to eliminate the pseudo-dependency on the "like"
    // argument
    auto broadcast_like = std::static_pointer_cast<op::BroadcastLike>(node);
    replace_node(node,
                 std::make_shared<op::Broadcast>(broadcast_like->get_argument(0),
                                                 broadcast_like->get_broadcast_shape(),
                                                 broadcast_like->get_broadcast_axes()));
    return true;
}

static bool eliminate_broadcast(const std::shared_ptr<Node>& node)
{
    auto broadcast = std::static_pointer_cast<op::Broadcast>(node);
    if (broadcast->get_input_shape(0) == broadcast->get_output_shape(0))
    {
        replace_node(node, node->get_argument(0));
        return true;
    }
    return false;
}

static bool eliminate_stop_gradient(const std::shared_ptr<Node>& node)
{
    replace_node(node, node->get_argument(0));
    return true;
}

static const std::unordered_map<std::type_index, std::function<bool(const std::shared_ptr<Node>&)>>
    dispatcher{{TI(op::Pad), &eliminate_pad},
               {TI(op::Sum), &eliminate_sum},
               {TI(op::Convert), &eliminate_convert},
               {TI(op::Slice), &eliminate_slice},
               {TI(op::StopGradient), &eliminate_stop_gradient},
               {TI(op::BroadcastLike), &replace_broadcast_like},
               {TI(op::Broadcast), &eliminate_broadcast}};

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

        // Here we're checking on a common base class of a family of template classes,
        // which is more than type info can handle.
        auto sclb = std::dynamic_pointer_cast<op::ScalarConstantLikeBase>(n);
        if (sclb != nullptr)
        {
            replace_node(sclb, sclb->as_constant());
            clobbered = true;
        }
    }

    return clobbered;
}
