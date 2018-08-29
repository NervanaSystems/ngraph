//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
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
#include "ngraph/op/convert.hpp"
#include "ngraph/op/pad.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/op/stop_gradient.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/util.hpp"
#include "nop_elimination.hpp"

#define TI(x) std::type_index(typeid(x))

#define HANDLER_DECL(x) static bool x(const std::shared_ptr<ngraph::Node>& node)

HANDLER_DECL(eliminate_pad)
{
    auto pad = std::dynamic_pointer_cast<ngraph::op::Pad>(node);
    if (pad->get_input_shape(0) == pad->get_output_shape(0))
    {
        ngraph::replace_node(node, node->get_argument(0));
        return true;
    }
    return false;
}

HANDLER_DECL(eliminate_sum)
{
    auto sum = std::dynamic_pointer_cast<ngraph::op::Sum>(node);
    if (sum->get_reduction_axes().empty())
    {
        ngraph::replace_node(node, node->get_argument(0));
        return true;
    }
    return false;
}

HANDLER_DECL(eliminate_convert)
{
    auto convert = std::dynamic_pointer_cast<ngraph::op::Convert>(node);
    if (convert->get_convert_element_type() == convert->get_argument(0)->get_element_type())
    {
        ngraph::replace_node(node, node->get_argument(0));
        return true;
    }
    return false;
}

HANDLER_DECL(eliminate_slice)
{
    auto slice = std::dynamic_pointer_cast<ngraph::op::Slice>(node);
    if (slice->get_input_shape(0) == slice->get_output_shape(0))
    {
        ngraph::replace_node(node, node->get_argument(0));
        return true;
    }
    return false;
}

HANDLER_DECL(eliminate_broadcast)
{
    auto broadcast = std::dynamic_pointer_cast<ngraph::op::Broadcast>(node);
    if (broadcast->get_input_shape(0) == broadcast->get_output_shape(0))
    {
        ngraph::replace_node(node, node->get_argument(0));
        return true;
    }
    return false;
}

HANDLER_DECL(eliminate_stop_gradient)
{
    ngraph::replace_node(node, node->get_argument(0));
    return true;
}

static const std::unordered_map<std::type_index,
                                std::function<bool(const std::shared_ptr<ngraph::Node>&)>>
    dispatcher{{TI(ngraph::op::Pad), &eliminate_pad},
               {TI(ngraph::op::Sum), &eliminate_sum},
               {TI(ngraph::op::Convert), &eliminate_convert},
               {TI(ngraph::op::Slice), &eliminate_slice},
               {TI(ngraph::op::StopGradient), &eliminate_stop_gradient},
               {TI(ngraph::op::Broadcast), &eliminate_broadcast}};

bool ngraph::pass::NopElimination::run_on_function(std::shared_ptr<ngraph::Function> function)
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
