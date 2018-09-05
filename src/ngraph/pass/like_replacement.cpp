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

#include "like_replacement.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/convert.hpp"
#include "ngraph/op/pad.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/op/stop_gradient.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/util.hpp"

#define TI(x) std::type_index(typeid(x))

#define HANDLER_DECL(x) static bool x(const std::shared_ptr<ngraph::Node>& node)

HANDLER_DECL(replace_broadcast_like)
{
    // Replace a broadcast like with the broadcast to eliminate the pseudo-dependency on the "like" argument
    auto broadcast_like = std::dynamic_pointer_cast<ngraph::op::BroadcastLike>(node);
    ngraph::replace_node(
        node,
        std::make_shared<ngraph::op::Broadcast>(broadcast_like->get_argument(0),
                                                broadcast_like->get_broadcast_shape(),
                                                broadcast_like->get_broadcast_axes()));
    return true;
}

static const std::unordered_map<std::type_index,
                                std::function<bool(const std::shared_ptr<ngraph::Node>&)>>
    dispatcher{{TI(ngraph::op::BroadcastLike), &replace_broadcast_like}};

bool ngraph::pass::LikeReplacement::run_on_function(std::shared_ptr<ngraph::Function> function)
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
        auto sclb = std::dynamic_pointer_cast<ngraph::op::ScalarConstantLikeBase>(n);
        if (sclb != nullptr)
        {
            ngraph::replace_node(sclb, sclb->as_constant());
            clobbered = true;
        }
    }

    return clobbered;
}
