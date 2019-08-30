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

#include "like_replacement.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/convert.hpp"
#include "ngraph/op/pad.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/op/stop_gradient.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

#define TI(x) type_index(typeid(x))

static bool replace_broadcast_like(const std::shared_ptr<ngraph::Node>& node)
{
    // Replace a broadcast like with the broadcast to eliminate the pseudo-dependency on the "like"
    // argument
    auto broadcast_like = static_pointer_cast<op::BroadcastLike>(node);
    replace_node(node,
                 make_shared<op::Broadcast>(broadcast_like->get_argument(0),
                                            broadcast_like->get_broadcast_shape(),
                                            broadcast_like->get_broadcast_axes()));
    return true;
}

static const unordered_map<type_index, function<bool(const shared_ptr<Node>&)>> dispatcher{
    {TI(op::BroadcastLike), &replace_broadcast_like}};

bool pass::LikeReplacement::run_on_function(shared_ptr<Function> function)
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
        auto sclb = dynamic_pointer_cast<op::ScalarConstantLikeBase>(n);
        if (sclb != nullptr)
        {
            replace_node(sclb, sclb->as_constant());
            clobbered = true;
        }
    }

    return clobbered;
}
