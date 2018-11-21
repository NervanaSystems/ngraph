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

#include "ngraph/builder/broadcast_scalar_to.hpp"
#include "ngraph/builder/shape_rank.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/experimental/dyn_broadcast.hpp"
#include "ngraph/op/experimental/range.hpp"

namespace ngraph
{
    namespace builder
    {
        std::shared_ptr<Node> broadcast_scalar_to(std::shared_ptr<Node> shape_node, std::shared_ptr<Node> scalar_node)
        {
            auto broadcast_axes = std::make_shared<op::Range>(op::Constant::create<uint64_t>(element::u64, Shape{}, {0}), builder::shape_rank(shape_node));
            return std::make_shared<op::DynBroadcast>(scalar_node, shape_node, broadcast_axes);
        }
    } // namespace builder
} // namespace ngraph
