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

#include "ngraph/builder/shape_rank.hpp"
#include "ngraph/op/experimental/shape_of.hpp"
#include "ngraph/op/reshape.hpp"

namespace ngraph
{
    namespace builder
    {
        std::shared_ptr<Node> shape_rank(std::shared_ptr<Node> shape_node)
        {
            // assumption: shape_node produces a vector of u64
            auto shape_of_shape_node = std::make_shared<op::ShapeOf>(shape_node); // Shape {1}
            auto scalar_shape_of_shape_node = std::make_shared<op::Reshape>(shape_of_shape_node, AxisVector{0}, Shape{});
            return scalar_shape_of_shape_node;
        }
    } // namespace builder
} // namespace ngraph
