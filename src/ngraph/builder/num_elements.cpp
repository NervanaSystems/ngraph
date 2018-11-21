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

#include "ngraph/builder/num_elements.hpp"
#include "ngraph/op/experimental/shape_of.hpp"
#include "ngraph/op/product.hpp"

namespace ngraph
{
    namespace builder
    {
        std::shared_ptr<Node> num_elements(std::shared_ptr<Node> node)
        {
            auto shape_node = std::make_shared<op::ShapeOf>(node); // Shape {?}
            auto shape_product = std::make_shared<op::Product>(shape_node, AxisSet{0}); // Shape {}
            return shape_product;
        }
    } // namespace builder
} // namespace ngraph
