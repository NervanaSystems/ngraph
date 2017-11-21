/*
 Copyright 2017 Nervana Systems Inc.
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
*/

#include "ngraph/builder/mean.hpp"
#include "ngraph/builder/reduce.hpp"
#include "ngraph/ops/add.hpp"
#include "ngraph/ops/constant.hpp"
#include "ngraph/ops/divide.hpp"

using namespace std;

namespace ngraph
{
    namespace builder
    {
        std::shared_ptr<Node> Mean(const std::shared_ptr<Node>& node, const AxisSet& reduction_axes)
        {
            size_t N = 1;
            for (auto a : reduction_axes) 
            {
                N *= node->get_shape()[a];
            }

            auto sum = create_reduction<op::Add>(node, "0", reduction_axes);
            const auto& et = node->get_element_type();
            auto divisor = make_shared<op::Constant>(et, sum->get_shape(), std::to_string(N));

            return make_shared<op::Divide>(sum, divisor);
        }
    } // namespace builder
} // namespace ngraph
