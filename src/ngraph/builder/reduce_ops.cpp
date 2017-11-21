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

#include "ngraph/builder/reduce_ops.hpp"
#include "ngraph/ops/add.hpp"
#include "ngraph/ops/divide.hpp"
#include "ngraph/ops/multiply.hpp"

using namespace std;

namespace ngraph
{
    namespace builder
    {
        template <typename T>
        inline std::shared_ptr<Node> create_reduction(const std::shared_ptr<Node>& node,
                                                      const std::string& init_val,
                                                      const AxisSet& reduction_axes)
        {
            const auto& et = node->get_element_type();

            auto f_A = std::make_shared<op::Parameter>(et, Shape{});
            auto f_B = std::make_shared<op::Parameter>(et, Shape{});
            auto f_rt = std::make_shared<TensorViewType>(et, Shape{});
            auto f = std::make_shared<Function>(
                std::make_shared<T>(f_A, f_B), f_rt, op::Parameters{f_A, f_B});

            auto init = std::make_shared<op::Constant>(et, Shape{}, init_val);

            return std::make_shared<op::Reduce>(node, init, f, reduction_axes);
        }

        std::shared_ptr<Node> Mean(const std::shared_ptr<Node>& node, const AxisSet& reduction_axes)
        {
            size_t N = 1;
            for (auto a : reduction_axes)
            {
                N *= node->get_shape()[a];
            }

            auto sum = Sum(node, reduction_axes);
            const auto& et = node->get_element_type();
            auto divisor = make_shared<op::Constant>(et, sum->get_shape(), std::to_string(N));

            return make_shared<op::Divide>(sum, divisor);
        }

        std::shared_ptr<Node> Prod(const std::shared_ptr<Node>& node, const AxisSet& reduction_axes)
        {
            return create_reduction<op::Multiply>(node, "1", reduction_axes);
        }

        std::shared_ptr<Node> Sum(const std::shared_ptr<Node>& node, const AxisSet& reduction_axes)
        {
            return create_reduction<op::Add>(node, "0", reduction_axes);
        }

    } // namespace builder
} // namespace ngraph
