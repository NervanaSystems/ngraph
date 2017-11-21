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
#include "ngraph/ops/subtract.hpp"
#include "ngraph/ops/power.hpp"


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

        std::shared_ptr<Node> L2Norm(const std::shared_ptr<Node>& node,
                                     const AxisSet& reduction_axes)
        {
            const auto& et = node->get_element_type();
            auto two = std::make_shared<op::Constant>(et, node->get_shape(), "2");
            auto pow = std::make_shared<op::Power>(node, two);

            auto summed = create_reduction<op::Add>(pow, "0", reduction_axes);
            auto half = std::make_shared<op::Constant>(et, summed->get_shape(), "0.5");

            return std::make_shared<op::Power>(summed, half);
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
            auto divisor = std::make_shared<op::Constant>(et, sum->get_shape(), std::to_string(N));

            return std::make_shared<op::Divide>(sum, divisor);
        }

        std::shared_ptr<Node> Prod(const std::shared_ptr<Node>& node, const AxisSet& reduction_axes)
        {
            return create_reduction<op::Multiply>(node, "1", reduction_axes);
        }

        std::shared_ptr<Node> Std_dev(const std::shared_ptr<Node>& node, const AxisSet& reduction_axes, 
                                       const bool bessel_correction)
        {   
            const auto& et = node->get_element_type();
            auto var = Variance(node, reduction_axes, bessel_correction);
            auto half = std::make_shared<op::Constant>(et, var->get_shape(), "0.5");

            return std::make_shared<op::Power>(var, half);
        }

        std::shared_ptr<Node> Sum(const std::shared_ptr<Node>& node, const AxisSet& reduction_axes)
        {
            return create_reduction<op::Add>(node, "0", reduction_axes);
        }
        
        std::shared_ptr<Node> Variance(const std::shared_ptr<Node>& node,
                                       const AxisSet& reduction_axes, 
                                       const bool bessel_correction)
        {
            const auto& et = node->get_element_type();
            size_t N = 1;
            for (auto a : reduction_axes)
            {
                N *= node->get_shape()[a];
            }

            auto sum = Sum(node, reduction_axes);

            auto two = std::make_shared<op::Constant>(et, node->get_shape(), "2");
            auto pow = std::make_shared<op::Power>(node, two);

            auto x2bar = create_reduction<op::Add>(pow, "0", reduction_axes);

            auto Nconst = std::make_shared<op::Constant>(et, sum->get_shape(), std::to_string(N));
            auto xbar2 = (sum * sum) / Nconst;

            auto diff = x2bar - xbar2;

            if (bessel_correction) {
                auto N1const = std::make_shared<op::Constant>(et, sum->get_shape(), std::to_string(N-1));
                return diff / N1const;
            } else {
                return diff / Nconst;
            }
        }

    } // namespace builder
} // namespace ngraph
