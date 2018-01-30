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
#include "ngraph/axis_set.hpp"
#include "ngraph/ops/add.hpp"
#include "ngraph/ops/divide.hpp"
#include "ngraph/ops/multiply.hpp"
#include "ngraph/ops/power.hpp"
#include "ngraph/ops/subtract.hpp"
#include "ngraph/ops/sum.hpp"

namespace ngraph
{
    namespace builder
    {
        size_t get_num_elements(const Shape& shape, const AxisSet& reduction_axes)
        {
            size_t N = 1;
            for (auto a : reduction_axes)
            {
                N *= shape[a];
            }
            return N;
        }

        std::shared_ptr<Node> l2_norm(const std::shared_ptr<Node>& node,
                                      const AxisSet& reduction_axes)
        {
            const auto& et = node->get_element_type();
            auto x2 = node * node;
            auto x2sum = std::make_shared<op::Sum>(x2, reduction_axes);

            // TODO(mbrookhart): Use Sqrt instead of Power
            auto half = op::Constant::create(et, x2sum->get_shape(), {0.5});
            return std::make_shared<op::Power>(x2sum, half);
        }

        std::shared_ptr<Node> mean(const std::shared_ptr<Node>& node, const AxisSet& reduction_axes)
        {
            auto xsum = std::make_shared<op::Sum>(node, reduction_axes);

            auto N = get_num_elements(node->get_shape(), reduction_axes);
            const auto& et = node->get_element_type();

            auto divisor = op::Constant::create(et, xsum->get_shape(), {N});

            return xsum / divisor;
        }

        std::shared_ptr<Node> std_dev(const std::shared_ptr<Node>& node,
                                      const AxisSet& reduction_axes,
                                      const bool bessel_correction)
        {
            auto var = variance(node, reduction_axes, bessel_correction);

            const auto& et = node->get_element_type();
            // TODO(mbrookhart): Use Sqrt instead of Power
            auto half = op::Constant::create(et, var->get_shape(), {0.5});
            return std::make_shared<op::Power>(var, half);
        }

        // This currently calculates [E[X^2] - E[X]^2] instead of [E[(X-\mu)^2]]
        // The second might be more numerically stable/easier to pattern match
        // It also requires adding a broadcast op, and would probably be slower
        // TODO(mbrookhart): Switch to E[(X-\mu)^2]?
        std::shared_ptr<Node> variance(const std::shared_ptr<Node>& node,
                                       const AxisSet& reduction_axes,
                                       const bool bessel_correction)
        {
            auto xsum = std::make_shared<op::Sum>(node, reduction_axes);

            auto x2 = node * node;

            auto x2sum = std::make_shared<op::Sum>(x2, reduction_axes);

            const auto& et = node->get_element_type();
            auto N = get_num_elements(node->get_shape(), reduction_axes);

            auto Nconst = op::Constant::create(et, xsum->get_shape(), {N});
            auto xbar2 = (xsum * xsum) / Nconst;

            auto diff = x2sum - xbar2;

            if (bessel_correction)
            {
                auto N1const = op::Constant::create(et, xsum->get_shape(), {N - 1});
                return diff / N1const;
            }
            else
            {
                return diff / Nconst;
            }
        }

    } // namespace builder
} // namespace ngraph
