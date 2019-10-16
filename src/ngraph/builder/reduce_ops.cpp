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

#include <numeric>

#include "ngraph/axis_set.hpp"
#include "ngraph/builder/autobroadcast.hpp"
#include "ngraph/builder/reduce_ops.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/sqrt.hpp"
#include "ngraph/op/subtract.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/util.hpp"

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

        std::shared_ptr<Node> l2_norm(const Output<Node>& node, const AxisSet& reduction_axes)
        {
            auto x2 = node * node;
            auto x2sum = std::make_shared<op::Sum>(x2, reduction_axes);

            return std::make_shared<op::Sqrt>(x2sum)->add_provenance_group_members_above({node});
        }

        std::shared_ptr<Node> mean(const Output<Node>& value, const AxisSet& reduction_axes)
        {
            auto xsum = std::make_shared<op::Sum>(value, reduction_axes);

            auto N = get_num_elements(value.get_shape(), reduction_axes);
            const auto& et = value.get_element_type();

            auto divisor = op::Constant::create(et, xsum->get_shape(), {N});

            return (xsum / divisor)->add_provenance_group_members_above({value});
        }

        std::shared_ptr<Node> std_dev(const Output<Node>& node,
                                      const AxisSet& reduction_axes,
                                      const bool bessel_correction)
        {
            return std::make_shared<op::Sqrt>(variance(node, reduction_axes, bessel_correction))
                ->add_provenance_group_members_above({node});
        }

        // This currently calculates [E[X^2] - E[X]^2] instead of [E[(X-\mu)^2]]
        // The second might be more numerically stable/easier to pattern match
        // It also requires adding a broadcast op, and would probably be slower
        // TODO(mbrookhart): Switch to E[(X-\mu)^2]?
        std::shared_ptr<Node> variance(const Output<Node>& value,
                                       const AxisSet& reduction_axes,
                                       const bool bessel_correction)
        {
            std::shared_ptr<Node> mu = mean(value, reduction_axes);

            auto reshape = value.get_shape();
            for (auto i : reduction_axes)
            {
                reshape[i] = 1;
            }

            ngraph::AxisVector order = ngraph::get_default_order(mu->get_shape());

            mu = std::make_shared<op::Reshape>(mu, order, reshape);

            Output<Node> diff = make_with_numpy_broadcast<op::Subtract>(value, mu);

            diff = std::make_shared<op::Sum>(diff * diff, reduction_axes);

            const auto& et = value.get_element_type();
            auto N = get_num_elements(value.get_shape(), reduction_axes);

            std::shared_ptr<Node> result;
            if (bessel_correction)
            {
                auto N1const = op::Constant::create(et, diff.get_shape(), {N - 1});
                result = diff / N1const;
            }
            else
            {
                auto Nconst = op::Constant::create(et, diff.get_shape(), {N});
                result = diff / Nconst;
            }
            return result->add_provenance_group_members_above({value});
        }

    } // namespace builder
} // namespace ngraph
