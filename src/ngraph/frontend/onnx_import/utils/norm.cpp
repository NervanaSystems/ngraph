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

#include "norm.hpp"
#include "ngraph/op/abs.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/convert.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/not_equal.hpp"
#include "ngraph/op/power.hpp"
#include "ngraph/op/sqrt.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/shape.hpp"
#include "utils/common.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace norm
        {
            namespace detail
            {
                std::shared_ptr<ngraph::Node> lp_norm(const std::shared_ptr<ngraph::Node>& node,
                                                      std::size_t p_norm,
                                                      const ngraph::AxisSet& reduction_axes)
                {
                    std::shared_ptr<ngraph::Node> abs_values{
                        std::make_shared<ngraph::op::Abs>(node)};
                    std::shared_ptr<ngraph::Node> p_node = ngraph::op::Constant::create(
                        node->get_element_type(),
                        node->get_shape(),
                        std::vector<float>(shape_size(node->get_shape()),
                                           static_cast<float>(p_norm)));

                    std::shared_ptr<ngraph::Node> values =
                        std::make_shared<ngraph::op::Power>(abs_values, p_node);

                    values = std::make_shared<ngraph::op::Sum>(values, reduction_axes);

                    std::shared_ptr<ngraph::Node> inv_p_node = ngraph::op::Constant::create(
                        values->get_element_type(),
                        values->get_shape(),
                        std::vector<float>(shape_size(values->get_shape()), 1.f / p_norm));

                    return {std::make_shared<ngraph::op::Power>(values, inv_p_node)};
                }
            }

            std::shared_ptr<ngraph::Node> l0_norm(const std::shared_ptr<ngraph::Node>& node,
                                                  const ngraph::AxisSet& reduction_axes)
            {
                std::shared_ptr<ngraph::Node> abs_values{std::make_shared<ngraph::op::Abs>(node)};
                std::shared_ptr<ngraph::Node> zero_node{ngraph::op::Constant::create(
                    node->get_element_type(),
                    node->get_shape(),
                    std::vector<float>(shape_size(node->get_shape()), 0.f))};

                std::shared_ptr<ngraph::Node> non_zero_values =
                    std::make_shared<ngraph::op::Convert>(
                        std::make_shared<ngraph::op::NotEqual>(abs_values, zero_node),
                        abs_values->get_element_type());

                return std::make_shared<ngraph::op::Sum>(non_zero_values, reduction_axes);
            }

            std::shared_ptr<ngraph::Node> l1_norm(const std::shared_ptr<ngraph::Node>& node,
                                                  const ngraph::AxisSet& reduction_axes)
            {
                return std::make_shared<ngraph::op::Sum>(std::make_shared<ngraph::op::Abs>(node),
                                                         reduction_axes);
            }

            std::shared_ptr<ngraph::Node> l2_norm(const std::shared_ptr<ngraph::Node>& node,
                                                  const ngraph::AxisSet& reduction_axes)
            {
                std::shared_ptr<ngraph::Node> abs_values{std::make_shared<ngraph::op::Abs>(node)};
                return {std::make_shared<ngraph::op::Sqrt>(
                    std::make_shared<ngraph::op::Sum>(abs_values * abs_values, reduction_axes))};
            }

            std::shared_ptr<ngraph::Node> lp_norm(const std::shared_ptr<ngraph::Node>& node,
                                                  const ngraph::AxisSet& reduction_axes,
                                                  std::size_t p_norm)
            {
                // The number of non-zero elements
                if (p_norm == 0)
                {
                    return l0_norm(node, reduction_axes);
                }
                //  sum of absolute values.
                else if (p_norm == 1)
                {
                    return l1_norm(node, reduction_axes);
                }
                // sqrt of sum of squares - Euclidean norm
                else if (p_norm == 2)
                {
                    return l2_norm(node, reduction_axes);
                }
                // generic case
                else
                {
                    return detail::lp_norm(node, p_norm, reduction_axes);
                }
            }

        } //namespace norm

    } // namespace onnx_import

} // namespace ngraph
