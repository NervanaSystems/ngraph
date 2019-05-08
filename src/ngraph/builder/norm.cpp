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
#include "ngraph/op/add.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/convert.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/not_equal.hpp"
#include "ngraph/op/power.hpp"
#include "ngraph/op/sqrt.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/shape.hpp"

namespace ngraph
{
    namespace builder
    {
        namespace detail
        {
            inline std::shared_ptr<Node> get_bias_node(const element::Type& element_type,
                                                       const Shape& node_shape,
                                                       float bias)
            {
                return op::Constant::create(
                    element_type, node_shape, std::vector<float>(shape_size(node_shape), bias));
            }

            std::shared_ptr<Node> lp_norm(const std::shared_ptr<Node>& node,
                                          std::size_t p_norm,
                                          const AxisSet& reduction_axes,
                                          float bias)
            {
                std::shared_ptr<Node> abs_values{std::make_shared<op::Abs>(node)};
                std::shared_ptr<Node> p_node = op::Constant::create(
                    node->get_element_type(),
                    node->get_shape(),
                    std::vector<float>(shape_size(node->get_shape()), static_cast<float>(p_norm)));

                std::shared_ptr<Node> values{std::make_shared<op::Power>(abs_values, p_node)};

                values = std::make_shared<op::Sum>(values, reduction_axes);

                std::shared_ptr<Node> bias_node{
                    detail::get_bias_node(values->get_element_type(), values->get_shape(), bias)};
                values = values + bias_node;

                std::shared_ptr<Node> inv_p_node = op::Constant::create(
                    values->get_element_type(),
                    values->get_shape(),
                    std::vector<float>(shape_size(values->get_shape()), 1.f / p_norm));

                return {std::make_shared<op::Power>(values, inv_p_node)};
            }
        }

        std::shared_ptr<Node> l0_norm(const std::shared_ptr<Node>& node,
                                      const AxisSet& reduction_axes)
        {
            std::shared_ptr<Node> abs_values{std::make_shared<op::Abs>(node)};
            std::shared_ptr<Node> zero_node{
                op::Constant::create(node->get_element_type(),
                                     node->get_shape(),
                                     std::vector<float>(shape_size(node->get_shape()), 0.f))};

            std::shared_ptr<Node> non_zero_values =
                std::make_shared<op::Convert>(std::make_shared<op::NotEqual>(abs_values, zero_node),
                                              abs_values->get_element_type());

            return std::make_shared<op::Sum>(non_zero_values, reduction_axes);
        }

        std::shared_ptr<Node>
            l1_norm(const std::shared_ptr<Node>& node, const AxisSet& reduction_axes, float bias)
        {
            std::shared_ptr<Node> values{
                std::make_shared<op::Sum>(std::make_shared<op::Abs>(node), reduction_axes)};

            std::shared_ptr<Node> bias_node{
                detail::get_bias_node(values->get_element_type(), values->get_shape(), bias)};

            return values + bias_node;
        }

        std::shared_ptr<Node>
            l2_norm(const std::shared_ptr<Node>& node, const AxisSet& reduction_axes, float bias)
        {
            // std::shared_ptr<Node> abs_values{std::make_shared<op::Abs>(node)};
            std::shared_ptr<Node> values{std::make_shared<op::Sum>(node * node, reduction_axes)};

            std::shared_ptr<Node> bias_node{
                detail::get_bias_node(values->get_element_type(), values->get_shape(), bias)};

            return {std::make_shared<op::Sqrt>(values + bias_node)};
        }

        std::shared_ptr<Node> lp_norm(const std::shared_ptr<Node>& node,
                                      const AxisSet& reduction_axes,
                                      std::size_t p_norm,
                                      float bias)
        {
            // The number of non-zero elements
            if (p_norm == 0)
            {
                return l0_norm(node, reduction_axes);
            }
            //  sum of absolute values.
            else if (p_norm == 1)
            {
                return l1_norm(node, reduction_axes, bias);
            }
            // sqrt of sum of squares - Euclidean norm
            else if (p_norm == 2)
            {
                return l2_norm(node, reduction_axes, bias);
            }
            // generic case
            else
            {
                return detail::lp_norm(node, p_norm, reduction_axes, bias);
            }
        }

    } // namespace builder

} // namespace ngraph
