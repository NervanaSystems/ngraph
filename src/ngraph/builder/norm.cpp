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
#include "ngraph/op/maximum.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/not_equal.hpp"
#include "ngraph/op/power.hpp"
#include "ngraph/op/sqrt.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/shape.hpp"

using namespace std;

namespace ngraph
{
    namespace builder
    {
        namespace detail
        {
            shared_ptr<Node> lp_norm(const Output<Node>& value,
                                     size_t p_norm,
                                     const AxisSet& reduction_axes,
                                     float bias)
            {
                // In general "entrywise" lp-norm for matrix `A` is defined as following double sum:
                // ||A||_p = ||vec(A)||_p = [sum_{i=1}^m sum_{j=1}^n abs(a_{i,j})^p]^{1/p}
                shared_ptr<Node> abs_values{make_shared<op::Abs>(value)};
                shared_ptr<Node> p_node = op::Constant::create(
                    value.get_element_type(),
                    value.get_shape(),
                    vector<float>(shape_size(value.get_shape()), static_cast<float>(p_norm)));

                // Get inner part of equation: abs_values^p_node, then sum over reduction_axes.
                shared_ptr<Node> values{make_shared<op::Power>(abs_values, p_node)};
                values = make_shared<op::Sum>(values, reduction_axes);

                shared_ptr<Node> bias_node{
                    op::Constant::create(values->get_element_type(),
                                         values->get_shape(),
                                         vector<float>(shape_size(values->get_shape()), bias))};

                values = values + bias_node;

                // Get outer part of equation: raise values to 1/p_norm exponent.
                shared_ptr<Node> inv_p_node = op::Constant::create(
                    values->get_element_type(),
                    values->get_shape(),
                    vector<float>(shape_size(values->get_shape()), 1.f / p_norm));

                return {make_shared<op::Power>(values, inv_p_node)
                            ->add_provenance_group_members_above({value})};
            }
        }

        shared_ptr<Node> l0_norm(const Output<Node>& value, const AxisSet& reduction_axes)
        {
            // L0 norm returns number of elements different from zero.
            shared_ptr<Node> zero_node{
                op::Constant::create(value.get_element_type(),
                                     value.get_shape(),
                                     vector<float>(shape_size(value.get_shape()), 0.f))};

            // Convert bool values to input node data type.
            shared_ptr<Node> non_zero_values = make_shared<op::Convert>(
                make_shared<op::NotEqual>(value, zero_node), value.get_element_type());

            return make_shared<op::Sum>(non_zero_values, reduction_axes)
                ->add_provenance_group_members_above({value});
        }

        shared_ptr<Node>
            l1_norm(const Output<Node>& value, const AxisSet& reduction_axes, float bias)
        {
            shared_ptr<Node> values{
                make_shared<op::Sum>(make_shared<op::Abs>(value), reduction_axes)};

            shared_ptr<Node> bias_node{
                op::Constant::create(values->get_element_type(),
                                     values->get_shape(),
                                     vector<float>(shape_size(values->get_shape()), bias))};

            return (values + bias_node)->add_provenance_group_members_above({value});
        }

        shared_ptr<Node> l2_norm(const Output<Node>& value,
                                 const AxisSet& reduction_axes,
                                 float bias,
                                 BiasMode bias_mode)
        {
            shared_ptr<Node> values{make_shared<op::Sum>(value * value, reduction_axes)};

            shared_ptr<Node> bias_node{
                op::Constant::create(values->get_element_type(),
                                     values->get_shape(),
                                     vector<float>(shape_size(values->get_shape()), bias))};
            shared_ptr<Node> result;
            switch (bias_mode)
            {
            case BiasMode::MAX:
            {
                result = make_shared<op::Sqrt>(make_shared<op::Maximum>(values, bias_node));
                break;
            }
            case BiasMode::ADD:
            default: result = make_shared<op::Sqrt>(values + bias_node);
            }
            return result->add_provenance_group_members_above({value});
        }

        shared_ptr<Node> lp_norm(const Output<Node>& value,
                                 const AxisSet& reduction_axes,
                                 size_t p_norm,
                                 float bias)
        {
            // The number of non-zero elements
            if (p_norm == 0)
            {
                return l0_norm(value, reduction_axes);
            }
            //  sum of absolute values.
            else if (p_norm == 1)
            {
                return l1_norm(value, reduction_axes, bias);
            }
            // sqrt of sum of squares - Euclidean norm
            else if (p_norm == 2)
            {
                return l2_norm(value, reduction_axes, bias);
            }
            // generic case
            else
            {
                return detail::lp_norm(value, p_norm, reduction_axes, bias);
            }
        }

    } // namespace builder

} // namespace ngraph
