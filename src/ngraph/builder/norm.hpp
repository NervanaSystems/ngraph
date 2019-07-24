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
#pragma once

#include <cstddef>
#include <memory>

#include "ngraph/axis_set.hpp"
#include "ngraph/node.hpp"

namespace ngraph
{
    namespace builder
    {
        /// \brief      Creates output which calculates L-0 norm of input tensor.
        ///
        /// \note       The L-0 norm represents the cardinality of elements different
        ///             from zero. This actually is not a "true" norm.
        ///
        /// \param[in]  value           The input tensor.
        /// \param[in]  reduction_axes  The axes along which we calculate norm.
        ///
        /// \return     L-0 norm of value.
        ///
        Output<Node> l0_norm_value(const Output<Node>& value, const AxisSet& reduction_axes);

        /// \brief      Creates node which calculates L-0 norm of input tensor.
        ///
        /// \note       The L-0 norm represents the cardinality of elements different
        ///             from zero. This actually is not a "true" norm.
        ///
        /// \param[in]  node            The input tensor node.
        /// \param[in]  reduction_axes  The axes along which we calculate norm.
        ///
        /// \return     Node which calculates L-0 norm values.
        ///
        inline std::shared_ptr<Node> l0_norm(const std::shared_ptr<Node>& node,
                                             const AxisSet& reduction_axes)
            NGRAPH_DEPRECATED("Replace with l0_norm_value.")
        {
            return l0_norm_value(node, reduction_axes).get_node_shared_ptr();
        }

        /// \brief      Calculates L-1 norm of a value.
        ///
        /// \note       The L-1 norm represents the sum of absolute values.
        ///
        /// \param[in]  value           The input tensor.
        /// \param[in]  reduction_axes  The axes along which we calculate norm.
        /// \param[in]  bias            The bias added to the calculated sum.
        ///
        /// \return     L-1 norm of value.
        ///
        Output<Node> l1_norm_value(const Output<Node>& value,
                                   const AxisSet& reduction_axes,
                                   float bias = 0.f);

        /// \brief      Creates node which calculates L-1 norm of input tensor.
        ///
        /// \note       The L-1 norm represents the sum of absolute values.
        ///
        /// \param[in]  node            The input tensor node.
        /// \param[in]  reduction_axes  The axes along which we calculate norm.
        /// \param[in]  bias            The bias added to the calculated sum.
        ///
        /// \return     Node which calculates L-1 norm values.
        ///
        inline std::shared_ptr<Node> l1_norm(const std::shared_ptr<Node>& node,
                                             const AxisSet& reduction_axes,
                                             float bias = 0.f)
            NGRAPH_DEPRECATED("Replace with l1_norm_value.")
        {
            return l1_norm_value(node, reduction_axes, bias).get_node_shared_ptr();
        }

        /// \brief      Calculates L-2 norm of input tensor.
        ///
        /// \note       The L-2 norm represents the square root of sum of squares of each
        ///             individual element.
        ///
        /// \param[in]  value           The input tensor.
        /// \param[in]  reduction_axes  The axes along which we calculate norm.
        /// \param[in]  bias            The bias added to the calculated sum.
        ///
        /// \return     L-2 norm of value.
        ///
        Output<Node> l2_norm_value(const Output<Node>& value,
                                   const AxisSet& reduction_axes,
                                   float bias = 0.f);

        /// \brief      Calculates L-2 norm of input tensor.
        ///
        /// \note       The L-2 norm represents the square root of sum of squares of each
        ///             individual element.
        ///
        /// \param[in]  node            The input tensor node.
        /// \param[in]  reduction_axes  The axes along which we calculate norm.
        /// \param[in]  bias            The bias added to the calculated sum.
        ///
        /// \return     Node which calculates L-2 norm values.
        ///
        inline std::shared_ptr<Node> l2_norm(const std::shared_ptr<Node>& node,
                                             const AxisSet& reduction_axes,
                                             float bias = 0.f)
            NGRAPH_DEPRECATED("Replace with l2_norm_value.")
        {
            return l2_norm_value(node, reduction_axes, bias).get_node_shared_ptr();
        }

        /// \brief      Creates node which calculates L-p norm on input tensor.
        ///
        /// \param[in]  value           The input tensor.
        /// \param[in]  reduction_axes  The axes along which we calculate norm.
        /// \param[in]  p_norm          The p norm to calculate.
        /// \param[in]  bias            The bias added to the calculated sum.
        ///
        /// \return     L-p norm of value.
        ///
        Output<Node> lp_norm_value(const Output<Node>& value,
                                   const AxisSet& reduction_axes,
                                   std::size_t p_norm = 2,
                                   float bias = 0.f);

        /// \brief      Creates node which calculates L-p norm on input tensor.
        ///
        /// \param[in]  node            The input nGraph tensor.
        /// \param[in]  reduction_axes  The axes along which we calculate norm.
        /// \param[in]  p_norm          The p norm to calculate.
        /// \param[in]  bias            The bias added to the calculated sum.
        ///
        /// \return     Node which calculates L-p norm.
        ///
        inline std::shared_ptr<Node> lp_norm(const std::shared_ptr<Node>& node,
                                             const AxisSet& reduction_axes,
                                             size_t p_norm = 2,
                                             float bias = 0.f)
            NGRAPH_DEPRECATED("Replace with lp_norm_value.")
        {
            return lp_norm_value(node, reduction_axes, p_norm, bias).get_node_shared_ptr();
        }

    } // namespace builder

} // namespace ngraph
