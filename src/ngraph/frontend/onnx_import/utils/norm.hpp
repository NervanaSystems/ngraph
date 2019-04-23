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

#include <memory>

#include "ngraph/axis_set.hpp"
#include "ngraph/node.hpp"
#include "ngraph/node_vector.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace norm
        {
            /// \brief      Calculates L-0 norm of input tensor.
            ///
            /// \note       The L-0 norm represents the cardinality of elements different
            ///             from zero. This actually is not a "true" norm.
            ///
            /// \param[in]  node            The input tensor node.
            /// \param[in]  reduction_axes  The axes along which we calculate norm.
            ///
            /// \return     Node with calculated L-0 norm values.
            ///
            std::shared_ptr<ngraph::Node> l0_norm(const std::shared_ptr<ngraph::Node>& node,
                                                  const ngraph::AxisSet& reduction_axes);

            /// \brief      Calculates L-1 norm of input tensor.
            ///
            /// \note       The L-1 norm represents the sum of absolute values.
            ///
            /// \param[in]  node            The input tensor node.
            /// \param[in]  reduction_axes  The axes along which we calculate norm.
            ///
            /// \return     Node with calculated L-1 norm values.
            ///
            std::shared_ptr<ngraph::Node> l1_norm(const std::shared_ptr<ngraph::Node>& node,
                                                  const ngraph::AxisSet& reduction_axes);

            /// \brief      Calculates L-2 norm of input tensor.
            ///
            /// \note       The L-2 norm represents the square root of sum of squares of each
            ///             individual element.
            ///
            /// \param[in]  node            The input tensor node.
            /// \param[in]  reduction_axes  The axes along which we calculate norm.
            ///
            /// \return     Node with calculated L-2 norm values.
            ///
            std::shared_ptr<ngraph::Node> l2_norm(const std::shared_ptr<ngraph::Node>& node,
                                                  const ngraph::AxisSet& reduction_axes);

            /// \brief      Calculates L-p norm on input tensor.
            ///
            /// \param[in]  node            The input nGraph tensor.
            /// \param[in]  reduction_axes  The axes along which we calculate norm.
            /// \param[in]  p_norm          The p norm to calculate.
            ///
            /// \return     Resulting L-p norm.
            ///
            std::shared_ptr<ngraph::Node> lp_norm(const std::shared_ptr<ngraph::Node>& node,
                                                  const ngraph::AxisSet& reduction_axes,
                                                  std::size_t p_norm = 2);

        } //namespace norm

    } // namespace onnx_import

} // namespace ngraph
