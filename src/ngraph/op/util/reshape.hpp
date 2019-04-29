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

#include "ngraph/axis_vector.hpp"
#include "ngraph/node.hpp"
#include "ngraph/util.hpp"

namespace ngraph
{
    namespace op
    {
        namespace util
        {
            /// \brief      Change shape of input tensor.
            ///
            /// \param[in]  node   The node which shape will be used as input to Reshape.
            /// \param[in]  shape  The new shape for input tensor.
            ///
            /// \return     The node representing a Reshape operation.
            ///
            std::shared_ptr<ngraph::Node> reshape(const std::shared_ptr<ngraph::Node>& node,
                                                  const AxisVector& axis_order,
                                                  const Shape& shape);

            inline std::shared_ptr<ngraph::Node> reshape(const std::shared_ptr<ngraph::Node>& node,
                                                         const Shape& shape)
            {
                return reshape(node, ngraph::get_default_order(node->get_shape().size()), shape);
            }

            /// \brief Permute axes according to specified axes_order parameter.
            ///
            /// \param node The node which axes we want to permute.
            /// \param axes_order The permutation of node tensor axes.
            ///
            /// \return: New node with permuted axes.
            std::shared_ptr<ngraph::Node> reorder_axes(const std::shared_ptr<ngraph::Node>& node,
                                                       std::vector<std::size_t> axes_order);
        } // namespace util
    }     // namespace  op
} // namespace  ngraph
