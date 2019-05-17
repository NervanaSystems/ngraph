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
                                                  const Shape& shape);

            /// \brief Permute axes according to specified axes_order parameter.
            ///
            /// \param node The node which axes we want to permute.
            /// \param axes_order The permutation of node tensor axes.
            ///
            /// \return: New node with permuted axes.
            std::shared_ptr<ngraph::Node> reorder_axes(const std::shared_ptr<ngraph::Node>& node,
                                                       std::vector<std::size_t> axes_order);

            /// \brief Return transposed tensor (with axes in reversed order).
            ///
            /// \param node Input tensor we want to transpose
            ///
            /// \return: New node with reversed dimensions.
            std::shared_ptr<ngraph::Node> transpose(const std::shared_ptr<ngraph::Node>& node);

            /// \brief Flatten the input tensor into a 2D matrix.
            ///
            /// \param node The tensor to be flattened.
            /// \param axis The axis dividing shape.
            ///
            /// \return The new node being a 2D matrix representing flattened input node.
            std::shared_ptr<ngraph::Node> flatten(const std::shared_ptr<ngraph::Node>& node,
                                                  int axis);

            /// \brief      Split node on specified axis into multiple parts.
            ///
            /// \param[in]  node          The input node.
            /// \param[in]  length_parts  The vector defining the lengts of each splitted part.
            /// \param[in]  axis          The axis we split input node on. Default value is zero axis.
            ///
            /// \return     The vector containing multiple nodes we split input node into.
            ///
            NodeVector split(const std::shared_ptr<ngraph::Node>& node,
                             const std::vector<std::size_t>& length_parts,
                             std::size_t axis = 0);

            /// \brief      Split node on specified axis into multiple parts.
            ///
            /// \param[in]  node          The input node.
            /// \param[in]  split_parts   The number of parts we want to split input node at given
            ///                           axis. The length of the axis to split must be divisible by
            ///                           this value.
            /// \param[in]  axis          The axis we split input node on. Default value is zero axis.
            ///
            /// \note       This implementation supports negative `axis` values (similar to NumPy
            ///             indexing).
            ///
            /// \return     The vector containing multiple nodes we split input node into.
            ///
            NodeVector split(const std::shared_ptr<ngraph::Node>& node,
                             std::size_t split_parts,
                             int axis = 0);
        } // namespace util
    }     // namespace  op
} // namespace  ngraph
