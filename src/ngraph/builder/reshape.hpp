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
#include <vector>

#include "ngraph/node.hpp"
#include "ngraph/shape.hpp"

namespace ngraph
{
    namespace builder
    {
        /// \brief      Change shape of input tensor.
        ///
        /// \param[in]  node   The node producing the tensor to be reshaped.
        /// \param[in]  shape  The new shape for input tensor.
        ///
        /// \return     The node representing a Reshape operation.
        ///
        std::shared_ptr<Node> reshape(const std::shared_ptr<Node>& node, const Shape& shape);

        /// \brief Permute axes according to specified axes_order parameter.
        ///
        /// \param node The node which axes we want to permute.
        /// \param axes_order The permutation of node tensor axes.
        ///
        /// \return: New node with permuted axes.
        std::shared_ptr<Node> reorder_axes(const std::shared_ptr<Node>& node,
                                           std::vector<std::size_t> axes_order);

        /// \brief Return transposed tensor (with axes in reversed order).
        ///
        /// \param node Input tensor we want to transpose
        ///
        /// \return: New node with reversed dimensions.
        std::shared_ptr<Node> transpose(const std::shared_ptr<Node>& node);

        /// \brief Flatten the input tensor into a 2D matrix.
        ///
        /// \param node The tensor to be flattened.
        /// \param axis The axis dividing shape.
        ///
        /// \return The new node will be a 2D matrix representing the flattened input node.
        std::shared_ptr<Node> flatten(const std::shared_ptr<Node>& node, int axis);

        /// \brief      Remove empty axes from input tensor.
        ///
        /// \param[in]  node  The node to be squeezed.
        /// \param[in]  axes  The vector defining indexes of axes to be removed.
        ///
        /// \return     The squeezed node.
        ///
        std::shared_ptr<Node> squeeze(const std::shared_ptr<Node>& node,
                                      std::vector<std::size_t> axes = {0});

        /// \brief      Collapse specified axes into single one.
        ///
        /// \note       Collapsed axes create a continuous range starting from outermost axis.
        ///
        /// \param[in]  node        The node to be reshaped.
        /// \param[in]  start_axis  The start axis index.
        /// \param[in]  end_axis    The end axis (inclusive) index.
        ///
        /// \return     The node with collapsed specified axes.
        ///
        std::shared_ptr<Node> collapse(const std::shared_ptr<Node>& node,
                                       const std::size_t start_axis,
                                       const std::size_t end_axis);

        /// \brief      Expands node tensor shape with empty axis at
        ///             specified position.
        ///
        /// \param[in]  node  The node to be expanded.
        /// \param[in]  axis  The position in the expanded axes where the
        ///                   new axis is placed.
        ///
        /// \return     The node with added empty axis.
        ///
        std::shared_ptr<Node> expand_dims(const std::shared_ptr<Node>& node, std::size_t axis = 0);
    } // namespace  builder
} // namespace  ngraph
