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
        /// \param[in]  node   The node which shape will be used as input to Reshape.
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
        /// \return The new node being a 2D matrix representing flattened input node.
        std::shared_ptr<Node> flatten(const std::shared_ptr<Node>& node, int axis);
    } // namespace  builder
} // namespace  ngraph
