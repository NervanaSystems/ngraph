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
        /// \brief      Change shape of a value
        ///
        /// \param[in]  value  The value to be reshaped.
        /// \param[in]  shape  The new shape.
        ///
        /// \return     The reshaped value.
        ///
        std::shared_ptr<Node> reshape(const Output<Node>& value, const Shape& shape);

        /// \brief Permute axes according to specified axes_order parameter.
        ///
        /// \param value The vlaue whose axes we want to permute.
        /// \param axes_order The permutation of axes.
        ///
        /// \return: Value with permuted axes.
        std::shared_ptr<Node> reorder_axes(const Output<Node>& value,
                                           std::vector<size_t> axes_order = {});

        /// \brief Return transposed vlaue (with axes in reversed order).
        ///
        /// \param value Value to transpose.
        ///
        /// \return: Value with reversed dimensions.
        std::shared_ptr<Node> transpose(const Output<Node>& value);

        /// \brief Flatten a value into a 2D matrix.
        ///
        /// \param value The tensor to be flattened.
        /// \param axis The axis dividing shape.
        ///
        /// \return The new value will be a 2D matrix representing the flattened input node.
        std::shared_ptr<Node> flatten(const Output<Node>& value, int axis);
    } // namespace  builder
} // namespace  ngraph
