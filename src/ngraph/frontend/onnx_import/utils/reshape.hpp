//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
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
#include <string>
#include <vector>

#include "ngraph/axis_vector.hpp"
#include "ngraph/node.hpp"
#include "ngraph/node_vector.hpp"
#include "ngraph/shape.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace reshape
        {
            /// \brief Flatten the input tensor into a 2D matrix.
            ///
            /// \param node The tensor to be flattened.
            /// \param axis The axis dividing shape.
            ///
            /// \return The new node being a 2D matrix representing flattened input node.
            std::shared_ptr<ngraph::Node> flatten(const std::shared_ptr<ngraph::Node>& node,
                                                  int axis);

            /// \brief      Gets the AxisVector filled with monotonic increasing
            ///             sequence.
            ///
            /// \param[in]  data_shape_size  The data shape size.
            /// \param[in]  start_value      The start_value for sequence. Default equals 0.
            ///
            /// \return     The filled AxisVector.
            ///
            AxisVector get_default_axis_vector(std::size_t data_shape_size,
                                               std::size_t start_value = 0);

            /// \brief      Infer `output_shape` dimension values.
            ///
            /// \par Inferention rules
            ///     \li         The input_shape may consist at most on -1 value. In this case the value
            ///                 is inferred from the size of the tensor and the remaining dimensions.
            ///     \li         If a dimension value is equal to 0, then its output value is going to
            ///                 be copied from the input_shape argument.
            ///
            /// \param[in]  node_name     The node name.
            /// \param[in]  input_shape   The input node shape.
            /// \param[in]  output_shape  The requested output shape for the input node data.
            ///
            /// \return     A vector containig new, valid node shape.
            ///
            std::vector<std::size_t> infer_dimensions(const std::string& node_name,
                                                      const std::vector<std::size_t>& input_shape,
                                                      const std::vector<std::size_t>& output_shape);

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

            /// \brief      Remove empty axes from input tensor.
            ///
            /// \param[in]  node  The node to be squeezed.
            /// \param[in]  axes  The vector defining indexes of axes to be removed.
            ///
            /// \return     The squeezed node.
            ///
            std::shared_ptr<ngraph::Node> squeeze(const std::shared_ptr<ngraph::Node>& node,
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
            std::shared_ptr<ngraph::Node> collapse(const std::shared_ptr<ngraph::Node>& node,
                                                   const std::size_t start_axis,
                                                   const std::size_t end_axis);

            /// \brief      Change shape of input tensor.
            ///
            /// \param[in]  node   The node which shape will be changed.
            /// \param[in]  shape  The new shape for input tensor.
            ///
            /// \return     The node representing reshaped input tensor.
            ///
            std::shared_ptr<ngraph::Node> reshape(const std::shared_ptr<ngraph::Node>& node,
                                                  const AxisVector& axis_order,
                                                  const Shape& shape);

            inline std::shared_ptr<ngraph::Node> reshape(const std::shared_ptr<ngraph::Node>& node,
                                                         const Shape& shape)
            {
                return reshape(node, get_default_axis_vector(node->get_shape().size()), shape);
            }

            /// \brief      Expands node tensor shape with empty axes.
            ///
            /// \param[in]  node                  The node to be expanded.
            /// \param[in]  outermost_axes_count  The number of added outermost axes.
            ///                                   At the front of the shape.
            /// \param[in]  innermost_axes_count  The number of added innermost axes.
            ///                                   At the end of the shape.
            ///
            /// \return     The node with added empty axes.
            ///
            std::shared_ptr<ngraph::Node> add_empty_axes(const std::shared_ptr<ngraph::Node>& node,
                                                         std::size_t outermost_axes_count = 1,
                                                         std::size_t innermost_axes_count = 0);

        } // namespace  reshape
    }     // namespace onnx_import
} // namespace ngraph
