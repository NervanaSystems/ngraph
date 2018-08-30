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

#include "ngraph/axis_vector.hpp"
#include "ngraph/node.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace reshape
        {
            ///
            /// \brief      Gets the AxisVector filled with monotonic increasing sequence.
            ///
            /// \param[in]  data_shape_size  The data shape size.
            /// \param[in]  start_value      The start_value for sequence.
            ///
            /// \return     The filled AxisVector.
            ///
            ngraph::AxisVector get_default_axis_vector(std::size_t data_shape_size,
                                                       std::size_t start_value);

            ///
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
            std::vector<std::size_t> infer_dimensions(std::string node_name,
                                                      std::vector<std::size_t> input_shape,
                                                      std::vector<std::size_t> output_shape);

            /// \brief Permute axes according to specified axes_order parameter.
            ///
            /// \param node The node which axes we want to permute.
            /// \param axes_order The permutation of node tensor axes.
            ///
            /// \return: New node with permuted axes.
            std::shared_ptr<ngraph::Node> reorder_axes(const std::shared_ptr<ngraph::Node>& node,
                                                       std::vector<int> axes_order);

            /// \brief Return transposed tensor (with axes in reversed order).
            ///
            /// \param node Input tensor we want to transpose
            ///
            /// \return: New node with reversed dimensions.
            std::shared_ptr<ngraph::Node> transpose(const std::shared_ptr<ngraph::Node>& node);

        } // namespace  reshape
    }     // namespace onnx_import
} // namespace ngraph
