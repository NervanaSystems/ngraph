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
#include <string>
#include <vector>

#include "ngraph/axis_vector.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/util/reshape.hpp"
#include "ngraph/shape.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace reshape
        {
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

            /// \brief      Expands node tensor shape with empty axis at
            ///             specified position.
            ///
            /// \param[in]  node  The node to be expanded.
            /// \param[in]  axis  The position in the expanded axes where the
            ///                   new axis is placed.
            ///
            /// \return     The node with added empty axis.
            ///
            std::shared_ptr<ngraph::Node> expand_dims(const std::shared_ptr<ngraph::Node>& node,
                                                      std::size_t axis = 0);

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

        } // namespace  reshape
    }     // namespace onnx_import
} // namespace ngraph
