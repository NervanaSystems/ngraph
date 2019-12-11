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

#include <algorithm>   // std::generate
#include <cmath>       // std::floor, std::min
#include <cstddef>     // std::size_t
#include <cstdint>     // std::int64_t
#include <iterator>    // std::begin, std::end
#include <memory>      // std::shared_ptr, std::make_shared
#include <type_traits> // std::enable_if
#include <vector>

#include "core/node.hpp"
#include "default_opset.hpp"
#include "ngraph/op/util/broadcasting.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/type/element_type.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace common
        {
            const ngraph::element::Type& get_ngraph_element_type(std::int64_t onnx_type);

            /// \brief      Return a monotonic sequence.
            ///
            /// \note       Limitations: this function may not work for very large integer values
            ///             (near numerical limits).
            ///
            /// \param[in]  start_value  The start value of the sequence.
            /// \param[in]  end_value    The end value of the sequence.
            /// \param[in]  step         The step value for the sequence.
            ///
            /// \tparam     T            The data value type.
            ///
            /// \return     The vector with monotonic sequence
            template <typename T>
            std::vector<T> get_monotonic_range(T end_value, T start_value = T{0}, T step = T{1})
            {
                auto value_count =
                    static_cast<std::size_t>(std::floor((end_value - start_value) / step));

                std::vector<T> range(value_count);

                // Calculate initial value (one step below starting value)
                size_t n = start_value - step;
                // Generate a vector of values by adding step to previous value
                std::generate(
                    std::begin(range), std::end(range), [&n, &step]() -> T { return n += step; });

                return range;
            }

            /// \brief      Handle out of range axis.
            ///
            /// \param[in]  node         The node with requested axis.
            /// \param[in]  axis         The requested axis value.
            /// \param[in]  tensor_rank  The corresponding tensor rank.
            ///
            /// \return    Checking if axis is in range [-tensor_rank, tensor_rank-1], otherwise
            /// returns error.
            ///            If negative axis, it counts from the last to the first axis, by adding
            ///            tensor_rank to axis.
            ///
            std::size_t validate_axis(const ngraph::onnx_import::Node& node,
                                      std::int64_t axis,
                                      std::int64_t tensor_rank);

            /// \brief      Handle out of range axis.
            ///
            /// \param[in]  node            The node with requested axis.
            /// \param[in]  axis            The requested axis value.
            /// \param[in]  tensor_rank     The corresponding tensor rank.
            /// \param[in]  axis_range_min  The min value of accepted range for axis.
            /// \param[in]  axis_range_max  The max value of accepted range for axis.
            ///
            /// \return     Checking if axis is in range [axis_range_min, axis_range_max], otherwise
            /// returns error.
            ////            If negative axis, it counts from the last to the first axis, by adding
            /// tensor_rank to axis.
            ///
            std::size_t validate_axis(const ngraph::onnx_import::Node& node,
                                      std::int64_t axis,
                                      std::int64_t tensor_rank,
                                      std::int64_t axis_range_min,
                                      std::int64_t axis_range_max);

            /// \brief      Handle out of range axes in vector.
            ///
            /// \param[in]  node         The node with requested axes.
            /// \param[in]  axes         The requested vector of axes.
            /// \param[in]  tensor_rank  The corresponding tensor rank.
            ///
            /// \return     If any negative axis in vector, it counts from the last to the first
            /// axis, by adding tensor_rank to axis.
            ///
            std::vector<std::size_t> validate_axes(const ngraph::onnx_import::Node& node,
                                                   std::vector<std::int64_t> axes,
                                                   std::int64_t tensor_rank);

            /// \brief Return the outputs of the node as vector.
            ///
            /// \param[in] node            Node with multiple outputs.
            ///
            /// \return                    Vector of outputs of input node.
            ngraph::NodeVector get_outputs(const std::shared_ptr<ngraph::Node>& node);

            /// \brief Creates a shifted square identity matrix.
            /// \note Shifting in the context of this operator means that
            ///       the matrix can be created with elements equal to 1 not only in the main
            ///       diagonal. Shifting adds an offset and moves the diagonal up or down
            ///
            /// \param[in] output_shape Shape of the resulting matrix.
            /// \param[in] output_type Element type of the resulting matrix.
            /// \param[in] shift Shifting of diagonal.
            ///
            /// \return A Constant node representing shifted identity matrix.
            template <typename T = double>
            std::shared_ptr<ngraph::op::Constant>
                shifted_square_identity(const Shape output_shape,
                                        const element::Type& output_type,
                                        const std::int64_t shift)
            {
                std::vector<T> identity_matrix(shape_size(output_shape), T{0});
                std::int64_t rows = output_shape[0];
                std::int64_t cols = output_shape[1];
                for (std::int64_t row = 0; row < rows; ++row)
                {
                    const std::int64_t diagonal_element_idx = (row * cols) + row + shift;
                    if (row + shift < 0)
                    {
                        continue;
                    }
                    else if (row + shift >= cols)
                    {
                        break;
                    }
                    identity_matrix.at(diagonal_element_idx) = T{1};
                }

                return std::make_shared<default_opset::Constant>(
                    output_type, output_shape, identity_matrix);
            }

            /// \brief Creates a square identity matrix.
            ///
            /// \param[in] n Order of the resulting matrix.
            ///
            /// \return A Constant node representing identity matrix with shape (n, n).
            template <typename T = double>
            std::shared_ptr<default_opset::Constant> square_identity(const size_t n,
                                                                     const element::Type& type)
            {
                return shifted_square_identity(Shape{n, n}, type, 0);
            }

        } // namespace  common
    }     // namespace onnx_import
} // namespace ngraph
