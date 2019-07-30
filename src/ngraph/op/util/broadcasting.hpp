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
#include "ngraph/op/broadcast.hpp"
#include "ngraph/shape.hpp"

namespace ngraph
{
    namespace op
    {
        /// \brief Cast shape of all input nodes for an element-wise operation that requires shape-compatibility
        ///
        /// \param inputs Original list of inputs
        ///
        /// \return Numpy-style broadcasted list of nodes.
        NodeVector numpy_style_broadcast(const NodeVector& inputs)
            NGRAPH_DEPRECATED("Replace with numpy_style_value_broadcast");

        /// \brief Cast shape of all input nodes for an element-wise operation that requires shape-compatibility
        ///
        /// \param values Original list of inputs
        ///
        /// \return Numpy-style broadcasted list of nodes.
        OutputVector numpy_style_broadcast_values(const OutputVector& values);

        /// \brief Cast shape of an output to the requested output shape using NumPy's broadcasting rules
        ///
        /// \param value original value
        /// \param shape requested output shape
        ///
        /// \return Broadcast output.
        std::shared_ptr<Node> numpy_style_broadcast(const Output<Node>& value, const Shape& shape);

        /// \brief Cast shape of two outputs to make them compatible for an element-wise binary operation.
        ///
        /// If necessary the right-hand-side argument will be broadcast to match the shape
        /// of left-hand-side argument. The starting of the mutually equal shape is
        /// specified by the argument "start_match_axis", and if it is not set,
        /// suffix matching is assumed.
        ///
        /// This style of broadcast was used in ONNX Op sets prior to version 7, where it was
        /// replaced by numpy-style broadcasting.
        ///
        /// \param left Node which contain input of binary op.
        /// \param right Node which contain input of binary op.
        /// \param start_match_axis position in shape denoting start of the mutually equal shape
        ///
        /// \return Left and right node after broadcasting.
        NodeVector legacy_style_broadcast_for_binary_operation(const std::shared_ptr<Node>& left,
                                                               const std::shared_ptr<Node>& right,
                                                               size_t start_match_axis)
            NGRAPH_DEPRECATED("Replace with legacy_style_value_broadcast_for_binary_operation");

        /// \brief Cast shape of two outputs to make them compatible for an element-wise binary operation.
        ///
        /// If necessary the right-hand-side argument will be broadcast to match the shape
        /// of left-hand-side argument. The starting of the mutually equal shape is
        /// specified by the argument "start_match_axis", and if it is not set,
        /// suffix matching is assumed.
        ///
        /// This style of broadcast was used in ONNX Op sets prior to version 7, where it was
        /// replaced by numpy-style broadcasting.
        ///
        /// \param left Node which contain input of binary op.
        /// \param right Node which contain input of binary op.
        /// \param start_match_axis position in shape denoting start of the mutually equal shape
        ///
        /// \return Left and right node after broadcasting.
        OutputVector legacy_style_broadcast_values_for_binary_operation(const Output<Node>& left,
                                                                        const Output<Node>& right,
                                                                        size_t start_match_axis);

        /// \brief      Broadcast shape of two nodes to make them compatible for a matrix multiplication.
        ///
        /// \note       This function is reflecting broadcasting behaviour of NumPy's `matmul` operation
        ///             (https://docs.scipy.org/doc/numpy/reference/generated/numpy.matmul.html)
        ///             This mean that only \"stack of matrices\" axes are bidirectionally broadcasted.
        ///             The last two dimension are left untouched.
        ///
        /// \param[in]  left   The Node providing data for the left-hand side of matrix multiplication.
        /// \param[in]  right  The Node providing data for the right-hand side of matrix multiplication.
        ///
        /// \return     The vector containing both nodes broadcasted.
        ///
        NodeVector numpy_style_broadcast_for_matmul_operation(const std::shared_ptr<Node>& left,
                                                              const std::shared_ptr<Node>& right)
            NGRAPH_DEPRECATED("Replace with numpy_style_broadcast_value_for_matmul_operation.");

        /// \brief      Broadcast shape of two nodes to make them compatible for a matrix multiplication.
        ///
        /// \note       This function is reflecting broadcasting behaviour of NumPy's `matmul` operation
        ///             (https://docs.scipy.org/doc/numpy/reference/generated/numpy.matmul.html)
        ///             This mean that only \"stack of matrices\" axes are bidirectionally broadcasted.
        ///             The last two dimension are left untouched.
        ///
        /// \param[in]  left   The Node providing data for the left-hand side of matrix multiplication.
        /// \param[in]  right  The Node providing data for the right-hand side of matrix multiplication.
        ///
        /// \return     The vector containing both outputs broadcasted.
        ///
        OutputVector numpy_style_broadcast_values_for_matmul_operation(const Output<Node>& left,
                                                                       const Output<Node>& right);

        /// \brief Generate a list of broadcast axes.
        ///
        /// \details Informally, a broadcast "adds" axes to the input tensor, replicating
        ///          elements from the input tensor as needed to fill the new dimensions.
        ///          Function calculate which of the output axes are added in this way.
        ///
        /// \param output_shape      The new shape for the output tensor.
        /// \param input_shape       The shape of input tensor.
        /// \param start_match_axis  The axis along which we want to replicate elements.
        ///                          The starting axis position (0-based) int the output
        ///                          shape from which the current shape of the tensor
        ///                          matches the desired new shape.
        ///
        /// \return The indices of added axes.
        AxisSet calculate_broadcast_axes(const Shape& output_shape,
                                         const Shape& input_shape,
                                         std::size_t start_match_axis);

        /// \brief Generate a list of broadcast along axes.
        ///
        /// \details Broadcast "adds" elements along axes to the input tensor, replicating
        ///          elements from the input tensor as needed to fill the new dimensions.
        ///          Function calculate which of the output axes are added in this way.
        ///
        ///          This function will attempt to match shapes, assuming the current shape
        ///          matches the rightmost positions of the desired new shape. This behaviour
        ///          is similar to NumPy's broadcasting.
        ///
        /// \param output_shape The new shape for the output tensor.
        /// \param input_shape  The shape of input tensor.
        ///
        /// \return             The indices of added axes.
        inline AxisSet calculate_broadcast_axes(const Shape& output_shape, const Shape& input_shape)
        {
            return calculate_broadcast_axes(
                output_shape, input_shape, output_shape.size() - input_shape.size());
        }

        inline std::shared_ptr<Node> make_broadcast_node(const Output<Node>& output,
                                                         Shape new_shape)
        {
            return std::make_shared<op::Broadcast>(
                output, new_shape, calculate_broadcast_axes(new_shape, output.get_shape()));
        }

        inline std::shared_ptr<Node> make_broadcast_node(const Output<Node>& value,
                                                         const Shape& new_shape,
                                                         std::size_t start_match_axis)
        {
            return std::make_shared<op::Broadcast>(
                value,
                new_shape,
                calculate_broadcast_axes(new_shape, value.get_shape(), start_match_axis));
        }
    } // namespace  op
} // namespace  ngraph
