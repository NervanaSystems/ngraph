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

#include <memory>

#include "ngraph/axis_set.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/shape.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        /// \brief Cast shape of two nodes to make them compatible for an element-wise binary operation.
        ///
        /// \param left Node which contain input of binary op.
        /// \param right Node which contain input of binary op.
        ///
        /// \return Left and right node after broadcasting.
        NodeVector
            numpy_style_broadcast_for_binary_operation(const std::shared_ptr<ngraph::Node>& left,
                                                       const std::shared_ptr<ngraph::Node>& right);

        /// \brief Cast shape of two nodes to make them compatible for an element-wise binary operation.
        ///
        /// \param inputs Left and right node (inputs of the binary op).
        ///
        /// \return Left and right node after broadcasting.
        inline NodeVector numpy_style_broadcast_for_binary_operation(NodeVector inputs)
        {
            return numpy_style_broadcast_for_binary_operation(inputs.at(0), inputs.at(1));
        }

        /// \brief Cast shape of two nodes to make them compatible for an element-wise binary operation.
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
        NodeVector
            legacy_style_broadcast_for_binary_operation(const std::shared_ptr<ngraph::Node>& left,
                                                        const std::shared_ptr<ngraph::Node>& right,
                                                        std::size_t start_match_axis);

        /// \brief      Broadcast shape of two nodes to make them compatible for a matrix multiplication.
        ///
        /// \note       This function is reflecting broadcasting behaviour of NumPys' `matmul` operation
        ///             \link https://docs.scipy.org/doc/numpy/reference/generated/numpy.matmul.html
        ///             This mean that only \"stack of matrices\" axes are bidirectionally broadcasted.
        ///             The last two dimension are left untouched.
        ///
        /// \param[in]  left   The Node providing data for the left-hand side of matrix multiplication.
        /// \param[in]  right  The Node providing data for the right-hand side of matrix multiplication.
        ///
        /// \return     The vector containing both nodes broadcasted.
        ///
        NodeVector
            numpy_style_broadcast_for_matmul_operation(const std::shared_ptr<ngraph::Node>& left,
                                                       const std::shared_ptr<ngraph::Node>& right);

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

        inline std::shared_ptr<ngraph::Node>
            make_broadcast_node(const std::shared_ptr<ngraph::Node>& node, ngraph::Shape new_shape)
        {
            return std::make_shared<ngraph::op::Broadcast>(
                node, new_shape, calculate_broadcast_axes(new_shape, node->get_shape()));
        }

        inline std::shared_ptr<ngraph::Node>
            make_broadcast_node(const std::shared_ptr<ngraph::Node>& node,
                                ngraph::Shape new_shape,
                                std::size_t start_match_axis)
        {
            return std::make_shared<ngraph::op::Broadcast>(
                node,
                new_shape,
                calculate_broadcast_axes(new_shape, node->get_shape(), start_match_axis));
        }
    } // namespace  onnx_import
} // namespace  ngraph
