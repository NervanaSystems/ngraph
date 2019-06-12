//*****************************************************************************
// Copyright 2018-2019 Intel Corporation
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

#include <cstddef>
#include <iterator>
#include <memory>
#include <vector>

#include "matmul_factory.hpp"
#include "ngraph/builder/make_constant.hpp"
#include "ngraph/builder/quantization/quantized_linear_matmul.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/dot.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/op/util/broadcasting.hpp"
#include "utils/reshape.hpp"

using namespace ngraph::onnx_import::matmul;

/// \brief      Slice the sub matrix from the input tensor.
///
/// \param[in]  node  The input tensor. Must be at most of rank 3.
/// \param[in]  idx   The index on the first axis, at which to slice sub-matrix.
///
/// \return     The node representing sub matrix.
///
static std::shared_ptr<ngraph::Node> get_sub_matrix(const std::shared_ptr<ngraph::Node>& node,
                                                    std::size_t idx)
{
    const ngraph::Shape& shape{node->get_shape()};
    if (shape.size() < 3)
    {
        return node;
    }
    // Below bounds defines the sub_matrix through ranges for each input node axis.
    ngraph::Coordinate lower_bounds(shape.size());
    ngraph::Coordinate upper_bounds = shape;
    // We assume `node` tensor is of rank equal 3, thus we slice the sub-matrix lying in the last
    // two dimensions at index `idx` of first axis.
    lower_bounds.at(0) = idx;
    upper_bounds.at(0) = idx + 1;

    auto sub_matrix = std::shared_ptr<ngraph::Node>{
        std::make_shared<ngraph::op::Slice>(node, lower_bounds, upper_bounds)};
    // Remove first single entry dim.
    return ngraph::onnx_import::reshape::squeeze(sub_matrix);
}

std::shared_ptr<ngraph::Node> MatmulFactory::get_left()
{
    return m_inputs.at(0);
}

std::shared_ptr<ngraph::Node> MatmulFactory::get_right()
{
    return m_inputs.at(1);
}

ngraph::NodeVector MatmulFactory::make_matmul_op()
{
    auto left = get_left();
    auto right = get_right();

    std::size_t left_rank{left->get_shape().size()};
    std::size_t right_rank{right->get_shape().size()};

    if (left_rank == 0 || right_rank == 0)
    {
        NGRAPH_WARN << (m_onnx_node) << " "
                    << "ONNX standard doesn't allow scalar operands, however nGraph "
                       "accepts them. Consider use of element-wise multiplication instead "
                       "to conform with ONNX standard.";
    }

    // First (easy) case that is already internally handled by Ngraph Dot operator.
    // Multiply two tensors where both of them has rank lower equal 2.
    if (left_rank <= 2 && right_rank <= 2)
    {
        return NodeVector{make_dot(left, right)};
    }

    // Second case:
    // Multiply two tensors where at least one of them is rank greater equal 3.

    // Broadcast input arguments only if both of them are not vectors.
    if (left_rank > 1 && right_rank > 1)
    {
        const NodeVector& broadcasted_nodes =
            ngraph::op::numpy_style_broadcast_for_matmul_operation(left, right);

        left = broadcasted_nodes.at(0);
        right = broadcasted_nodes.at(1);
    }
    const auto& left_shape = left->get_shape();
    const auto& right_shape = right->get_shape();

    // Collapse both tensors _stack of matrices_ axes (all except the last two).
    // This will make easier further dot product calculations.
    if (left_shape.size() > 3)
    {
        left = onnx_import::reshape::collapse(left, 0, left_shape.size() - 3);
    }
    if (right_shape.size() > 3)
    {
        right = onnx_import::reshape::collapse(right, 0, right_shape.size() - 3);
    }

    // Perform multiple small dot products
    std::size_t groups = left->get_shape().at(0);
    // If we haven't broadcast earlier this means that one of the inputs is a vector,
    // thus the number of groups is defined by the shape of the bigger tensor.
    if (right->get_shape().size() > left->get_shape().size())
    {
        groups = right->get_shape().at(0);
    }
    NodeVector small_dots(groups);

    for (std::size_t g = 0; g < groups; ++g)
    {
        const auto sliced_left = get_sub_matrix(left, g);
        const auto sliced_right = get_sub_matrix(right, g);
        auto sub_dot = make_dot(sliced_left, sliced_right);

        // Expand sub_dot result with single empty outermost axis, in order to
        // later concatenate sub_dots at this axis.
        small_dots.at(g) = onnx_import::reshape::expand_dims(sub_dot);
    }

    // Concatenate sub_dots on groups axis.
    auto result = std::make_shared<ngraph::op::Concat>(small_dots, 0);

    if (left_shape.size() <= 3 && right_shape.size() <= 3)
    {
        return {result};
    }
    // Expand result _stack of matrices_ axes to get expected result shape.
    else
    {
        const Shape& shape{result->get_shape()};
        Shape result_shape(std::next(std::begin(shape)), std::end(shape));
        result_shape.insert(std::begin(result_shape),
                            std::begin(left_shape),
                            std::next(std::begin(left_shape), left_shape.size() - 2));
        return {std::make_shared<ngraph::op::Reshape>(
            result, ngraph::get_default_order(shape.size()), result_shape)};
    }
}

std::shared_ptr<ngraph::Node> MatmulFactory::make_dot(const std::shared_ptr<ngraph::Node>& left,
                                                      const std::shared_ptr<ngraph::Node>& right)
{
    return std::make_shared<ngraph::op::Dot>(left, right);
}

std::shared_ptr<ngraph::Node> QLinearMatmulFactory::get_right()
{
    return m_inputs.at(3);
}

std::shared_ptr<ngraph::Node>
    QLinearMatmulFactory::make_dot(const std::shared_ptr<ngraph::Node>& left,
                                   const std::shared_ptr<ngraph::Node>& right)
{
    return ngraph::builder::quantization::QuantizedLinearMatmul(left,
                                                                right,
                                                                m_inputs.at(1),
                                                                m_inputs.at(2),
                                                                m_inputs.at(4),
                                                                m_inputs.at(5),
                                                                m_inputs.at(6),
                                                                m_inputs.at(7));
}

std::shared_ptr<ngraph::Node>
    MatmulIntegerFactory::make_dot(const std::shared_ptr<ngraph::Node>& left,
                                   const std::shared_ptr<ngraph::Node>& right)
{
    auto num_inputs = m_inputs.size();

    if (num_inputs == 2)
    {
        return ngraph::builder::quantization::QuantizedLinearMatmulInteger(left, right);
    }

    auto left_zero_point = m_inputs.at(2);
    auto right_zero_point = ngraph::builder::make_constant(right->get_element_type(), Shape{}, 0);
    if (num_inputs == 4)
    {
        right_zero_point = m_inputs.at(3);
    }

    return ngraph::builder::quantization::QuantizedLinearMatmulInteger(
        left, right, left_zero_point, right_zero_point);
}
