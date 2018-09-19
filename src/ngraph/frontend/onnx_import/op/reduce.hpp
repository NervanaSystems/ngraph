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

#include "ngraph/node_vector.hpp"
#include "ngraph/op/abs.hpp"
#include "ngraph/op/exp.hpp"
#include "ngraph/op/log.hpp"
#include "ngraph/op/max.hpp"
#include "ngraph/op/min.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/product.hpp"
#include "ngraph/op/sqrt.hpp"
#include "ngraph/op/sum.hpp"

#include "core/node.hpp"
#include "utils/broadcasting.hpp"
#include "utils/reduction.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            /// \brief      Compute the log sum of the input tensor's elements along the provided axes.
            ///
            /// \par Overview
            ///     The output tensor has the same rank as the input if Node attribute keepdims
            ///     equals 1. If keepdims equals 0, then the output tensor have the reduced
            ///     dimension pruned.
            ///
            /// \param[in]  node  The ONNX node representing operation.
            ///
            /// \return     The nGraph node equivalent of the ONNX operation.
            ///
            inline NodeVector reduce_log_sum(const Node& node)
            {
                auto sum_node = reduction::make_ng_reduction_op<ngraph::op::Sum>(
                    node, node.get_ng_inputs().at(0));
                return {std::make_shared<ngraph::op::Log>(sum_node)};
            }

            /// \brief      Compute the log sum exponent of the input tensor's elements along the provided axes.
            ///
            /// \par Overview
            ///     The output tensor has the same rank as the input if Node attribute keepdims
            ///     equals 1. If keepdims equals 0, then the output tensor have the reduced
            ///     dimension pruned.
            ///
            /// \param[in]  node  The ONNX node representing operation.
            ///
            /// \return     The nGraph node equivalent of the ONNX operation.
            ///
            inline NodeVector reduce_log_sum_exp(const Node& node)
            {
                auto exp_node = std::make_shared<ngraph::op::Exp>(node.get_ng_inputs().at(0));
                auto sum_node = reduction::make_ng_reduction_op<ngraph::op::Sum>(node, exp_node);
                return {std::make_shared<ngraph::op::Log>(sum_node)};
            }

            /// \brief      Compute the L1 norm of the input tensor's element along the provided axes.
            ///
            /// \par Overview
            ///     The output tensor has the same rank as the input if Node attribute keepdims
            ///     equals 1. If keepdims equals 0, then the output tensor have the reduced
            ///     dimension pruned.
            ///
            /// \param[in]  node  The ONNX node representing operation.
            ///
            /// \return     The nGraph node equivalent of the ONNX operation.
            ///
            inline NodeVector reduce_l1(const Node& node)
            {
                auto abs_node = std::make_shared<ngraph::op::Abs>(node.get_ng_inputs().at(0));
                return {reduction::make_ng_reduction_op<ngraph::op::Sum>(node, abs_node)};
            }

            /// \brief      Compute the L2 norm of the input tensor's element along the provided axes.
            ///
            /// \par Overview
            ///     The output tensor has the same rank as the input if Node attribute keepdims
            ///     equals 1. If keepdims equals 0, then the output tensor have the reduced
            ///     dimension pruned.
            ///
            /// \param[in]  node  The ONNX node representing operation.
            ///
            /// \return     The nGraph node equivalent of the ONNX operation.
            ///
            inline NodeVector reduce_l2(const Node& node)
            {
                NodeVector ng_inputs{node.get_ng_inputs()};
                auto square_node =
                    std::make_shared<ngraph::op::Multiply>(ng_inputs.at(0), ng_inputs.at(0));
                auto sum_node = reduction::make_ng_reduction_op<ngraph::op::Sum>(node, square_node);
                return {std::make_shared<ngraph::op::Sqrt>(sum_node)};
            }

            /// \brief      Compute the maximum value of the input tensor's elements along the provided axes.
            ///
            /// \par Overview
            ///     The output tensor has the same rank as the input if Node attribute keepdims
            ///     equals 1. If keepdims equals 0, then the output tensor have the reduced
            ///     dimension pruned.
            ///
            /// \param[in]  node  The ONNX node representing operation.
            ///
            /// \return     The nGraph node equivalent of the ONNX operation.
            ///
            inline NodeVector reduce_max(const Node& node)
            {
                return {reduction::make_ng_reduction_op<ngraph::op::Max>(
                    node, node.get_ng_inputs().at(0))};
            }

            /// \brief      Compute the mean value of the input tensor's elements along the provided axes.
            ///
            /// \par Overview
            ///     The output tensor has the same rank as the input if Node attribute keepdims
            ///     equals 1. If keepdims equals 0, then the output tensor have the reduced
            ///     dimension pruned.
            ///
            /// \param[in]  node  The ONNX node representing operation.
            ///
            /// \return     The nGraph node equivalent of the ONNX operation.
            ///
            NodeVector reduce_mean(const Node& node);

            /// \brief      Compute the minimum value of the input tensor's elements along the provided axes.
            ///
            /// \par Overview
            ///     The output tensor has the same rank as the input if Node attribute keepdims
            ///     equals 1. If keepdims equals 0, then the output tensor have the reduced
            ///     dimension pruned.
            ///
            /// \param[in]  node  The ONNX node representing operation.
            ///
            /// \return     The nGraph node equivalent of the ONNX operation.
            ///
            inline NodeVector reduce_min(const Node& node)
            {
                return {reduction::make_ng_reduction_op<ngraph::op::Min>(
                    node, node.get_ng_inputs().at(0))};
            }

            /// \brief      Compute the product of the input tensor's elements along the provided axes.
            ///
            /// \par Overview
            ///     The output tensor has the same rank as the input if Node attribute keepdims
            ///     equals 1. If keepdims equals 0, then the output tensor have the reduced
            ///     dimension pruned.
            ///
            /// \param[in]  node  The ONNX node representing operation.
            ///
            /// \return     The nGraph node equivalent of the ONNX operation.
            ///
            inline NodeVector reduce_prod(const Node& node)
            {
                return {reduction::make_ng_reduction_op<ngraph::op::Product>(
                    node, node.get_ng_inputs().at(0))};
            }

            /// \brief      Compute the sum of the input tensor's elements along the provided axes.
            ///
            /// \par Overview
            ///     The output tensor has the same rank as the input if Node attribute keepdims
            ///     equals 1. If keepdims equals 0, then the output tensor have the reduced
            ///     dimension pruned.
            ///
            /// \param[in]  node  The ONNX node representing operation.
            ///
            /// \return     The nGraph node equivalent of the ONNX operation.
            ///
            inline NodeVector reduce_sum(const Node& node)
            {
                return {reduction::make_ng_reduction_op<ngraph::op::Sum>(
                    node, node.get_ng_inputs().at(0))};
            }

            /// \brief      Compute the sum square of the input tensor's element along the provided axes.
            ///
            /// \par Overview
            ///     The output tensor has the same rank as the input if Node attribute keepdims
            ///     equals 1. If keepdims equals 0, then the output tensor have the reduced
            ///     dimension pruned.
            ///
            /// \param[in]  node  The ONNX node representing operation.
            ///
            /// \return     The nGraph node equivalent of the ONNX operation.
            ///
            inline NodeVector reduce_sum_square(const Node& node)
            {
                NodeVector ng_inputs{node.get_ng_inputs()};
                auto square_node =
                    std::make_shared<ngraph::op::Multiply>(ng_inputs.at(0), ng_inputs.at(0));
                return {reduction::make_ng_reduction_op<ngraph::op::Sum>(node, square_node)};
            }

        } // namespace op
    }     // namespace onnx_import
} // namespace ngraph
