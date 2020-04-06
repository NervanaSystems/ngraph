//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#include "core/node.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                /// \brief      Compute the log sum of the input tensor's elements along the
                ///             provided axes.
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
                NodeVector reduce_log_sum(const Node& node);

                /// \brief      Compute the log sum exponent of the input tensor's elements along
                ///             the provided axes.
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
                NodeVector reduce_log_sum_exp(const Node& node);

                /// \brief      Compute the L1 norm of the input tensor's element along the provided
                ///             axes.
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
                NodeVector reduce_l1(const Node& node);

                /// \brief      Compute the L2 norm of the input tensor's element along the provided
                ///             axes.
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
                NodeVector reduce_l2(const Node& node);

                /// \brief      Compute the maximum value of the input tensor's elements along the
                ///             provided axes.
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
                NodeVector reduce_max(const Node& node);

                /// \brief      Compute the mean value of the input tensor's elements along the
                ///             provided axes.
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

                /// \brief      Compute the minimum value of the input tensor's elements along the
                ///             provided axes.
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
                NodeVector reduce_min(const Node& node);

                /// \brief      Compute the product of the input tensor's elements along the
                ///             provided axes.
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
                NodeVector reduce_prod(const Node& node);

                /// \brief      Compute the sum of the input tensor's elements along the provided
                ///             axes.
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
                NodeVector reduce_sum(const Node& node);

                /// \brief      Compute the sum square of the input tensor's element along the
                ///             provided axes.
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
                NodeVector reduce_sum_square(const Node& node);

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
