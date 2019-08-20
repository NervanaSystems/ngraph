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

#include "ngraph/axis_set.hpp"
#include "ngraph/node.hpp"

namespace ngraph
{
    namespace builder
    {
        // clang-format off
        /// \brief Sum-based L2 Norm of a Tensor.
        ///
        /// Calculates
        ///
        /// \f$\left(\sum_{i=1}^{N} x_i^2\right)^{0.5}\f$
        ///
        /// Where `i` traverses all of the axes provided in `reduction_axes`
        ///
        /// ## Inputs
        ///
        /// |                  | Type                              | Description                                          |
        /// | ---------------- | --------------------------------- | ---------------------------------------------------- |
        /// | `value`          | \f$E[d_1,\dots,d_n]~(n \geq 0)\f$ | An input tensor of any shape                         |
        /// | `reduction_axes` | AxesSet                           | The axes to eliminate through reduction (0 indexed). |
        ///
        /// ## Output
        ///
        /// | Type                                      | Description                                                                                                      |
        /// | ----------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
        /// | \f$E[\textit{delete}(A,d_1,\dots,d_n)]\f$ | The tensor \f$T\f$, where \f$T\f$ is the input tensor with the `reduction_axes` \f$A\f$ eliminated by reduction. |
        // clang-format on
        std::shared_ptr<Node> l2_norm(const Output<Node>& value, const AxisSet& reduction_axes);

        // clang-format off
        /// \brief Sum-based Mean of a Tensor.
        ///
        /// Calculates
        ///
        /// \f$\sum_{i=1}^{N} \frac{x_i}{N}\f$
        ///
        /// Where `i` traverses all of the axes provided in `reduction_axes`
        ///
        /// ## Inputs
        ///
        /// |                  | Type                              | Description |                                        |
        /// | ---------------- | --------------------------------- | ---------------------------------------------------- |
        /// | `node`           | \f$E[d_1,\dots,d_n]~(n \geq 0)\f$ | An input tensor of any shape                         |
        /// | `reduction_axes` | AxesSet                           | The axes to eliminate through reduction (0 indexed). |
        ///
        /// ## Output
        ///
        /// | Type                                      | Description                                                                                                      |
        /// | ----------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
        /// | \f$E[\textit{delete}(A,d_1,\dots,d_n)]\f$ | The tensor \f$T\f$, where \f$T\f$ is the input tensor with the `reduction_axes` \f$A\f$ eliminated by reduction. |
        // clang-format on
        std::shared_ptr<Node> mean(const Output<Node>& node, const AxisSet& reduction_axes);

        // clang-format off
        /// \brief Sum-based Standard Deviation of a Tensor.
        ///
        /// If bessel_correct is true, calculates
        ///
        /// \f$\sqrt{\frac{\sum_{i=1}^{N}\left(x_i-\bar{x}\right)^2}{N-1}}\f$
        ///
        /// else, calculates
        ///
        /// \f$\sqrt{\frac{\sum_{i=1}^{N}\left(x_i-\bar{x}\right)^2}{N}}\f$
        ///
        /// Where `i` traverses all of the axes provided in `reduction_axes` and \f$\bar{x} = \sum_{i=1}^{N} \frac{x_i}{N}\f$
        ///
        /// ## Inputs
        ///
        /// |                     | Type                              | Description                                                  |
        /// | ------------------- | --------------------------------- | ------------------------------------------------------------ |
        /// | `value`             | \f$E[d_1,\dots,d_n]~(n \geq 0)\f$ | An input tensor of any shape                                 |
        /// | `reduction_axes`    | AxesSet                           | The axes to eliminate through reduction (0 indexed).         |
        /// | `bessel_correction` | bool (default = false)            | Enable Bessel's correction to std_dev for Small sample sizes |
        ///
        /// ## Output
        ///
        /// | Type                                      | Description                                                                                                      |
        /// | ----------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
        /// | \f$E[\textit{delete}(A,d_1,\dots,d_n)]\f$ | The tensor \f$T\f$, where \f$T\f$ is the input tensor with the `reduction_axes` \f$A\f$ eliminated by reduction. |
        // clang-format on
        std::shared_ptr<Node> std_dev(const Output<Node>& value,
                                      const AxisSet& reduction_axes,
                                      const bool bessel_correction = false);

        // clang-format off
        /// \brief Sum-based Variance of a Tensor.
        ///
        /// If bessel_correct is true, calculates
        ///
        /// \f$\frac{\sum_{i=1}^{N}\left(x_i-\bar{x}\right)^2}{N-1}\f$
        ///
        /// else, calculates
        ///
        /// \f$\frac{\sum_{i=1}^{N}\left(x_i-\bar{x}\right)^2}{N}\f$
        ///
        /// Where `i` traverses all of the axes provided in `reduction_axes` and \f$\bar{x} = \sum_{i=1}^{N} \frac{x_i}{N}\f$
        ///
        /// ## Inputs
        ///
        /// |                     | Type                              | Description                                                  |
        /// | ------------------- | --------------------------------- | ------------------------------------------------------------ |
        /// | `value              | \f$E[d_1,\dots,d_n]~(n \geq 0)\f$ | An input tensor of any shape                                 |
        /// | `reduction_axes`    | AxesSet                           | The axes to eliminate through reduction (0 indexed).         |
        /// | `bessel_correction` | bool (default = false)            | Enable Bessel's correction to std_dev for Small sample sizes |
        ///
        /// ## Output
        ///
        /// | Type                                      | Description                                                                                                      |
        /// | ----------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
        /// | \f$E[\textit{delete}(A,d_1,\dots,d_n)]\f$ | The tensor \f$T\f$, where \f$T\f$ is the input tensor with the `reduction_axes` \f$A\f$ eliminated by reduction. |
        // clang-format on
        std::shared_ptr<Node> variance(const Output<Node>& value,
                                       const AxisSet& reduction_axes,
                                       const bool bessel_correction = false);

    } // namespace builder
} // namespace ngraph
