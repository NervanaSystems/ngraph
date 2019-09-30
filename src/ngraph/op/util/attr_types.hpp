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

namespace ngraph
{
    namespace op
    {
        /// \brief Modes for the `Pad` operator.
        enum class PadMode
        {
            CONSTANT = 0,
            EDGE,
            REFLECT,
            SYMMETRIC
        };

        /// \brief Padding Type used for `Convolution` and `Pooling`
        ///
        /// Follows ONNX padding type definitions
        /// EXPLICIT   - Pad dimensions are explicity specified
        /// SAME_LOWER - Pad dimensions computed to match input shape
        ///              Ceil(num_dims/2) at the beginning and
        ///              Floor(num_dims/2) at the end
        /// SAME_UPPER - Pad dimensions computed to match input shape
        ///              Floor(num_dims/2) at the beginning and
        ///              Ceil(num_dims/2) at the end
        /// VALID      - No padding
        ///
        enum class PadType
        {
            EXPLICIT = 0,
            SAME_LOWER,
            SAME_UPPER,
            VALID,
            AUTO = SAME_UPPER,
            NOTSET = EXPLICIT,
        };

        /// \brief Specifies the algorithm to use for implicit broadcasting of a tensor
        ///        to align with another tensor
        ///
        /// NONE  - No implicit broadcasting of tensor
        /// NUMPY - Numpy-style implicit broadcasting
        ///         (https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
        ///         Right-align dimensions of the two tensors, with missing dimensions
        ///         treated as size 1 dimensions. After alignment, for each dimension,
        ///         their sizes should either match or one of them should be of size 1.
        ///         Size 1 dimension will be implicitly broadcast to match the other
        ///         size.
        ///
        ///         E.g.,
        ///              A: Shape(2, 1, 6)
        ///              B: Shape(   3, 1)
        ///         Result: Shape(2, 3, 6)
        ///
        ///              A: Shape(2, 1, 6)
        ///              B: Shape(   3, 1)
        ///         Result: Shape(2, 3, 6)
        ///
        /// TODO: Add more implicit broadcast modes used by frameworks
        enum class AutoBroadcastType
        {
            NONE = 0,
            NUMPY
        };

        /// \brief Specifies how eps is combined with L2 value
        enum class EpsMode
        {
            // Add bias to norm
            ADD,
            // Calculate max of norm and bias
            MAX
        };

        /// \brief Implicit broadcast specification
        struct AutoBroadcastSpec
        {
            AutoBroadcastSpec()
                : m_type(AutoBroadcastType::NONE)
                , m_axis(0)
            {
            }
            AutoBroadcastSpec(AutoBroadcastType type)
                : m_type(type)
                , m_axis(0)
            {
            }
            AutoBroadcastSpec(AutoBroadcastType type, size_t axis)
                : m_type(type)
                , m_axis(axis)
            {
            }

            AutoBroadcastType m_type; // Implicit broadcasting algorithm
            size_t m_axis;            // Axis to start alignment on
        };
    }
}
