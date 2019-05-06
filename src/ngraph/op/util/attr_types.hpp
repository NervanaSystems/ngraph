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

namespace ngraph
{
    namespace op
    {
        /// \brief Modes for the `Pad` operator.
        enum class PadMode
        {
            CONSTANT = 0,
            EDGE,
            REFLECT
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
            NOTSET = EXPLICIT
        };
    }
}
