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

#include <cmath>

#include "ngraph/runtime/reference/autobroadcast_binop.hpp"

using namespace std;

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            /// \brief Reference kernel for mod computation for integer type.
            ///
            /// \tparam T Type of input and output tensors.
            ///
            /// \param arg0 Pointer to the buffer for dividend operand input tensor.
            /// \param arg1 Pointer to the buffer for divisor input tensor.
            /// \param out Pointer to the buffer for output tensor. This must be pre-allocated by
            ///            the caller, and must be large enough to hold a tensor of the correct
            ///            shape.
            /// \param arg0_shape Shape of arg0.
            /// \param arg1_shape Shape of arg1.
            /// \param broadcast_spec auto broadcast type
            template <typename T>
            typename std::enable_if<std::is_integral<T>::value>::type
                mod(const T* arg0,
                    const T* arg1,
                    T* out,
                    const Shape& arg0_shape,
                    const Shape& arg1_shape,
                    const op::AutoBroadcastSpec& broadcast_spec)
            {
                autobroadcast_binop(
                    arg0, arg1, out, arg0_shape, arg1_shape, broadcast_spec, [](T x, T y) -> T {
                        return x % y;
                    });
            }

            /// \brief Reference kernel for mod computation for floating-point type.
            ///
            /// \tparam T Type of input and output tensors.
            ///
            /// \param arg0 Pointer to the buffer for dividend operand input tensor.
            /// \param arg1 Pointer to the buffer for divisor input tensor.
            /// \param out Pointer to the buffer for output tensor. This must be pre-allocated by
            ///            the caller, and must be large enough to hold a tensor of the correct
            ///            shape.
            /// \param arg0_shape Shape of arg0.
            /// \param arg1_shape Shape of arg1.
            /// \param broadcast_spec auto broadcast type
            template <typename T>
            typename std::enable_if<std::is_floating_point<T>::value ||
                                    std::is_same<T, bfloat16>::value ||
                                    std::is_same<T, float16>::value>::type
                mod(const T* arg0,
                    const T* arg1,
                    T* out,
                    const Shape& arg0_shape,
                    const Shape& arg1_shape,
                    const op::AutoBroadcastSpec& broadcast_spec)
            {
                autobroadcast_binop(
                    arg0, arg1, out, arg0_shape, arg1_shape, broadcast_spec, [](T x, T y) -> T {
                        return std::fmod(x, y);
                    });
            }
        }
    }
}
