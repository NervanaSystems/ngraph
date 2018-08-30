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

#include <cstddef>
#include <stdexcept>
#include <type_traits>

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            // NOTE: Execution throws `std::domain_error` if either a non-integral value or an out-of-bounds
            // value is detected in the input tensor.

            // In English: return type is void and T must be an integral type.
            template <typename T>
            typename std::enable_if<std::is_integral<T>::value>::type
                divide(const T* arg0, const T* arg1, T* out, size_t count)
            {
                for (size_t i = 0; i < count; i++)
                {
                    if (arg1[i] == 0)
                    {
                        throw std::domain_error("integer division by zero");
                    }
                    out[i] = arg0[i] / arg1[i];
                }
            }

            // In English: return type is void and T must be a floating point type.
            template <typename T>
            typename std::enable_if<std::is_floating_point<T>::value>::type
                divide(const T* arg0, const T* arg1, T* out, size_t count)
            {
                for (size_t i = 0; i < count; i++)
                {
                    // TODO: Here we do not check for div by zero, so we'll get +-inf here
                    // if arg1[i] == 0. Is that the right thing to do? Jury's still out.
                    out[i] = arg0[i] / arg1[i];
                }
            }
        }
    }
}
