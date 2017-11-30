// ----------------------------------------------------------------------------
// Copyright 2017 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------

#pragma once

#include <type_traits>

namespace ngraph
{
    namespace runtime
    {
        namespace kernel
        {
            template <typename T>
            void divide(T* arg0, T* arg1, T* out, size_t count)
            {
                for (size_t i = 0; i < count; i++)
                {
                    // The slightly odd way of testing arg1[i] == 0 is because this template is
                    // instantiated at both integral and floating-point types, and == on floating
                    // point will trigger a warning even if it's never actually evaluated.
                    if (!std::is_floating_point<T>::value && (arg1[i] >= 0 && arg1[i] <= 0))
                    {
                        throw std::domain_error("integer division by zero");
                    }
                    out[i] = arg0[i] / arg1[i];
                }
            }
        }
    }
}
