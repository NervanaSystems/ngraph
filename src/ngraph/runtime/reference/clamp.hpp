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
#include <cstddef>

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            template <typename T>
            T double_to_T(double x, double float_to_int_converter(double))
            {
                if (std::is_integral<T>())
                {
                    double min_t = static_cast<double>(std::numeric_limits<T>::min());
                    double max_t = static_cast<double>(std::numeric_limits<T>::max());
                    x = std::max(x, min_t);
                    x = std::min(x, max_t);
                    x = float_to_int_converter(x);
                }
                return static_cast<T>(x);
            }

            template <typename T>
            void clamp(const T* arg, T* out, double min, double max, size_t count)
            {
                for (size_t i = 0; i < count; i++)
                {
                    T min_t = double_to_T<T>(min, [](double x) { return std::ceil(x); });
                    T max_t = double_to_T<T>(max, [](double x) { return std::floor(x); });

                    if (arg[i] < min_t)
                    {
                        out[i] = min_t;
                    }
                    else if (arg[i] > max)
                    {
                        out[i] = max_t;
                    }
                    else
                    {
                        out[i] = arg[i];
                    }
                }
            }
        }
    }
}
