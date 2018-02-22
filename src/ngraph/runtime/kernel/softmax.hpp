/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#pragma once

#include <cmath>

namespace ngraph
{
    namespace runtime
    {
        namespace kernel
        {
            template <typename T>
            void softmax(T* arg, T* out, size_t count)
            {
                T m = arg[0];
                for (size_t i = 1; i < count; i++)
                {
                    if (arg[i] > m)
                    {
                        m = arg[i];
                    }
                }
                T d = 0;
                for (size_t i = 0; i < count; i++)
                {
                    T e = std::exp(arg[i] - m);
                    out[i] = e;
                    d += e;
                }
                d = 1 / d;
                for (size_t i = 0; i < count; i++)
                {
                    out[i] *= d;
                }
            }
        }
    }
}
