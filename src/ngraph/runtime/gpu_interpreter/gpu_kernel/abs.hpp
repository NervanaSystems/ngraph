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

#include <cmath>

extern "C"
void runVecAbs(float* arg, float* out, size_t count);

namespace ngraph
{
    namespace runtime
    {
        namespace gpu_kernel
        {
            template <typename T>
            void abs(T* arg, T* out, size_t count)
            {
                for (size_t i = 0; i < count; i++)
                {
                    // TODO: generic "abs" doesn't work here for some reason.
                    out[i] = (arg[i] < 0 ? -arg[i] : arg[i]);
                }
            }

            template <>
            inline void abs<float>(float* arg, float* out, size_t count)
            {
		runVecAbs(arg, out, count);
            }
        }
    }
}
