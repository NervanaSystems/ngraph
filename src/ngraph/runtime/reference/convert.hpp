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
    namespace runtime
    {
        namespace reference
        {
            template <typename TI, typename TO>
            void convert(const TI* arg, TO* out, size_t count)
            {
                for (size_t i = 0; i < count; ++i)
                {
                    out[i] = static_cast<TO>(arg[i]);
                }
            }

            template <typename T>
            void convert_to_bool(const T* arg, char* out, size_t count)
            {
                for (size_t i = 0; i < count; ++i)
                {
                    out[i] = static_cast<char>(static_cast<bool>(arg[i]));
                }
            }

            template <typename TI>
            void convert_float_to_bf16(void* arg, void* out, size_t count)
            {
                int* a = static_cast<int*>(arg);
                char16_t* b = static_cast<char16_t*>(out);

                for (; count != 0; b++, count--, a++)
                {
                    *b = (a[0] >> 16) & 0xffff;
                }
            }

            template <typename TI>
            void convert_bf16_to_float(void* arg, void* out, size_t count)
            {
                char16_t* a = static_cast<char16_t*>(arg);
                int* b = static_cast<int*>(out);

                for (; count != 0; a++, b++, count--)
                {
                    *b = (a[0] & 0xffff) << 16;
                }
            }
        }
    }
}
