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

#include <cmath>
#include <utility>

#include "ngraph/coordinate_transform.hpp"
#include "ngraph/shape_util.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            template <typename T>
            void batch_dot(const T* arg0,
                           const T* arg1,
                           T* out,
                           const Shape& arg0_shape,
                           const Shape& arg1_shape,
                           const Shape& out_shape)
            {
                // check for transpose

                const size_t batch_size = arg0_shape[0];
                const size_t arg0_offset = arg0_shape[1] * arg0_shape[2];
                const size_t arg1_offset = arg1_shape[1] * arg1_shape[2];
                for (size_t i = 0; i < batch_size; ++i) {
                    dot(arg0+i*arg0_offset,
                        arg1+i*arg1_offset,
                        Shape{arg0})
                }
            }
        }
    }
}
