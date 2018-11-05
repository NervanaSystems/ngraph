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

#include "pad.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            namespace kernel
            {
                void pad_4d_float32(float* input,
                                    float* output,
                                    float* pad_value,
                                    const Shape& input_shape,
                                    const Shape& output_shape,
                                    const Shape& padding_below,
                                    const Shape& padding_above,
                                    int arena)
                {
                    pad<float, 4>(input,
                                  output,
                                  pad_value,
                                  input_shape,
                                  output_shape,
                                  padding_below,
                                  padding_above,
                                  arena);
                }
            }
        }
    }
}
