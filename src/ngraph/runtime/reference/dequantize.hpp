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

#include <cmath>

#include "ngraph/axis_set.hpp"
#include "ngraph/coordinate_transform.hpp"
#include "ngraph/shape_util.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            template <typename QUANT, typename REAL>
            void dequantize(const QUANT* input,
                            const REAL* scale,
                            const QUANT* offset,
                            REAL* output,
                            const Shape& input_shape,
                            const Shape& scale_offset_shape,
                            const AxisSet& axes)
            {
                CoordinateTransform input_transform(input_shape);
                CoordinateTransform scale_offset_transform(scale_offset_shape);

                for (const Coordinate& input_coord : input_transform)
                {
                    Coordinate scale_offset_coord = project(input_coord, axes);

                    output[input_transform.index(input_coord)] =
                        static_cast<REAL>(
                            (input[input_transform.index(input_coord)] -
                             offset[scale_offset_transform.index(scale_offset_coord)])) *
                        scale[scale_offset_transform.index(scale_offset_coord)];
                }
            }
        }
    }
}
