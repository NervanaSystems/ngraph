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

#include "ngraph/coordinate_transform.hpp"
#include "ngraph/shape_util.hpp"
#include "ngraph/type/bfloat16.hpp"
#include "ngraph/type/float16.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            template <typename T>
            void cumsum(const T* arg,
                        T* out,
                        const Shape& in_shape,
                        const Shape& out_shape,
                        const int64_t axis,
                        const int exclusive,
                        const int reverse)
            {
                auto temp_shape = reduce(out_shape, AxisSet{static_cast<size_t>(axis)});
                CoordinateTransform temp_transform(temp_shape);
                std::vector<T> cs(shape_size(temp_shape));

                for (const Coordinate& output_coord : temp_transform)
                {
                    out[temp_transform.index(output_coord)] = 0;
                    cs[temp_transform.index(output_coord)] = 0;
                }

                CoordinateTransform input_transform(in_shape);
                T prev = 0;
                for (const Coordinate& input_coord : input_transform)
                {
                    // TODO (pthoreho): Add support for exclsuive and reverse mode

                    // points to the current element in the input tensor
                    T current = arg[input_transform.index(input_coord)];
                    // holds the reference of the output corrosponding to the given input tensor
                    T& z = out[input_transform.index(input_coord)];
                    z = prev + current;
                    // captures the result of the current output for cummulative sum in the
                    // subsequent sum
                    prev = z;
                }
            }
        }
    }
}
