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

#include "ngraph/common.hpp"
#include "ngraph/view.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace kernel
        {
            template <typename T>
            void reduce(T* arg0,
                        T* arg1, // TODO: really we should just pass a T here.
                        T* out,
                        const Shape& in_shape,
                        const Shape& out_shape,
                        const AxisSet& reduction_axes,
                        std::function<T(T, T)> reduction_function)
            {
                View output_view(out_shape);

                for (Coordinate output_coord : output_view)
                {
                    out[output_view.index(output_coord)] = *arg1;
                }

                View input_view(in_shape);

                for (Coordinate input_coord : input_view)
                {
                    Coordinate output_coord = project_coordinate(input_coord, reduction_axes);
                    size_t input_index = input_view.index(input_coord);
                    size_t output_index = output_view.index(output_coord);

                    out[output_index] = reduction_function(out[output_index], arg0[input_index]);
                }
            }
        }
    }
}
