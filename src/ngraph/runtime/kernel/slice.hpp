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
            void slice(T* arg,
                       T* out,
                       const Shape& arg_shape,
                       const Coordinate& lower_bounds,
                       const Coordinate& upper_bounds,
                       const Strides& strides,
                       const Shape& out_shape)
            {
                View input_view(arg_shape, lower_bounds, upper_bounds, strides);
                View output_view(out_shape);

                View::Iterator output_it = output_view.begin();

                for (Coordinate in_coord : input_view)
                {
                    Coordinate out_coord = *output_it;
                    ++output_it;

                    out[output_view.index(out_coord)] = arg[input_view.index(in_coord)];
                }
            }
        }
    }
}
