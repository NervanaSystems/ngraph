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
            void reshape(T* arg,
                         T* out,
                         const Shape& in_shape,
                         const AxisVector& in_axis_order,
                         const Shape& out_shape)
            {
                // Unfortunately we don't yet have a constructor for View that lets us pass only source_space_shape
                // and source_axis_order so we have to construct the defaults here.
                Shape in_start_corner(in_shape.size(), 0); // (0,...0)
                Shape in_strides(in_shape.size(), 1);      // (1,...,1)

                View input_view(in_shape,
                                in_start_corner,
                                in_shape,
                                in_strides,
                                in_axis_order);

                View output_view(out_shape);
                View::Iterator output_it = output_view.begin();

                for (Coordinate input_coord : input_view)
                {
                    Coordinate output_coord = *output_it;
                    ++output_it;

                    out[output_view.index(output_coord)] = arg[input_view.index(input_coord)];
                }
            }
        }
    }
}
