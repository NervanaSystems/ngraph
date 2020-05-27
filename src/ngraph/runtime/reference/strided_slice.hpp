//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#include "ngraph/check.hpp"
#include "ngraph/coordinate_transform.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/reshape.hpp"
#include "ngraph/runtime/reference/reverse.hpp"
#include "ngraph/runtime/reference/slice.hpp"
#include "ngraph/slice_plan.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            template <typename T>
            void strided_slice(const T* arg, T* out, const Shape& arg_shape, const SlicePlan& sp)
            {
                runtime::AlignedBuffer slice_out_buffer(shape_size(sp.reshape_in_shape) *
                                                        sizeof(T));
                slice<T>(arg,
                         slice_out_buffer.get_ptr<T>(),
                         arg_shape,
                         Coordinate(sp.begins.begin(), sp.begins.end()),
                         Coordinate(sp.ends.begin(), sp.ends.end()),
                         Strides(sp.strides.begin(), sp.strides.end()),
                         sp.reshape_in_shape);

                runtime::AlignedBuffer reshape_out_buffer(shape_size(sp.reshape_out_shape) *
                                                          sizeof(T));
                reshape<T>(slice_out_buffer.get_ptr<T>(),
                           reshape_out_buffer.get_ptr<T>(),
                           sp.reshape_in_shape,
                           get_default_order(sp.reshape_in_shape.size()),
                           sp.reshape_out_shape);

                reverse<T>(reshape_out_buffer.get_ptr<T>(),
                           out,
                           sp.reshape_out_shape,
                           sp.reshape_out_shape,
                           sp.reverse_axes);
            }
        }
    }
}