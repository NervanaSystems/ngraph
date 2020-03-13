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

#include "ngraph/axis_vector.hpp"
#include "ngraph/shape.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace opt_kernel
        {
            class ReshapeIndexer
            {
            public:
                ReshapeIndexer(const Shape& in_shape,
                               const AxisVector& in_axis_order,
                               const Shape& out_shape);
                size_t next();

            private:
                const Shape& m_in_shape;
                const AxisVector& m_in_axis_order;
                const Shape& m_out_shape;
                size_t m_current_input_index;
            };


            // template <typename T>
            // void reshape(const T* in,
            //              T* out,
            //              const Shape& in_shape,
            //              const AxisVector& in_axis_order,
            //              const Shape& out_shape)
            // {
            //     switch (in_shape.size())
            //     {
            //     case 0: reshape_in0<T>(in, out, in_shape, in_axis_order, out_shape); break;
            //     case 1: reshape_in1<T>(in, out, in_shape, in_axis_order, out_shape); break;
            //     case 2: reshape_in2<T>(in, out, in_shape, in_axis_order, out_shape); break;
            //     case 3: reshape_in3<T>(in, out, in_shape, in_axis_order, out_shape); break;
            //     case 4: reshape_in4<T>(in, out, in_shape, in_axis_order, out_shape); break;
            //     case 5: reshape_in5<T>(in, out, in_shape, in_axis_order, out_shape); break;
            //     case 6: reshape_in6<T>(in, out, in_shape, in_axis_order, out_shape); break;
            //     default: reference::reshape(in, out, in_shape, in_axis_order, out_shape); break;
            //     }
            // }
        }
    }
}
