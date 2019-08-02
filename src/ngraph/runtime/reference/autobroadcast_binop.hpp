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

#include "ngraph/coordinate_transform.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/shape_util.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            template <typename T, typename U, typename Functor>
            void autobroadcast_binop(const T* arg0,
                                     const T* arg1,
                                     U* out,
                                     const Shape& arg0_shape,
                                     const Shape& arg1_shape,
                                     const op::AutoBroadcastSpec& broadcast_spec,
                                     Functor elementwise_functor)
            {
                switch (broadcast_spec.m_type)
                {
                case op::AutoBroadcastType::NONE:
                    for (size_t i = 0; i < shape_size(arg0_shape); i++)
                    {
                        out[i] = elementwise_functor(arg0[i], arg1[i]);
                    }
                    break;
                case op::AutoBroadcastType::NUMPY:
                    Shape arg0_fake_shape = arg0_shape;
                    Shape arg1_fake_shape = arg1_shape;

                    while (arg0_fake_shape.size() < arg1_fake_shape.size())
                    {
                        arg0_fake_shape.insert(arg0_fake_shape.begin(), 1);
                    }

                    while (arg1_fake_shape.size() < arg0_fake_shape.size())
                    {
                        arg1_fake_shape.insert(arg1_fake_shape.begin(), 1);
                    }

                    Shape arg0_squeezed_shape;
                    Shape arg1_squeezed_shape;
                    AxisSet arg0_squeezed_axes;
                    AxisSet arg1_squeezed_axes;
                    Shape output_shape;

                    for (size_t i = 0; i < arg0_fake_shape.size(); i++)
                    {
                        if (arg0_fake_shape[i] == 1)
                        {
                            arg0_squeezed_axes.insert(i);
                        }
                        else
                        {
                            arg0_squeezed_shape.push_back(arg0_fake_shape[i]);
                        }

                        if (arg1_fake_shape[i] == 1)
                        {
                            arg1_squeezed_axes.insert(i);
                        }
                        else
                        {
                            arg1_squeezed_shape.push_back(arg1_fake_shape[i]);
                        }

                        output_shape.push_back(std::max(arg0_fake_shape[i], arg1_fake_shape[i]));
                    }

                    CoordinateTransform arg0_transform(arg0_squeezed_shape);
                    CoordinateTransform arg1_transform(arg1_squeezed_shape);
                    CoordinateTransform output_transform(output_shape);

                    for (const Coordinate& output_coord : output_transform)
                    {
                        Coordinate arg0_coord = reduce(output_coord, arg0_squeezed_axes);
                        Coordinate arg1_coord = reduce(output_coord, arg1_squeezed_axes);
                        out[output_transform.index(output_coord)] =
                            elementwise_functor(arg0[arg0_transform.index(arg0_coord)],
                                                arg1[arg1_transform.index(arg1_coord)]);
                    }
                }
            }
        }
    }
}
