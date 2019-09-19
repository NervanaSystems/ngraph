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

#include <Eigen/Dense>
#include <cmath>
#include <omp.h>
#include <utility>

#include "ngraph/runtime/reference/broadcast.hpp"
#include "ngraph/shape_util.hpp"
#include "ngraph/util.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace gcpu
        {
            namespace kernel
            {
#ifdef PARALLEL
                static std::tuple<size_t, size_t> get_start_finish(size_t size)
                {
                    const size_t nthreads = omp_get_num_threads();
                    const size_t ithread = omp_get_thread_num();
                    const size_t start = ithread * size / nthreads;
                    const size_t finish = (ithread + 1) * size / nthreads;
                    return std::make_tuple(start, finish);
                }
#endif
                template <typename T>
                void broadcast_2d(const T* in,
                                  T* out,
                                  const Shape& in_shape,
                                  const Shape& out_shape,
                                  const AxisSet& broadcast_axes)
                {
                    size_t index[2];
                    size_t* out_index =
                        (broadcast_axes.find(0) == broadcast_axes.end() ? &index[0] : &index[1]);
                    for (index[0] = 0; index[0] < out_shape[0]; ++index[0])
                    {
                        for (index[1] = 0; index[1] < out_shape[1]; ++index[1])
                        {
                            out[index[0] * out_shape[1] + index[1]] = in[*out_index];
                        }
                    }
                }

                // #define PARALLEL
                template <typename T>
                void broadcast_3d(const T* in,
                                  T* out,
                                  const Shape& in_shape,
                                  const Shape& out_shape,
                                  const AxisSet& broadcast_axes)
                {
#ifdef PARALLEL
#pragma omp parallel
#endif
                    {
                        size_t start;
                        size_t finish;
#ifdef PARALLEL
                        std::tie(start, finish) = get_start_finish(out_shape[0]);
#else
                        start = 0;
                        finish = out_shape[0];
#endif
                        size_t index[3];
                        size_t* out_index = 0;
                        for (size_t i = 0; i < 3; i++)
                        {
                            if (broadcast_axes.count(i) == 0)
                            {
                                out_index = &index[i];
                                break;
                            }
                        }
                        for (index[0] = start; index[0] < finish; ++index[0])
                        {
                            for (index[1] = 0; index[1] < out_shape[1]; ++index[1])
                            {
                                for (index[2] = 0; index[2] < out_shape[2]; ++index[2])
                                {
                                    out[index[0] * out_shape[1] * out_shape[2] +
                                        index[1] * out_shape[2] + index[2]] = in[*out_index];
                                }
                            }
                        }
                    }
                }

                template <typename T>
                void broadcast_4d(const T* in,
                                  T* out,
                                  const Shape& in_shape,
                                  const Shape& out_shape,
                                  const AxisSet& broadcast_axes)
                {
                    size_t index[4];
                    size_t* out_index = 0;
                    for (size_t i = 0; i < 4; i++)
                    {
                        if (broadcast_axes.count(i) == 0)
                        {
                            out_index = &index[i];
                            break;
                        }
                    }
                    for (index[0] = 0; index[0] < out_shape[0]; ++index[0])
                    {
                        for (index[1] = 0; index[1] < out_shape[1]; ++index[1])
                        {
                            for (index[2] = 0; index[2] < out_shape[2]; ++index[2])
                            {
                                for (index[3] = 0; index[3] < out_shape[3]; ++index[3])
                                {
                                    out[index[0] * out_shape[1] * out_shape[2] * out_shape[3] +
                                        index[1] * out_shape[2] * out_shape[3] +
                                        index[2] * out_shape[3] + index[3]] = in[*out_index];
                                }
                            }
                        }
                    }
                }

                template <typename T>
                void broadcast_5d(const T* in,
                                  T* out,
                                  const Shape& in_shape,
                                  const Shape& out_shape,
                                  const AxisSet& broadcast_axes)
                {
                    size_t index[5];
                    size_t* out_index = 0;
                    for (size_t i = 0; i < 5; i++)
                    {
                        if (broadcast_axes.count(i) == 0)
                        {
                            out_index = &index[i];
                            break;
                        }
                    }
                    for (index[0] = 0; index[0] < out_shape[0]; ++index[0])
                    {
                        for (index[1] = 0; index[1] < out_shape[1]; ++index[1])
                        {
                            for (index[2] = 0; index[2] < out_shape[2]; ++index[2])
                            {
                                for (index[3] = 0; index[3] < out_shape[3]; ++index[3])
                                {
                                    for (index[4] = 0; index[4] < out_shape[4]; ++index[4])
                                    {
                                        out[index[0] * out_shape[1] * out_shape[2] * out_shape[3] *
                                                out_shape[4] +
                                            index[1] * out_shape[2] * out_shape[3] * out_shape[4] +
                                            index[2] * out_shape[3] * out_shape[4] +
                                            index[3] * out_shape[4] + index[4]] = in[*out_index];
                                    }
                                }
                            }
                        }
                    }
                }

                template <typename T>
                void broadcast_6d(const T* in,
                                  T* out,
                                  const Shape& in_shape,
                                  const Shape& out_shape,
                                  const AxisSet& broadcast_axes)
                {
                    size_t index[6];
                    size_t* out_index = 0;
                    for (size_t i = 0; i < 6; i++)
                    {
                        if (broadcast_axes.count(i) == 0)
                        {
                            out_index = &index[i];
                            break;
                        }
                    }
                    for (index[0] = 0; index[0] < out_shape[0]; ++index[0])
                    {
                        for (index[1] = 0; index[1] < out_shape[1]; ++index[1])
                        {
                            for (index[2] = 0; index[2] < out_shape[2]; ++index[2])
                            {
                                for (index[3] = 0; index[3] < out_shape[3]; ++index[3])
                                {
                                    for (index[4] = 0; index[4] < out_shape[4]; ++index[4])
                                    {
                                        for (index[5] = 0; index[5] < out_shape[5]; ++index[5])
                                        {
                                            out[index[0] * out_shape[1] * out_shape[2] *
                                                    out_shape[3] * out_shape[4] * out_shape[5] +
                                                index[1] * out_shape[2] * out_shape[3] *
                                                    out_shape[4] * out_shape[5] +
                                                index[2] * out_shape[3] * out_shape[4] *
                                                    out_shape[5] +
                                                index[3] * out_shape[4] * out_shape[5] +
                                                index[4] * out_shape[5] + index[5]] =
                                                in[*out_index];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                template <typename T>
                void broadcast(const T* in,
                               T* out,
                               const Shape& in_shape,
                               const Shape& out_shape,
                               const AxisSet& broadcast_axes)
                {
                    if (in_shape.size() == 0)
                    {
                        for (size_t i = 0; i < shape_size(out_shape); ++i)
                        {
                            out[i] = in[0];
                        }
                    }
                    else if (in_shape.size() == 1)
                    {
                        switch (out_shape.size())
                        {
                        case 2:
                            broadcast_2d<T>(in, out, in_shape, out_shape, broadcast_axes);
                            break;
                        case 3:
                            broadcast_3d<T>(in, out, in_shape, out_shape, broadcast_axes);
                            break;
                        case 4:
                            broadcast_4d<T>(in, out, in_shape, out_shape, broadcast_axes);
                            break;
                        case 5:
                            broadcast_5d<T>(in, out, in_shape, out_shape, broadcast_axes);
                            break;
                        case 6:
                            broadcast_6d<T>(in, out, in_shape, out_shape, broadcast_axes);
                            break;
                        default:
                            runtime::reference::broadcast<T>(
                                in, out, in_shape, out_shape, broadcast_axes);
                            break;
                        }
                    }
                    else
                    {
                        runtime::reference::broadcast<T>(
                            in, out, in_shape, out_shape, broadcast_axes);
                    }
                }
            }
        }
    }
}
