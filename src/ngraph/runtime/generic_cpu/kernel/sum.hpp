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

#include <unsupported/Eigen/CXX11/Tensor>

#include "ngraph/axis_vector.hpp"
#include "ngraph/shape.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace gcpu
        {
            namespace kernel
            {
                template <typename ElementType, unsigned int Rank, unsigned int ReductionDims>
                void reduce_sum(void* input,
                                void* output,
                                const Shape& input_shape,
                                const Shape& output_shape,
                                const AxisSet& reduction_axes)
                {
                    Eigen::array<Eigen::Index, Rank> in_dims;
                    Eigen::array<Eigen::Index, Rank - ReductionDims> out_dims;
                    Eigen::array<Eigen::Index, ReductionDims> reduction_dims;

                    for (int i = 0; i < Rank; i++)
                    {
                        in_dims[i] = input_shape[i];
                    }

                    for (int i = 0; i < Rank - ReductionDims; i++)
                    {
                        out_dims[i] = output_shape[i];
                    }

                    int i = 0;
                    for (auto axis : reduction_axes)
                    {
                        reduction_dims[i++] = axis;
                    }

                    Eigen::TensorMap<
                        Eigen::Tensor<ElementType, Rank - ReductionDims, Eigen::RowMajor>>
                        out(static_cast<ElementType*>(output), out_dims);
                    Eigen::TensorMap<Eigen::Tensor<ElementType, Rank, Eigen::RowMajor>> in(
                        static_cast<ElementType*>(input), in_dims);
                    out = in.sum(reduction_dims);
                }

                template <typename T>
                void sum(const T* arg,
                         T* out,
                         const Shape& in_shape,
                         const Shape& out_shape,
                         const AxisSet& reduction_axes)
                {
                    NGRAPH_INFO << in_shape;
                    NGRAPH_INFO << out_shape;
                    NGRAPH_INFO << reduction_axes;
                    T* in = const_cast<T*>(arg);
                    switch (in_shape.size())
                    {
                    case 0:
                        switch (reduction_axes.size())
                        {
                        case 0:
                            reduce_sum<T, 0, 0>(in, out, in_shape, out_shape, reduction_axes);
                            break;
                        }
                        break;
                    case 1:
                        switch (reduction_axes.size())
                        {
                        case 0:
                            reduce_sum<T, 1, 0>(in, out, in_shape, out_shape, reduction_axes);
                            break;
                        case 1:
                            reduce_sum<T, 1, 1>(in, out, in_shape, out_shape, reduction_axes);
                            break;
                        }
                        break;
                    case 2:
                        switch (reduction_axes.size())
                        {
                        case 0:
                            reduce_sum<T, 2, 0>(in, out, in_shape, out_shape, reduction_axes);
                            break;
                        case 1:
                            reduce_sum<T, 2, 1>(in, out, in_shape, out_shape, reduction_axes);
                            break;
                        case 2:
                            reduce_sum<T, 2, 2>(in, out, in_shape, out_shape, reduction_axes);
                            break;
                        }
                        break;
                    case 3:
                        switch (reduction_axes.size())
                        {
                        case 0:
                            reduce_sum<T, 3, 0>(in, out, in_shape, out_shape, reduction_axes);
                            break;
                        case 1:
                            reduce_sum<T, 3, 1>(in, out, in_shape, out_shape, reduction_axes);
                            break;
                        case 2:
                            reduce_sum<T, 3, 2>(in, out, in_shape, out_shape, reduction_axes);
                            break;
                        case 3:
                            reduce_sum<T, 3, 3>(in, out, in_shape, out_shape, reduction_axes);
                            break;
                        }
                        break;
                    case 4:
                        switch (reduction_axes.size())
                        {
                        case 0:
                            reduce_sum<T, 4, 0>(in, out, in_shape, out_shape, reduction_axes);
                            break;
                        case 1:
                            reduce_sum<T, 4, 1>(in, out, in_shape, out_shape, reduction_axes);
                            break;
                        case 2:
                            reduce_sum<T, 4, 2>(in, out, in_shape, out_shape, reduction_axes);
                            break;
                        case 3:
                            reduce_sum<T, 4, 3>(in, out, in_shape, out_shape, reduction_axes);
                            break;
                        case 4:
                            reduce_sum<T, 4, 4>(in, out, in_shape, out_shape, reduction_axes);
                            break;
                        }
                        break;
                    case 5:
                        switch (reduction_axes.size())
                        {
                        case 0:
                            reduce_sum<T, 5, 0>(in, out, in_shape, out_shape, reduction_axes);
                            break;
                        case 1:
                            reduce_sum<T, 5, 1>(in, out, in_shape, out_shape, reduction_axes);
                            break;
                        case 2:
                            reduce_sum<T, 5, 2>(in, out, in_shape, out_shape, reduction_axes);
                            break;
                        case 3:
                            reduce_sum<T, 5, 3>(in, out, in_shape, out_shape, reduction_axes);
                            break;
                        case 4:
                            reduce_sum<T, 5, 4>(in, out, in_shape, out_shape, reduction_axes);
                            break;
                        case 5:
                            reduce_sum<T, 5, 5>(in, out, in_shape, out_shape, reduction_axes);
                            break;
                        }
                        break;
                    case 6:
                        switch (reduction_axes.size())
                        {
                        case 0:
                            reduce_sum<T, 6, 0>(in, out, in_shape, out_shape, reduction_axes);
                            break;
                        case 1:
                            reduce_sum<T, 6, 1>(in, out, in_shape, out_shape, reduction_axes);
                            break;
                        case 2:
                            reduce_sum<T, 6, 2>(in, out, in_shape, out_shape, reduction_axes);
                            break;
                        case 3:
                            reduce_sum<T, 6, 3>(in, out, in_shape, out_shape, reduction_axes);
                            break;
                        case 4:
                            reduce_sum<T, 6, 4>(in, out, in_shape, out_shape, reduction_axes);
                            break;
                        case 5:
                            reduce_sum<T, 6, 5>(in, out, in_shape, out_shape, reduction_axes);
                            break;
                        case 6:
                            reduce_sum<T, 6, 6>(in, out, in_shape, out_shape, reduction_axes);
                            break;
                        }
                        break;
                    }
                }
            }
        }
    }
}
