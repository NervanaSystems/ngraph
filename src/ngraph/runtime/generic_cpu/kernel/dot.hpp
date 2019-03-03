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

#include <Eigen/Dense>
#include <cmath>
#include <omp.h>
#include <utility>

#include "ngraph/coordinate_transform.hpp"
#include "ngraph/runtime/reference/dot.hpp"
#include "ngraph/shape_util.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace gcpu
        {
            namespace kernel
            {
                template <typename T>
                void dot(const T* arg0,
                         const T* arg1,
                         T* out,
                         const Shape& arg0_shape,
                         const Shape& arg1_shape,
                         const Shape& out_shape,
                         size_t reduction_axes_count)
                {
                    if (arg0_shape.size() == 2 && arg1_shape.size() == 2 && out_shape.size() == 2)
                    {
                        Eigen::Map<
                            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
                            a0(const_cast<T*>(arg0), arg0_shape[0], arg0_shape[1]);
                        Eigen::Map<
                            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
                            a1(const_cast<T*>(arg1), arg1_shape[0], arg1_shape[1]);
                        Eigen::Map<
                            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
                            o(const_cast<T*>(out), out_shape[0], out_shape[1]);
                        o = a0 * a1;
                    }
                    else
                    {
                        reference::dot(arg0,
                                       arg1,
                                       out,
                                       arg0_shape,
                                       arg1_shape,
                                       out_shape,
                                       reduction_axes_count);
                    }
                }
            }
        }
    }
}
