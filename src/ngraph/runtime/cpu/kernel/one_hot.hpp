/*******************************************************************************
* Copyright 2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#pragma once

#include "ngraph/runtime/cpu/cpu_eigen_utils.hpp"
#include "ngraph/runtime/reference/one_hot.hpp"
#include "ngraph/shape.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            namespace kernel
            {
                template <typename ElementType>
                void one_hot_rank_0(void* arg,
                                    void* out,
                                    const Shape& arg_shape,
                                    const Shape& out_shape,
                                    size_t one_hot_axis)

                {
                    size_t bounds = out_shape[one_hot_axis];

                    eigen::EigenVector<ElementType> arg_vector(
                        static_cast<ElementType*>(arg), eigen::fmt::V(shape_size(arg_shape)));
                    eigen::EigenVector<ElementType> out_vector(
                        static_cast<ElementType*>(out), eigen::fmt::V(shape_size(out_shape)));

                    out_vector.setZero();
                    auto pos_raw = arg_vector(0, 0);
                    if (floor(pos_raw) != pos_raw)
                    {
                        throw(std::range_error("One-hot: non-integral value in input"));
                    }
                    size_t pos = pos_raw;
                    if (pos >= bounds)
                    {
                        throw(std::range_error("One-hot: value is out of category range"));
                    }
                    out_vector(pos, 0) = 1;
                }

                template <typename ElementType>
                void one_hot_rank_1(void* arg,
                                    void* out,
                                    const Shape& arg_shape,
                                    const Shape& out_shape,
                                    const Strides& out_strides,
                                    size_t one_hot_axis)

                {
                    size_t bounds = out_shape[one_hot_axis];
                    eigen::EigenVector<ElementType> arg_vector(
                        static_cast<ElementType*>(arg), eigen::fmt::V(shape_size(arg_shape)));
                    eigen::EigenMatrix<ElementType> out_vector(
                        static_cast<ElementType*>(out), eigen::fmt::M(out_shape, out_strides));

                    out_vector.setZero();
                    for (size_t i = 0; i < arg_shape[0]; i++)
                    {
                        auto pos_raw = arg_vector(i, 0);
                        if (floor(pos_raw) != pos_raw)
                        {
                            throw(std::range_error("One-hot: non-integral value in input"));
                        }
                        size_t pos = pos_raw;
                        if (pos >= bounds)
                        {
                            throw(std::range_error("One-hot: value is out of category range"));
                        }
                        one_hot_axis == 0 ? out_vector(pos, i) = 1 : out_vector(i, pos) = 1;
                    }
                }

                template <typename ElementType>
                void one_hot_rank_2_or_more(void* arg,
                                            void* out,
                                            const Shape& arg_shape,
                                            const Shape& out_shape,
                                            size_t one_hot_axis)

                {
                    reference::one_hot<ElementType>(static_cast<const ElementType*>(arg),
                                                    static_cast<ElementType*>(out),
                                                    arg_shape,
                                                    out_shape,
                                                    one_hot_axis);
                }
            }
        }
    }
}
