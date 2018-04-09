/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
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

#include <cstddef>
#include <cstdint>

// CBLAS types and wrappers

namespace cblas
{
    enum class Layout
    {
        RowMajor = 101,
        ColMajor = 102
    };

    enum class Transpose
    {
        None = 111,
        Transpose = 112,
        ConjTrans = 113
    };

    enum class UpperLower
    {
        Upper = 121,
        Lower = 122
    };

    enum class Diag
    {
        NonUnit = 131,
        Unit = 132
    };

    enum class Side
    {
        Left = 141,
        Right = 142
    };

    enum class Storage
    {
        Packed = 151
    };

    enum class Ident
    {
        AMatrix = 161,
        BMatrix = 162
    };

    enum class Offset
    {
        RowOffset = 171,
        ColOffset = 172,
        FixOffset = 173
    };

    extern "C" {
    void cblas_sgemm(const Layout layout,
                     const Transpose TransA,
                     const Transpose TransB,
                     const int64_t M,
                     const int64_t N,
                     const int64_t K,
                     const float alpha,
                     const float* A,
                     const int64_t lda,
                     const float* B,
                     const int64_t ldb,
                     const float beta,
                     float* C,
                     const int64_t ldc);
    }
}

namespace mkl
{
    extern "C" {
    void MKL_Somatcopy(char ordering,
                       char trans,
                       size_t rows,
                       size_t cols,
                       const float alpha,
                       const float* A,
                       size_t lda,
                       float* B,
                       size_t ldb);
    }
}

namespace ngraph
{
    class Shape;
    class AxisSet;
    class AxisVector;

    namespace runtime
    {
        namespace cpu
        {
            namespace kernel
            {
                void pad_4d_float32(float* input,
                                    float* output,
                                    float pad_value,
                                    const Shape& input_shape,
                                    const Shape& output_shape,
                                    const Shape& padding_below,
                                    const Shape& padding_above);

                void reduce_sum_all_1d_float32(float* input,
                                               float* output,
                                               const Shape& input_shape,
                                               const Shape& output_shape);

                void reduce_sum_all_2d_float32(float* input,
                                               float* output,
                                               const Shape& input_shape,
                                               const Shape& output_shape);

                void reduce_sum_2d_1rd_float32(float* input,
                                               float* output,
                                               const Shape& input_shape,
                                               const Shape& output_shape,
                                               const AxisSet& reduction_axes);

                void reduce_sum_all_4d_float32(float* input,
                                               float* output,
                                               const Shape& input_shape,
                                               const Shape& output_shape);

                void reduce_max_2d_1rd_float32(float* input,
                                               float* output,
                                               const Shape& input_shape,
                                               const Shape& output_shape,
                                               const AxisSet& reduction_axes);

                void reshape_3d_3d_float32(float* input,
                                           float* output,
                                           const Shape& input_shape,
                                           const AxisVector& input_axis_order,
                                           const Shape& output_shape);

                void reshape_4d_4d_float32(float* input,
                                           float* output,
                                           const Shape& input_shape,
                                           const AxisVector& input_axis_order,
                                           const Shape& output_shape);
            }
        }
    }
}
