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

#include "ngraph/types/element_type.hpp"

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
                     const ngraph::element::Int64::type M,
                     const ngraph::element::Int64::type N,
                     const ngraph::element::Int64::type K,
                     const ngraph::element::Float32::type alpha,
                     const ngraph::element::Float32::type* A,
                     const ngraph::element::Int64::type lda,
                     const ngraph::element::Float32::type* B,
                     const ngraph::element::Int64::type ldb,
                     const ngraph::element::Float32::type beta,
                     ngraph::element::Float32::type* C,
                     const ngraph::element::Int64::type ldc);
    }
}

namespace mkl
{
    extern "C" {
    void MKL_Somatcopy(char ordering,
                       char trans,
                       size_t rows,
                       size_t cols,
                       const ngraph::element::Float32::type alpha,
                       const ngraph::element::Float32::type* A,
                       size_t lda,
                       ngraph::element::Float32::type* B,
                       size_t ldb);
    }
}
