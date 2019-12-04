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

// NOTE: This file follows nGraph format style.
// Follows nGraph naming convention for public APIs only, else MLIR naming convention.

#include "contrib/mlir/backend/cpu/cpu_backend.hpp"
#include "cpu_runtime.hpp"
#include "ngraph/check.hpp"

#include "ngraph/runtime/cpu/cpu_kernels.hpp"
#include "ngraph/runtime/cpu/mkldnn_utils.hpp"

using namespace ngraph;
using namespace ngraph::runtime::ngmlir;

/// Callback for Softmax
static void __mlir_mkldnn_softmax(size_t rank, void* input, void* output, const size_t softmax_axis)
{
    auto memRefInput = reinterpret_cast<StaticMemRef*>(input);
    auto memRefOutput = reinterpret_cast<StaticMemRef*>(output);
    mkldnn::memory::dims dims(rank);
    mkldnn::memory::dims strides(rank);
    for (auto i = 0; i < rank; i++)
    {
        dims[i] = memRefInput->shapeAndStrides[i];
        strides[i] = memRefInput->shapeAndStrides[rank + i];
    }

    // build mkldnn primitive and execute
    mkldnn::memory::data_type dtype = mkldnn::memory::data_type::f32;
    auto input_desc = mkldnn::memory::desc(dims, dtype, strides);
    auto softmax_desc =
        mkldnn::softmax_forward::desc(mkldnn::prop_kind::forward_scoring, input_desc, softmax_axis);
    mkldnn::primitive_attr attr;
    mkldnn::engine cpu_engine(mkldnn::engine::kind::cpu, 0);
    auto softmax_pd = mkldnn::softmax_forward::primitive_desc(softmax_desc, attr, cpu_engine);
    mkldnn::softmax_forward softmax(softmax_pd);

    mkldnn::memory in{
        softmax_pd.src_desc(), cpu_engine, memRefInput->basePtr + memRefInput->offset};
    mkldnn::memory out{
        softmax_pd.dst_desc(), cpu_engine, memRefOutput->basePtr + memRefOutput->offset};

    std::unordered_map<int, mkldnn::memory> exec_args = {{MKLDNN_ARG_SRC, in},
                                                         {MKLDNN_ARG_DST, out}};

    mkldnn::stream s(cpu_engine);
    try
    {
        softmax.execute(s, exec_args);
        s.wait();
    }
    catch (const mkldnn::error& e)
    {
        throw ngraph_error("Could not run mkdnn primitive " + std::string(e.message));
    }
}

extern "C" void __mlir_mkldnn_softmax_2d(void* input, void* output, const size_t softmax_axis)
{
    __mlir_mkldnn_softmax(2, input, output, softmax_axis);
}

extern "C" void __mlir_mkldnn_softmax_4d(void* input, void* output, const size_t softmax_axis)
{
    __mlir_mkldnn_softmax(4, input, output, softmax_axis);
}

/// Callback for MatMul
extern "C" void __mlir_cblas_sgemm(
    void* matAPtr, void* matBPtr, void* matCPtr, const bool transposeA, const bool transposeB)
{
    auto memRefmatA = reinterpret_cast<StaticMemRef*>(matAPtr);
    auto memRefmatB = reinterpret_cast<StaticMemRef*>(matBPtr);
    auto memRefmatC = reinterpret_cast<StaticMemRef*>(matCPtr);

    auto m = memRefmatA->shapeAndStrides[0];
    auto k = memRefmatA->shapeAndStrides[1];
    auto n = memRefmatB->shapeAndStrides[1];
    auto lda = memRefmatA->shapeAndStrides[1];
    auto ldb = memRefmatB->shapeAndStrides[1];

    if (transposeA)
    {
        m = memRefmatA->shapeAndStrides[1];
        k = memRefmatA->shapeAndStrides[0];
    }
    if (transposeB)
    {
        n = memRefmatB->shapeAndStrides[0];
    }

    auto ldc = n;

    cblas::cblas_sgemm(cblas::Layout::RowMajor,
                       transposeA ? cblas::Transpose::Transpose : cblas::Transpose::None,
                       transposeB ? cblas::Transpose::Transpose : cblas::Transpose::None,
                       m,
                       n,
                       k,
                       1.0f,
                       memRefmatA->basePtr + memRefmatA->offset,
                       std::max<size_t>(1, lda),
                       memRefmatB->basePtr + memRefmatB->offset,
                       std::max<size_t>(1, ldb),
                       0.0f,
                       memRefmatC->basePtr + memRefmatC->offset,
                       std::max<size_t>(1, ldc));
}

/// Callback for Gemm
extern "C" void __mlir_cblas_sgemm_with_bias(void* matAPtr,
                                             void* matBPtr,
                                             void* matCPtr,
                                             void* matOutPtr,
                                             const bool transposeA,
                                             const bool transposeB,
                                             const size_t m,
                                             const size_t n,
                                             const size_t k,
                                             const size_t lda,
                                             const size_t ldb,
                                             const size_t ldc,
                                             const float alpha,
                                             const float beta,
                                             const int broadcastHint)
{
    auto* matA = *(reinterpret_cast<float**>(matAPtr));
    auto* matB = *(reinterpret_cast<float**>(matBPtr));
    auto* matC = *(reinterpret_cast<float**>(matCPtr));
    auto* matOut = *(reinterpret_cast<float**>(matOutPtr));

    cblas::cblas_sgemm(cblas::Layout::RowMajor,
                       transposeA ? cblas::Transpose::Transpose : cblas::Transpose::None,
                       transposeB ? cblas::Transpose::Transpose : cblas::Transpose::None,
                       m,
                       n,
                       k,
                       alpha,
                       matA,
                       std::max<size_t>(1, lda),
                       matB,
                       std::max<size_t>(1, ldb),
                       0.0f,
                       matOut,
                       std::max<size_t>(1, ldc));

    if (broadcastHint == 0)
    {
        std::vector<float> ones(m, 1.0f);
        cblas::cblas_sgemm(cblas::Layout::RowMajor,
                           cblas::Transpose::None,
                           cblas::Transpose::None,
                           m,
                           n,
                           1,
                           beta,
                           ones.data(),
                           1,
                           matC,
                           std::max<size_t>(1, n),
                           1.0f,
                           matOut,
                           std::max<size_t>(1, ldc));
    }
    else if (broadcastHint == 1)
    {
        std::vector<float> ones(n, 1.0f);
        cblas::cblas_sgemm(cblas::Layout::RowMajor,
                           cblas::Transpose::None,
                           cblas::Transpose::None,
                           m,
                           n,
                           1,
                           beta,
                           matC,
                           1,
                           ones.data(),
                           std::max<size_t>(1, n),
                           1.0f,
                           matOut,
                           std::max<size_t>(1, ldc));
    }
    else if (broadcastHint == 2)
    {
        std::vector<float> ones(m, 1.0f);
        std::vector<float> bias(n, *matC);
        cblas::cblas_sgemm(cblas::Layout::RowMajor,
                           cblas::Transpose::None,
                           cblas::Transpose::None,
                           m,
                           n,
                           1,
                           beta,
                           ones.data(),
                           1,
                           bias.data(),
                           std::max<size_t>(1, n),
                           1.0f,
                           matOut,
                           std::max<size_t>(1, ldc));
    }
    else
    {
        std::vector<float> identity(n * n, 0.0f);
        for (auto i = 0; i < n * n; i += n + 1)
        {
            identity[i] = 1.0;
        }
        cblas::cblas_sgemm(cblas::Layout::RowMajor,
                           cblas::Transpose::None,
                           cblas::Transpose::None,
                           m,
                           n,
                           n,
                           beta,
                           matC,
                           std::max<size_t>(1, n),
                           identity.data(),
                           std::max<size_t>(1, n),
                           1.0f,
                           matOut,
                           std::max<size_t>(1, ldc));
    }
}

extern "C" void __mlir_cblas_sgemm_scalar_bias(void* matAPtr,
                                               void* matBPtr,
                                               void* matCPtr,
                                               void* matOutPtr,
                                               const bool transposeA,
                                               const bool transposeB,
                                               const size_t m,
                                               const size_t n,
                                               const size_t k,
                                               const size_t lda,
                                               const size_t ldb,
                                               const size_t ldc,
                                               const float alpha,
                                               const float beta,
                                               const int broadcastHint)
{
    __mlir_cblas_sgemm_with_bias(matAPtr,
                                 matBPtr,
                                 matCPtr,
                                 matOutPtr,
                                 transposeA,
                                 transposeB,
                                 m,
                                 n,
                                 k,
                                 lda,
                                 ldb,
                                 ldc,
                                 alpha,
                                 beta,
                                 broadcastHint);
}

extern "C" void __mlir_cblas_sgemm_1d_bias(void* matAPtr,
                                           void* matBPtr,
                                           void* matCPtr,
                                           void* matOutPtr,
                                           const bool transposeA,
                                           const bool transposeB,
                                           const size_t m,
                                           const size_t n,
                                           const size_t k,
                                           const size_t lda,
                                           const size_t ldb,
                                           const size_t ldc,
                                           const float alpha,
                                           const float beta,
                                           const int broadcastHint)
{
    __mlir_cblas_sgemm_with_bias(matAPtr,
                                 matBPtr,
                                 matCPtr,
                                 matOutPtr,
                                 transposeA,
                                 transposeB,
                                 m,
                                 n,
                                 k,
                                 lda,
                                 ldb,
                                 ldc,
                                 alpha,
                                 beta,
                                 broadcastHint);
}

extern "C" void __mlir_cblas_sgemm_2d_bias(void* matAPtr,
                                           void* matBPtr,
                                           void* matCPtr,
                                           void* matOutPtr,
                                           const bool transposeA,
                                           const bool transposeB,
                                           const size_t m,
                                           const size_t n,
                                           const size_t k,
                                           const size_t lda,
                                           const size_t ldb,
                                           const size_t ldc,
                                           const float alpha,
                                           const float beta,
                                           const int broadcastHint)
{
    __mlir_cblas_sgemm_with_bias(matAPtr,
                                 matBPtr,
                                 matCPtr,
                                 matOutPtr,
                                 transposeA,
                                 transposeB,
                                 m,
                                 n,
                                 k,
                                 lda,
                                 ldb,
                                 ldc,
                                 alpha,
                                 beta,
                                 broadcastHint);
}
