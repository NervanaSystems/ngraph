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
#include "contrib/mlir/callback_utils.hpp"
#include "cpu_runtime.hpp"
#include "ngraph/check.hpp"

#include "ngraph/runtime/cpu/cpu_kernels.hpp"
#include "ngraph/runtime/cpu/mkldnn_utils.hpp"

using namespace ngraph;
using namespace ngraph::runtime::ngmlir;

/// Callback for MaxPoolBackprop
static void __mlir_mkldnn_maxpoolbackprop(size_t rank,
                                          StaticMemRef* memRefSrc,
                                          StaticMemRef* memRefDelta,
                                          StaticMemRef* memRefOutput,
                                          void* attrs)
{
    mkldnn::memory::dims srcDims(rank);
    mkldnn::memory::dims srcStrides(rank);
    mkldnn::memory::dims deltaDims(rank);
    mkldnn::memory::dims deltaStrides(rank);
    for (auto i = 0; i < rank; i++)
    {
        srcDims[i] = memRefSrc->shapeAndStrides[i];
        srcStrides[i] = memRefSrc->shapeAndStrides[rank + i];
        deltaDims[i] = memRefDelta->shapeAndStrides[i];
        deltaStrides[i] = memRefDelta->shapeAndStrides[rank + i];
    }

    // build mkldnn primitive and execute
    mkldnn::memory::data_type dtype = mkldnn::memory::data_type::f32;
    auto diff_dst_desc = mkldnn::memory::desc(deltaDims, dtype, deltaStrides);
    auto diff_src_desc = mkldnn::memory::desc(srcDims, dtype, srcStrides);
    mkldnn::primitive_attr attr;
    mkldnn::engine cpu_engine(mkldnn::engine::kind::cpu, 0);
    mkldnn::pooling_forward::primitive_desc maxpool_pd_f;
    mkldnn::pooling_backward::primitive_desc maxpool_pd_b;
    if (rank == 4)
    {
        auto pAttrs = reinterpret_cast<poolAttrs<2>*>(attrs);
        auto maxpool_desc_f = mkldnn::pooling_forward::desc(
            mkldnn::prop_kind::forward_training,
            mkldnn::algorithm::pooling_max,
            diff_src_desc,
            diff_dst_desc,
            mkldnn::memory::dims{pAttrs->windowStrides[0], pAttrs->windowStrides[1]},
            mkldnn::memory::dims{pAttrs->windowShape[0], pAttrs->windowShape[1]},
            mkldnn::memory::dims{pAttrs->padBelow[0], pAttrs->padBelow[1]},
            mkldnn::memory::dims{pAttrs->padAbove[0], pAttrs->padAbove[1]});
        auto maxpool_desc_b = mkldnn::pooling_backward::desc(
            mkldnn::algorithm::pooling_max,
            diff_src_desc,
            diff_dst_desc,
            mkldnn::memory::dims{pAttrs->windowStrides[0], pAttrs->windowStrides[1]},
            mkldnn::memory::dims{pAttrs->windowShape[0], pAttrs->windowShape[1]},
            mkldnn::memory::dims{pAttrs->padBelow[0], pAttrs->padBelow[1]},
            mkldnn::memory::dims{pAttrs->padAbove[0], pAttrs->padAbove[1]});
        maxpool_pd_f = mkldnn::pooling_forward::primitive_desc(maxpool_desc_f, attr, cpu_engine);
        maxpool_pd_b = mkldnn::pooling_backward::primitive_desc(
            maxpool_desc_b, attr, cpu_engine, maxpool_pd_f);
    }
    else if (rank == 5)
    {
        auto pAttrs = reinterpret_cast<poolAttrs<3>*>(attrs);
        auto maxpool_desc_f = mkldnn::pooling_forward::desc(
            mkldnn::prop_kind::forward_training,
            mkldnn::algorithm::pooling_max,
            diff_src_desc,
            diff_dst_desc,
            mkldnn::memory::dims{
                pAttrs->windowStrides[0], pAttrs->windowStrides[1], pAttrs->windowStrides[2]},
            mkldnn::memory::dims{
                pAttrs->windowShape[0], pAttrs->windowShape[1], pAttrs->windowShape[2]},
            mkldnn::memory::dims{pAttrs->padBelow[0], pAttrs->padBelow[1], pAttrs->padBelow[2]},
            mkldnn::memory::dims{pAttrs->padAbove[0], pAttrs->padAbove[1], pAttrs->padAbove[2]});
        auto maxpool_desc_b = mkldnn::pooling_backward::desc(
            mkldnn::algorithm::pooling_max,
            diff_src_desc,
            diff_dst_desc,
            mkldnn::memory::dims{
                pAttrs->windowStrides[0], pAttrs->windowStrides[1], pAttrs->windowStrides[2]},
            mkldnn::memory::dims{
                pAttrs->windowShape[0], pAttrs->windowShape[1], pAttrs->windowShape[2]},
            mkldnn::memory::dims{pAttrs->padBelow[0], pAttrs->padBelow[1], pAttrs->padBelow[2]},
            mkldnn::memory::dims{pAttrs->padAbove[0], pAttrs->padAbove[1], pAttrs->padAbove[2]});
        auto maxpool_pd_f =
            mkldnn::pooling_forward::primitive_desc(maxpool_desc_f, attr, cpu_engine);
        maxpool_pd_f = mkldnn::pooling_forward::primitive_desc(maxpool_desc_f, attr, cpu_engine);
        maxpool_pd_b = mkldnn::pooling_backward::primitive_desc(
            maxpool_desc_b, attr, cpu_engine, maxpool_pd_f);
    }

    mkldnn::pooling_forward maxpool_f(maxpool_pd_f);
    mkldnn::memory src_mem{maxpool_pd_b.diff_src_desc(), cpu_engine, memRefSrc->allocatedPtr};
    mkldnn::memory dst_mem{maxpool_pd_b.diff_dst_desc(), cpu_engine};
    mkldnn::memory workspace{maxpool_pd_f.workspace_desc(), cpu_engine};

    mkldnn::pooling_backward maxpool_b(maxpool_pd_b);
    mkldnn::memory diff_dst{maxpool_pd_b.diff_dst_desc(), cpu_engine, memRefDelta->allocatedPtr};
    mkldnn::memory diff_src{maxpool_pd_b.diff_src_desc(), cpu_engine, memRefOutput->allocatedPtr};

    std::unordered_map<int, mkldnn::memory> exec_args_f = {
        {MKLDNN_ARG_SRC, src_mem}, {MKLDNN_ARG_WORKSPACE, workspace}, {MKLDNN_ARG_DST, dst_mem}};
    std::unordered_map<int, mkldnn::memory> exec_args_b = {{MKLDNN_ARG_DIFF_DST, diff_dst},
                                                           {MKLDNN_ARG_WORKSPACE, workspace},
                                                           {MKLDNN_ARG_DIFF_SRC, diff_src}};

    mkldnn::stream s(cpu_engine);
    try
    {
        maxpool_f.execute(s, exec_args_f);
        s.wait();
        maxpool_b.execute(s, exec_args_b);
        s.wait();
    }
    catch (const mkldnn::error& e)
    {
        throw ngraph_error("Could not run mkdnn primitive " + std::string(e.message));
    }
}

/// Callback for AvgPoolBackprop
static void __mlir_mkldnn_avgpoolbackprop(size_t rank,
                                          StaticMemRef* memRefInput,
                                          StaticMemRef* memRefOutput,
                                          void* attrs)
{
    mkldnn::memory::dims dims(rank);
    mkldnn::memory::dims strides(rank);
    mkldnn::memory::dims outDims(rank);
    for (auto i = 0; i < rank; i++)
    {
        dims[i] = memRefInput->shapeAndStrides[i];
        strides[i] = memRefInput->shapeAndStrides[rank + i];
        outDims[i] = memRefOutput->shapeAndStrides[i];
    }

    // build mkldnn primitive and execute
    mkldnn::memory::data_type dtype = mkldnn::memory::data_type::f32;
    auto diff_dst_desc = mkldnn::memory::desc(dims, dtype, strides);
    auto diff_src_desc = mkldnn::memory::desc(outDims, dtype, mkldnn::memory::format_tag::any);
    mkldnn::primitive_attr attr;
    mkldnn::engine cpu_engine(mkldnn::engine::kind::cpu, 0);
    mkldnn::pooling_backward::primitive_desc avgpool_pd_b;
    if (rank == 4)
    {
        auto pAttrs = reinterpret_cast<poolAttrs<2>*>(attrs);
        auto avgpool_desc_f = mkldnn::pooling_forward::desc(
            mkldnn::prop_kind::forward_training,
            (pAttrs->includePaddingInAvgComputation
                 ? mkldnn::algorithm::pooling_avg_include_padding
                 : mkldnn::algorithm::pooling_avg_exclude_padding),
            diff_src_desc,
            diff_dst_desc,
            mkldnn::memory::dims{pAttrs->windowStrides[0], pAttrs->windowStrides[1]},
            mkldnn::memory::dims{pAttrs->windowShape[0], pAttrs->windowShape[1]},
            mkldnn::memory::dims{pAttrs->padBelow[0], pAttrs->padBelow[1]},
            mkldnn::memory::dims{pAttrs->padAbove[0], pAttrs->padAbove[1]});
        auto avgpool_desc_b = mkldnn::pooling_backward::desc(
            (pAttrs->includePaddingInAvgComputation
                 ? mkldnn::algorithm::pooling_avg_include_padding
                 : mkldnn::algorithm::pooling_avg_exclude_padding),
            diff_src_desc,
            diff_dst_desc,
            mkldnn::memory::dims{pAttrs->windowStrides[0], pAttrs->windowStrides[1]},
            mkldnn::memory::dims{pAttrs->windowShape[0], pAttrs->windowShape[1]},
            mkldnn::memory::dims{pAttrs->padBelow[0], pAttrs->padBelow[1]},
            mkldnn::memory::dims{pAttrs->padAbove[0], pAttrs->padAbove[1]});
        auto avgpool_pd_f =
            mkldnn::pooling_forward::primitive_desc(avgpool_desc_f, attr, cpu_engine);
        avgpool_pd_b = mkldnn::pooling_backward::primitive_desc(
            avgpool_desc_b, attr, cpu_engine, avgpool_pd_f);
    }
    else if (rank == 5)
    {
        auto pAttrs = reinterpret_cast<poolAttrs<3>*>(attrs);
        auto avgpool_desc_f = mkldnn::pooling_forward::desc(
            mkldnn::prop_kind::forward_training,
            (pAttrs->includePaddingInAvgComputation
                 ? mkldnn::algorithm::pooling_avg_include_padding
                 : mkldnn::algorithm::pooling_avg_exclude_padding),
            diff_src_desc,
            diff_dst_desc,
            mkldnn::memory::dims{
                pAttrs->windowStrides[0], pAttrs->windowStrides[1], pAttrs->windowStrides[2]},
            mkldnn::memory::dims{
                pAttrs->windowShape[0], pAttrs->windowShape[1], pAttrs->windowShape[2]},
            mkldnn::memory::dims{pAttrs->padBelow[0], pAttrs->padBelow[1], pAttrs->padBelow[2]},
            mkldnn::memory::dims{pAttrs->padAbove[0], pAttrs->padAbove[1], pAttrs->padAbove[2]});
        auto avgpool_desc_b = mkldnn::pooling_backward::desc(
            (pAttrs->includePaddingInAvgComputation
                 ? mkldnn::algorithm::pooling_avg_include_padding
                 : mkldnn::algorithm::pooling_avg_exclude_padding),
            diff_src_desc,
            diff_dst_desc,
            mkldnn::memory::dims{
                pAttrs->windowStrides[0], pAttrs->windowStrides[1], pAttrs->windowStrides[2]},
            mkldnn::memory::dims{
                pAttrs->windowShape[0], pAttrs->windowShape[1], pAttrs->windowShape[2]},
            mkldnn::memory::dims{pAttrs->padBelow[0], pAttrs->padBelow[1], pAttrs->padBelow[2]},
            mkldnn::memory::dims{pAttrs->padAbove[0], pAttrs->padAbove[1], pAttrs->padAbove[2]});
        auto avgpool_pd_f =
            mkldnn::pooling_forward::primitive_desc(avgpool_desc_f, attr, cpu_engine);
        avgpool_pd_b = mkldnn::pooling_backward::primitive_desc(
            avgpool_desc_b, attr, cpu_engine, avgpool_pd_f);
    }

    mkldnn::pooling_backward avgpool(avgpool_pd_b);
    mkldnn::memory in{avgpool_pd_b.diff_dst_desc(), cpu_engine, memRefInput->allocatedPtr};
    mkldnn::memory out{avgpool_pd_b.diff_src_desc(), cpu_engine, memRefOutput->allocatedPtr};

    std::unordered_map<int, mkldnn::memory> exec_args = {{MKLDNN_ARG_DIFF_DST, in},
                                                         {MKLDNN_ARG_DIFF_SRC, out}};

    mkldnn::stream s(cpu_engine);
    try
    {
        avgpool.execute(s, exec_args);
        s.wait();
    }
    catch (const mkldnn::error& e)
    {
        throw ngraph_error("Could not run mkdnn primitive " + std::string(e.message));
    }
}

/// Callback for AvgPool and MaxPool
static void __mlir_mkldnn_pooling(
    size_t rank, StaticMemRef* memRefInput, StaticMemRef* memRefOutput, void* attrs, OpType type)
{
    mkldnn::memory::dims dims(rank);
    mkldnn::memory::dims strides(rank);
    mkldnn::memory::dims outDims(rank);
    for (auto i = 0; i < rank; i++)
    {
        dims[i] = memRefInput->shapeAndStrides[i];
        strides[i] = memRefInput->shapeAndStrides[rank + i];
        outDims[i] = memRefOutput->shapeAndStrides[i];
    }

    // build mkldnn primitive and execute
    mkldnn::memory::data_type dtype = mkldnn::memory::data_type::f32;
    auto input_desc = mkldnn::memory::desc(dims, dtype, strides);
    auto result_desc = mkldnn::memory::desc(outDims, dtype, mkldnn::memory::format_tag::any);
    mkldnn::primitive_attr attr;
    mkldnn::engine cpu_engine(mkldnn::engine::kind::cpu, 0);
    mkldnn::pooling_forward::primitive_desc pool_pd;
    if (rank == 4)
    {
        auto pAttrs = reinterpret_cast<poolAttrs<2>*>(attrs);
        mkldnn::algorithm alg = type == OpType::MAXPOOL
                                    ? mkldnn::algorithm::pooling_max
                                    : (pAttrs->includePaddingInAvgComputation
                                           ? mkldnn::algorithm::pooling_avg_include_padding
                                           : mkldnn::algorithm::pooling_avg_exclude_padding);
        auto pool_desc = mkldnn::pooling_forward::desc(
            mkldnn::prop_kind::forward_inference,
            alg,
            input_desc,
            result_desc,
            mkldnn::memory::dims{pAttrs->windowStrides[0], pAttrs->windowStrides[1]},
            mkldnn::memory::dims{pAttrs->windowShape[0], pAttrs->windowShape[1]},
            mkldnn::memory::dims{pAttrs->padBelow[0], pAttrs->padBelow[1]},
            mkldnn::memory::dims{pAttrs->padAbove[0], pAttrs->padAbove[1]});
        pool_pd = mkldnn::pooling_forward::primitive_desc(pool_desc, attr, cpu_engine);
    }
    else if (rank == 5)
    {
        auto pAttrs = reinterpret_cast<poolAttrs<3>*>(attrs);
        mkldnn::algorithm alg = type == OpType::MAXPOOL
                                    ? mkldnn::algorithm::pooling_max
                                    : (pAttrs->includePaddingInAvgComputation
                                           ? mkldnn::algorithm::pooling_avg_include_padding
                                           : mkldnn::algorithm::pooling_avg_exclude_padding);
        auto pool_desc = mkldnn::pooling_forward::desc(
            mkldnn::prop_kind::forward_inference,
            alg,
            input_desc,
            result_desc,
            mkldnn::memory::dims{
                pAttrs->windowStrides[0], pAttrs->windowStrides[1], pAttrs->windowStrides[2]},
            mkldnn::memory::dims{
                pAttrs->windowShape[0], pAttrs->windowShape[1], pAttrs->windowShape[2]},
            mkldnn::memory::dims{pAttrs->padBelow[0], pAttrs->padBelow[1], pAttrs->padBelow[2]},
            mkldnn::memory::dims{pAttrs->padAbove[0], pAttrs->padAbove[1], pAttrs->padAbove[2]});
        pool_pd = mkldnn::pooling_forward::primitive_desc(pool_desc, attr, cpu_engine);
    }

    mkldnn::pooling_forward pool(pool_pd);
    mkldnn::memory in{pool_pd.src_desc(), cpu_engine, memRefInput->allocatedPtr};
    mkldnn::memory out{pool_pd.dst_desc(), cpu_engine, memRefOutput->allocatedPtr};

    std::unordered_map<int, mkldnn::memory> exec_args = {{MKLDNN_ARG_SRC, in},
                                                         {MKLDNN_ARG_DST, out}};

    mkldnn::stream s(cpu_engine);
    try
    {
        pool.execute(s, exec_args);
        s.wait();
    }
    catch (const mkldnn::error& e)
    {
        throw ngraph_error("Could not run mkdnn primitive " + std::string(e.message));
    }
}

/// Callback for Softmax
static void __mlir_mkldnn_softmax(size_t rank,
                                  StaticMemRef* memRefInput,
                                  StaticMemRef* memRefOutput,
                                  void* attrs)
{
    mkldnn::memory::dims dims(rank);
    mkldnn::memory::dims strides(rank);
    for (auto i = 0; i < rank; i++)
    {
        dims[i] = memRefInput->shapeAndStrides[i];
        strides[i] = memRefInput->shapeAndStrides[rank + i];
    }
    auto softmax_axis = *reinterpret_cast<int64_t*>(attrs);

    // build mkldnn primitive and execute
    mkldnn::memory::data_type dtype = mkldnn::memory::data_type::f32;
    auto input_desc = mkldnn::memory::desc(dims, dtype, strides);
    auto softmax_desc =
        mkldnn::softmax_forward::desc(mkldnn::prop_kind::forward_scoring, input_desc, softmax_axis);
    mkldnn::primitive_attr attr;
    mkldnn::engine cpu_engine(mkldnn::engine::kind::cpu, 0);
    auto softmax_pd = mkldnn::softmax_forward::primitive_desc(softmax_desc, attr, cpu_engine);
    mkldnn::softmax_forward softmax(softmax_pd);

    mkldnn::memory in{softmax_pd.src_desc(), cpu_engine, memRefInput->allocatedPtr};
    mkldnn::memory out{softmax_pd.dst_desc(), cpu_engine, memRefOutput->allocatedPtr};

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

/// Callback for MatMul
static void __mlir_cblas_sgemm(StaticMemRef* memRefmatA,
                               StaticMemRef* memRefmatB,
                               StaticMemRef* memRefmatC,
                               void* attrs)
{
    auto gAttrs = reinterpret_cast<gemmAttrs*>(attrs);
    cblas::cblas_sgemm(cblas::Layout::RowMajor,
                       gAttrs->transposeA ? cblas::Transpose::Transpose : cblas::Transpose::None,
                       gAttrs->transposeB ? cblas::Transpose::Transpose : cblas::Transpose::None,
                       gAttrs->m,
                       gAttrs->n,
                       gAttrs->k,
                       1.0f,
                       reinterpret_cast<float*>(memRefmatA->allocatedPtr),
                       std::max<size_t>(1, gAttrs->lda),
                       reinterpret_cast<float*>(memRefmatB->allocatedPtr),
                       std::max<size_t>(1, gAttrs->ldb),
                       0.0f,
                       reinterpret_cast<float*>(memRefmatC->allocatedPtr),
                       std::max<size_t>(1, gAttrs->ldc));
}

/// Callback for Gemm
static void __mlir_cblas_sgemm_with_bias(StaticMemRef* memRefmatA,
                                         StaticMemRef* memRefmatB,
                                         StaticMemRef* memRefmatC,
                                         StaticMemRef* memRefmatOut,
                                         void* attrs)
{
    auto gAttrs = reinterpret_cast<gemmAttrs*>(attrs);
    auto transposeA = gAttrs->transposeA;
    auto transposeB = gAttrs->transposeB;
    auto m = gAttrs->m;
    auto n = gAttrs->n;
    auto k = gAttrs->k;
    auto lda = gAttrs->lda;
    auto ldb = gAttrs->ldb;
    auto ldc = gAttrs->ldc;
    auto alpha = gAttrs->alpha;
    auto beta = gAttrs->beta;
    auto broadcastHint = gAttrs->broadcastHint;

    auto matA = reinterpret_cast<float*>(memRefmatA->allocatedPtr);
    auto matB = reinterpret_cast<float*>(memRefmatB->allocatedPtr);
    auto matC = reinterpret_cast<float*>(memRefmatC->allocatedPtr);
    auto matOut = reinterpret_cast<float*>(memRefmatOut->allocatedPtr);

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

extern "C" void __mlir_callback_1_input(void* input, void* output, void* attrs, OpType type)
{
    auto unrankedMemRefInput = reinterpret_cast<UnrankedMemRef*>(input);
    auto unrankedMemRefOutput = reinterpret_cast<UnrankedMemRef*>(output);

    if (type == OpType::SOFTMAX)
    {
        __mlir_mkldnn_softmax(unrankedMemRefInput->rank,
                              unrankedMemRefInput->memRefDescPtr,
                              unrankedMemRefOutput->memRefDescPtr,
                              attrs);
    }
    else if (type == OpType::AVGPOOL || type == OpType::MAXPOOL)
    {
        __mlir_mkldnn_pooling(unrankedMemRefInput->rank,
                              unrankedMemRefInput->memRefDescPtr,
                              unrankedMemRefOutput->memRefDescPtr,
                              attrs,
                              type);
    }
    else if (type == OpType::AVGPOOLBACKPROP)
    {
        __mlir_mkldnn_avgpoolbackprop(unrankedMemRefInput->rank,
                                      unrankedMemRefInput->memRefDescPtr,
                                      unrankedMemRefOutput->memRefDescPtr,
                                      attrs);
    }
}

extern "C" void
    __mlir_callback_2_inputs(void* input0, void* input1, void* output, void* attrs, OpType type)
{
    auto unrankedMemRefInput0 = reinterpret_cast<UnrankedMemRef*>(input0);
    auto unrankedMemRefInput1 = reinterpret_cast<UnrankedMemRef*>(input1);
    auto unrankedMemRefOutput = reinterpret_cast<UnrankedMemRef*>(output);

    if (type == OpType::MAXPOOLBACKPROP)
    {
        __mlir_mkldnn_maxpoolbackprop(unrankedMemRefInput0->rank,
                                      unrankedMemRefInput0->memRefDescPtr,
                                      unrankedMemRefInput1->memRefDescPtr,
                                      unrankedMemRefOutput->memRefDescPtr,
                                      attrs);
    }
    else if (type == OpType::MATMUL)
    {
        __mlir_cblas_sgemm(unrankedMemRefInput0->memRefDescPtr,
                           unrankedMemRefInput1->memRefDescPtr,
                           unrankedMemRefOutput->memRefDescPtr,
                           attrs);
    }
}

extern "C" void __mlir_callback_3_inputs(
    void* input0, void* input1, void* input2, void* output, void* attrs, OpType type)
{
    auto unrankedMemRefInput0 = reinterpret_cast<UnrankedMemRef*>(input0);
    auto unrankedMemRefInput1 = reinterpret_cast<UnrankedMemRef*>(input1);
    auto unrankedMemRefInput2 = reinterpret_cast<UnrankedMemRef*>(input2);
    auto unrankedMemRefOutput = reinterpret_cast<UnrankedMemRef*>(output);

    if (type == OpType::GEMM)
    {
        __mlir_cblas_sgemm_with_bias(unrankedMemRefInput0->memRefDescPtr,
                                     unrankedMemRefInput1->memRefDescPtr,
                                     unrankedMemRefInput2->memRefDescPtr,
                                     unrankedMemRefOutput->memRefDescPtr,
                                     attrs);
    }
}
