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

// NOTE: This file follows nGraph format style.
// Follows nGraph naming convention for public APIs only, else MLIR naming convention.

#include "callback_utils.hpp"
#include "contrib/mlir/backend/cpu/cpu_backend.hpp"
#include "cpu_runtime.hpp"
#include "ngraph/check.hpp"

#include "ngraph/runtime/cpu/cpu_kernels.hpp"
#include "ngraph/runtime/cpu/mkldnn_utils.hpp"

using namespace ngraph;
using namespace ngraph::runtime::ngmlir;

static bool inline compareMkldnnDims(mkldnn_dims_t& arr1, mkldnn_dims_t& arr2, size_t size)
{
    for (auto i = 0; i < size; i++)
    {
        if (arr1[i] != arr2[i])
        {
            return false;
        }
    }
    return true;
}

static bool compareMkldnnStridesOrder(mkldnn_dims_t& strides1, mkldnn_dims_t& strides2, size_t size)
{
    std::vector<size_t> indices1(size, 0), indices2(size, 0);
    for (size_t i = 0; i < size; i++)
    {
        indices1[i] = i;
        indices2[i] = i;
    }
    std::sort(indices1.begin(), indices1.begin(), [&](const size_t& n1, const size_t& n2) {
        return strides1[n1] < strides1[n2];
    });
    std::sort(indices2.begin(), indices2.begin(), [&](const size_t& n1, const size_t& n2) {
        return strides2[n1] < strides2[n2];
    });

    for (auto i = 0; i < size; i++)
    {
        if (indices1[i] != indices2[i])
        {
            return false;
        }
    }
    return true;
}

static bool compareMkldnnMdFormats(const mkldnn::memory::desc& lhs, const mkldnn::memory::desc& rhs)
{
    mkldnn_memory_desc_t md1 = lhs.data, md2 = rhs.data;

    if (md1.format_kind != md2.format_kind)
    {
        return false;
    }

    if (md1.format_kind != static_cast<mkldnn_format_kind_t>(mkldnn::memory::format_kind::blocked))
    {
        // mkldnn not implemented yet
        return false;
    }

    if (md1.ndims != md2.ndims)
    {
        return false;
    }

    auto blk1 = md1.format_desc.blocking;
    auto blk2 = md2.format_desc.blocking;

    if (blk1.inner_nblks != blk2.inner_nblks ||
        !compareMkldnnDims(blk1.inner_blks, blk2.inner_blks, blk1.inner_nblks) ||
        !compareMkldnnDims(blk1.inner_idxs, blk2.inner_idxs, blk1.inner_nblks))
    {
        return false;
    }

    return compareMkldnnStridesOrder(blk1.strides, blk2.strides, md1.ndims);
}

static mkldnn::memory convertLayoutIfDiff(const mkldnn::memory::desc& lhs,
                                          const mkldnn::memory::desc& rhs,
                                          void* ptr,
                                          mkldnn::engine cpuEngine)
{
    if (!compareMkldnnMdFormats(lhs, rhs))
    {
        mkldnn::memory reorderIn = {lhs, cpuEngine, ptr};
        mkldnn::memory reorderOut = {rhs, cpuEngine};
        mkldnn::reorder convert(reorderIn, reorderOut);
        std::unordered_map<int, mkldnn::memory> execArgs = {{MKLDNN_ARG_SRC, reorderIn},
                                                            {MKLDNN_ARG_DST, reorderOut}};
        mkldnn::stream s(cpuEngine);
        try
        {
            convert.execute(s, execArgs);
            s.wait();
        }
        catch (const mkldnn::error& e)
        {
            throw ngraph_error("Could not run mkdnn primitive " + std::string(e.message));
        }
        return reorderOut;
    }
    else
    {
        return mkldnn::memory{lhs, cpuEngine, ptr};
    }
}

static void convertOutputLayout(mkldnn::memory& reorderIn,
                                const mkldnn::memory::desc& rhs,
                                void* ptr,
                                mkldnn::engine cpuEngine)
{
    mkldnn::memory reorderOut = {rhs, cpuEngine, ptr};
    mkldnn::reorder convert(reorderIn, reorderOut);
    std::unordered_map<int, mkldnn::memory> execArgs = {{MKLDNN_ARG_SRC, reorderIn},
                                                        {MKLDNN_ARG_DST, reorderOut}};
    mkldnn::stream s(cpuEngine);
    try
    {
        convert.execute(s, execArgs);
        s.wait();
    }
    catch (const mkldnn::error& e)
    {
        throw ngraph_error("Could not run mkdnn primitive " + std::string(e.message));
    }
}

static mkldnn::algorithm getConvAlgo()
{
#if defined(NGRAPH_ENABLE_CPU_CONV_AUTO)
    return mkldnn::algorithm::convolution_auto;
#else
    return mkldnn::algorithm::convolution_direct;
#endif
}

/// Callback for ConvBias
static void __mlir_mkldnn_convbias(size_t rank,
                                   StaticMemRef* memRefData,
                                   StaticMemRef* memRefWeights,
                                   StaticMemRef* memRefBias,
                                   StaticMemRef* memRefOutput,
                                   opAttrs* attrsPtr)
{
    mkldnn::memory::dims dataDims(rank);
    mkldnn::memory::dims dataStrides(rank);
    mkldnn::memory::dims weightsDims(rank);
    mkldnn::memory::dims weightsStrides(rank);
    mkldnn::memory::dims biasDims(1);
    mkldnn::memory::dims biasStrides(1);
    mkldnn::memory::dims resultDims(rank);
    mkldnn::memory::dims resultStrides(rank);
    biasDims[0] = memRefBias->shapeAndStrides[0];
    biasStrides[0] = memRefBias->shapeAndStrides[1];
    for (auto i = 0; i < rank; i++)
    {
        dataDims[i] = memRefData->shapeAndStrides[i];
        dataStrides[i] = memRefData->shapeAndStrides[rank + i];
        weightsDims[i] = memRefWeights->shapeAndStrides[i];
        weightsStrides[i] = memRefWeights->shapeAndStrides[rank + i];
        resultDims[i] = memRefOutput->shapeAndStrides[i];
        resultStrides[i] = memRefOutput->shapeAndStrides[rank + i];
    }

    // build mkldnn primitive and execute
    mkldnn::algorithm alg = getConvAlgo();
    mkldnn::memory::data_type dtype = mkldnn::memory::data_type::f32;
    auto dataDesc = mkldnn::memory::desc(dataDims, dtype, mkldnn::memory::FORMAT::any);
    auto dataDescOrigin = mkldnn::memory::desc(dataDims, dtype, dataStrides);
    auto weightsDesc = mkldnn::memory::desc(weightsDims, dtype, mkldnn::memory::FORMAT::any);
    auto weightsDescOrigin = mkldnn::memory::desc(weightsDims, dtype, weightsStrides);
    auto biasDesc = mkldnn::memory::desc(biasDims, dtype, mkldnn::memory::FORMAT::any);
    auto resultDesc = mkldnn::memory::desc(resultDims, dtype, mkldnn::memory::FORMAT::any);
    auto resultDescOrigin = mkldnn::memory::desc(resultDims, dtype, resultStrides);

    mkldnn::primitive_attr attr;
    mkldnn::engine cpuEngine(mkldnn::engine::kind::cpu, 0);
    mkldnn::convolution_forward::primitive_desc convPd;
    mkldnn::post_ops ops;
    const float opsScale = 1.f;
    const float opsAlpha = -0.f; // relu negative slope
    const float opsBeta = 0.f;
    ops.append_eltwise(opsScale, mkldnn::algorithm::eltwise_relu, opsAlpha, opsBeta);
    if (rank == 3)
    {
        auto convAttrs = (*attrsPtr).convAttrs1d;
        try
        {
            auto convDesc = mkldnn::convolution_forward::desc(
                mkldnn::prop_kind::forward_inference,
                alg,
                dataDesc,
                weightsDesc,
                biasDesc,
                resultDesc,
                mkldnn::memory::dims{convAttrs.windowStrides[0]},
                mkldnn::memory::dims{convAttrs.windowDilation[0] - 1},
                mkldnn::memory::dims{convAttrs.padBelow[0]},
                mkldnn::memory::dims{convAttrs.padAbove[0]});
            if (convAttrs.withRelu)
            {
                attr.set_post_ops(ops);
            }
            convPd = mkldnn::convolution_forward::primitive_desc(convDesc, attr, cpuEngine);
        }
        catch (const mkldnn::error& e)
        {
            throw ngraph_error("Could not create mkldnn conv descriptor " + std::string(e.message));
        }
    }
    else if (rank == 4)
    {
        auto convAttrs = (*attrsPtr).convAttrs2d;
        try
        {
            auto convDesc = mkldnn::convolution_forward::desc(
                mkldnn::prop_kind::forward_inference,
                alg,
                dataDesc,
                weightsDesc,
                biasDesc,
                resultDesc,
                mkldnn::memory::dims{convAttrs.windowStrides[0], convAttrs.windowStrides[1]},
                mkldnn::memory::dims{convAttrs.windowDilation[0] - 1,
                                     convAttrs.windowDilation[1] - 1},
                mkldnn::memory::dims{convAttrs.padBelow[0], convAttrs.padBelow[1]},
                mkldnn::memory::dims{convAttrs.padAbove[0], convAttrs.padAbove[1]});
            if (convAttrs.withRelu)
            {
                attr.set_post_ops(ops);
            }
            convPd = mkldnn::convolution_forward::primitive_desc(convDesc, attr, cpuEngine);
        }
        catch (const mkldnn::error& e)
        {
            throw ngraph_error("Could not create mkldnn conv descriptor " + std::string(e.message));
        }
    }
    else if (rank == 5)
    {
        auto convAttrs = (*attrsPtr).convAttrs3d;
        try
        {
            auto convDesc = mkldnn::convolution_forward::desc(
                mkldnn::prop_kind::forward_inference,
                alg,
                dataDesc,
                weightsDesc,
                biasDesc,
                resultDesc,
                mkldnn::memory::dims{convAttrs.windowStrides[0],
                                     convAttrs.windowStrides[1],
                                     convAttrs.windowStrides[2]},
                mkldnn::memory::dims{convAttrs.windowDilation[0] - 1,
                                     convAttrs.windowDilation[1] - 1,
                                     convAttrs.windowDilation[2] - 1},
                mkldnn::memory::dims{
                    convAttrs.padBelow[0], convAttrs.padBelow[1], convAttrs.padBelow[2]},
                mkldnn::memory::dims{
                    convAttrs.padAbove[0], convAttrs.padAbove[1], convAttrs.padAbove[2]});
            if (convAttrs.withRelu)
            {
                attr.set_post_ops(ops);
            }
            convPd = mkldnn::convolution_forward::primitive_desc(convDesc, attr, cpuEngine);
        }
        catch (const mkldnn::error& e)
        {
            throw ngraph_error("Could not create mkldnn conv descriptor " + std::string(e.message));
        }
    }

    mkldnn::convolution_forward conv(convPd);
    mkldnn::memory data =
        convertLayoutIfDiff(dataDescOrigin, convPd.src_desc(), memRefData->allocatedPtr, cpuEngine);
    mkldnn::memory weights = convertLayoutIfDiff(
        weightsDescOrigin, convPd.weights_desc(), memRefWeights->allocatedPtr, cpuEngine);
    mkldnn::memory bias{convPd.bias_desc(), cpuEngine, memRefBias->allocatedPtr};
    mkldnn::memory out;
    bool needConvert = false;
    if (!compareMkldnnMdFormats(resultDescOrigin, convPd.dst_desc()))
    {
        out = mkldnn::memory(convPd.dst_desc(), cpuEngine);
        needConvert = true;
    }
    else
    {
        out = mkldnn::memory(convPd.dst_desc(), cpuEngine, memRefOutput->allocatedPtr);
    }

    std::unordered_map<int, mkldnn::memory> execArgs = {{MKLDNN_ARG_SRC, data},
                                                        {MKLDNN_ARG_WEIGHTS, weights},
                                                        {MKLDNN_ARG_BIAS, bias},
                                                        {MKLDNN_ARG_DST, out}};

    mkldnn::stream s(cpuEngine);
    try
    {
        conv.execute(s, execArgs);
        s.wait();
    }
    catch (const mkldnn::error& e)
    {
        throw ngraph_error("Could not run mkdnn primitive " + std::string(e.message));
    }

    if (needConvert)
    {
        convertOutputLayout(out, resultDescOrigin, memRefOutput->allocatedPtr, cpuEngine);
    }
}

/// Callback for MaxPoolBackprop
static void __mlir_mkldnn_maxpoolbackprop(size_t rank,
                                          StaticMemRef* memRefSrc,
                                          StaticMemRef* memRefDelta,
                                          StaticMemRef* memRefOutput,
                                          opAttrs* attrsPtr)
{
    mkldnn::memory::dims srcDims(rank);
    mkldnn::memory::dims srcStrides(rank);
    mkldnn::memory::dims deltaDims(rank);
    mkldnn::memory::dims deltaStrides(rank);
    mkldnn::memory::dims outDims(rank);
    mkldnn::memory::dims outStrides(rank);
    for (auto i = 0; i < rank; i++)
    {
        srcDims[i] = memRefSrc->shapeAndStrides[i];
        srcStrides[i] = memRefSrc->shapeAndStrides[rank + i];
        deltaDims[i] = memRefDelta->shapeAndStrides[i];
        deltaStrides[i] = memRefDelta->shapeAndStrides[rank + i];
        outDims[i] = memRefOutput->shapeAndStrides[i];
        outStrides[i] = memRefOutput->shapeAndStrides[rank + i];
    }

    // build mkldnn primitive and execute
    auto requiredFormat = rank == 4 ? mkldnn::memory::FORMAT::nchw : mkldnn::memory::FORMAT::ncdhw;
    mkldnn::memory::data_type dtype = mkldnn::memory::data_type::f32;
    auto diffDstDesc = mkldnn::memory::desc(deltaDims, dtype, requiredFormat);
    auto diffSrcDesc = mkldnn::memory::desc(outDims, dtype, requiredFormat);
    auto srcDescOrigin = mkldnn::memory::desc(srcDims, dtype, srcStrides);
    auto diffDstDescOrigin = mkldnn::memory::desc(deltaDims, dtype, deltaStrides);
    auto diffSrcDescOrigin = mkldnn::memory::desc(outDims, dtype, outStrides);

    mkldnn::primitive_attr attr;
    mkldnn::engine cpuEngine(mkldnn::engine::kind::cpu, 0);
    mkldnn::pooling_forward::primitive_desc maxpoolPdF;
    mkldnn::pooling_backward::primitive_desc maxpoolPdB;
    if (rank == 4)
    {
        poolAttrs<2> pAttrs = (*attrsPtr).poolAttrs2d;
        try
        {
            auto maxpoolDescF = mkldnn::pooling_forward::desc(
                mkldnn::prop_kind::forward_training,
                mkldnn::algorithm::pooling_max,
                diffSrcDesc,
                diffDstDesc,
                mkldnn::memory::dims{pAttrs.windowStrides[0], pAttrs.windowStrides[1]},
                mkldnn::memory::dims{pAttrs.windowShape[0], pAttrs.windowShape[1]},
                mkldnn::memory::dims{pAttrs.padBelow[0], pAttrs.padBelow[1]},
                mkldnn::memory::dims{pAttrs.padAbove[0], pAttrs.padAbove[1]});
            auto maxpoolDescB = mkldnn::pooling_backward::desc(
                mkldnn::algorithm::pooling_max,
                diffSrcDesc,
                diffDstDesc,
                mkldnn::memory::dims{pAttrs.windowStrides[0], pAttrs.windowStrides[1]},
                mkldnn::memory::dims{pAttrs.windowShape[0], pAttrs.windowShape[1]},
                mkldnn::memory::dims{pAttrs.padBelow[0], pAttrs.padBelow[1]},
                mkldnn::memory::dims{pAttrs.padAbove[0], pAttrs.padAbove[1]});
            maxpoolPdF = mkldnn::pooling_forward::primitive_desc(maxpoolDescF, attr, cpuEngine);
            maxpoolPdB =
                mkldnn::pooling_backward::primitive_desc(maxpoolDescB, attr, cpuEngine, maxpoolPdF);
        }
        catch (const mkldnn::error& e)
        {
            throw ngraph_error("Could not create mkldnn max pooling descriptor " +
                               std::string(e.message));
        }
    }
    else if (rank == 5)
    {
        poolAttrs<3> pAttrs = (*attrsPtr).poolAttrs3d;
        try
        {
            auto maxpoolDescF = mkldnn::pooling_forward::desc(
                mkldnn::prop_kind::forward_training,
                mkldnn::algorithm::pooling_max,
                diffSrcDesc,
                diffDstDesc,
                mkldnn::memory::dims{
                    pAttrs.windowStrides[0], pAttrs.windowStrides[1], pAttrs.windowStrides[2]},
                mkldnn::memory::dims{
                    pAttrs.windowShape[0], pAttrs.windowShape[1], pAttrs.windowShape[2]},
                mkldnn::memory::dims{pAttrs.padBelow[0], pAttrs.padBelow[1], pAttrs.padBelow[2]},
                mkldnn::memory::dims{pAttrs.padAbove[0], pAttrs.padAbove[1], pAttrs.padAbove[2]});
            auto maxpoolDescB = mkldnn::pooling_backward::desc(
                mkldnn::algorithm::pooling_max,
                diffSrcDesc,
                diffDstDesc,
                mkldnn::memory::dims{
                    pAttrs.windowStrides[0], pAttrs.windowStrides[1], pAttrs.windowStrides[2]},
                mkldnn::memory::dims{
                    pAttrs.windowShape[0], pAttrs.windowShape[1], pAttrs.windowShape[2]},
                mkldnn::memory::dims{pAttrs.padBelow[0], pAttrs.padBelow[1], pAttrs.padBelow[2]},
                mkldnn::memory::dims{pAttrs.padAbove[0], pAttrs.padAbove[1], pAttrs.padAbove[2]});
            maxpoolPdF = mkldnn::pooling_forward::primitive_desc(maxpoolDescF, attr, cpuEngine);
            maxpoolPdB =
                mkldnn::pooling_backward::primitive_desc(maxpoolDescB, attr, cpuEngine, maxpoolPdF);
        }
        catch (const mkldnn::error& e)
        {
            throw ngraph_error("Could not create mkldnn max pooling descriptor " +
                               std::string(e.message));
        }
    }

    mkldnn::pooling_forward maxpoolF(maxpoolPdF);
    mkldnn::memory srcMem = convertLayoutIfDiff(
        srcDescOrigin, maxpoolPdB.diff_src_desc(), memRefSrc->allocatedPtr, cpuEngine);
    mkldnn::memory dstMem{maxpoolPdB.diff_dst_desc(), cpuEngine};
    mkldnn::memory workspace{maxpoolPdF.workspace_desc(), cpuEngine};

    mkldnn::pooling_backward maxpoolB(maxpoolPdB);
    mkldnn::memory diffDst = convertLayoutIfDiff(
        diffDstDescOrigin, maxpoolPdB.diff_dst_desc(), memRefDelta->allocatedPtr, cpuEngine);
    mkldnn::memory diffSrc;
    bool needConvert = false;
    if (!compareMkldnnMdFormats(diffSrcDescOrigin, maxpoolPdB.diff_src_desc()))
    {
        diffSrc = mkldnn::memory(maxpoolPdB.diff_src_desc(), cpuEngine);
        needConvert = true;
    }
    else
    {
        diffSrc = mkldnn::memory(maxpoolPdB.diff_src_desc(), cpuEngine, memRefOutput->allocatedPtr);
    }

    std::unordered_map<int, mkldnn::memory> execArgsF = {
        {MKLDNN_ARG_SRC, srcMem}, {MKLDNN_ARG_WORKSPACE, workspace}, {MKLDNN_ARG_DST, dstMem}};
    std::unordered_map<int, mkldnn::memory> execArgsB = {{MKLDNN_ARG_DIFF_DST, diffDst},
                                                         {MKLDNN_ARG_WORKSPACE, workspace},
                                                         {MKLDNN_ARG_DIFF_SRC, diffSrc}};

    mkldnn::stream s(cpuEngine);
    try
    {
        maxpoolF.execute(s, execArgsF);
        s.wait();
        maxpoolB.execute(s, execArgsB);
        s.wait();
    }
    catch (const mkldnn::error& e)
    {
        throw ngraph_error("Could not run mkdnn primitive " + std::string(e.message));
    }

    if (needConvert)
    {
        convertOutputLayout(diffSrc, diffSrcDescOrigin, memRefOutput->allocatedPtr, cpuEngine);
    }
}

/// Callback for AvgPoolBackprop
static void __mlir_mkldnn_avgpoolbackprop(size_t rank,
                                          StaticMemRef* memRefInput,
                                          StaticMemRef* memRefOutput,
                                          opAttrs* attrsPtr)
{
    mkldnn::memory::dims dims(rank);
    mkldnn::memory::dims strides(rank);
    mkldnn::memory::dims outDims(rank);
    mkldnn::memory::dims outStrides(rank);
    for (auto i = 0; i < rank; i++)
    {
        dims[i] = memRefInput->shapeAndStrides[i];
        strides[i] = memRefInput->shapeAndStrides[rank + i];
        outDims[i] = memRefOutput->shapeAndStrides[i];
        outStrides[i] = memRefOutput->shapeAndStrides[rank + i];
    }

    // build mkldnn primitive and execute
    auto requiredFormat = rank == 4 ? mkldnn::memory::FORMAT::nchw : mkldnn::memory::FORMAT::ncdhw;
    mkldnn::memory::data_type dtype = mkldnn::memory::data_type::f32;
    auto diffDstDesc = mkldnn::memory::desc(dims, dtype, requiredFormat);
    auto diffSrcDesc = mkldnn::memory::desc(outDims, dtype, requiredFormat);
    auto diffDstDescOrigin = mkldnn::memory::desc(dims, dtype, strides);
    auto diffSrcDescOrigin = mkldnn::memory::desc(outDims, dtype, outStrides);
    mkldnn::primitive_attr attr;
    mkldnn::engine cpuEngine(mkldnn::engine::kind::cpu, 0);
    mkldnn::pooling_backward::primitive_desc avgpoolPdB;
    if (rank == 4)
    {
        poolAttrs<2> pAttrs = (*attrsPtr).poolAttrs2d;
        try
        {
            auto avgpoolDescF = mkldnn::pooling_forward::desc(
                mkldnn::prop_kind::forward_training,
                (pAttrs.includePaddingInAvgComputation
                     ? mkldnn::algorithm::pooling_avg_include_padding
                     : mkldnn::algorithm::pooling_avg_exclude_padding),
                diffSrcDesc,
                diffDstDesc,
                mkldnn::memory::dims{pAttrs.windowStrides[0], pAttrs.windowStrides[1]},
                mkldnn::memory::dims{pAttrs.windowShape[0], pAttrs.windowShape[1]},
                mkldnn::memory::dims{pAttrs.padBelow[0], pAttrs.padBelow[1]},
                mkldnn::memory::dims{pAttrs.padAbove[0], pAttrs.padAbove[1]});
            auto avgpoolDescB = mkldnn::pooling_backward::desc(
                (pAttrs.includePaddingInAvgComputation
                     ? mkldnn::algorithm::pooling_avg_include_padding
                     : mkldnn::algorithm::pooling_avg_exclude_padding),
                diffSrcDesc,
                diffDstDesc,
                mkldnn::memory::dims{pAttrs.windowStrides[0], pAttrs.windowStrides[1]},
                mkldnn::memory::dims{pAttrs.windowShape[0], pAttrs.windowShape[1]},
                mkldnn::memory::dims{pAttrs.padBelow[0], pAttrs.padBelow[1]},
                mkldnn::memory::dims{pAttrs.padAbove[0], pAttrs.padAbove[1]});
            auto avgpoolPdF =
                mkldnn::pooling_forward::primitive_desc(avgpoolDescF, attr, cpuEngine);
            avgpoolPdB =
                mkldnn::pooling_backward::primitive_desc(avgpoolDescB, attr, cpuEngine, avgpoolPdF);
        }
        catch (const mkldnn::error& e)
        {
            throw ngraph_error("Could not create mkldnn avg pooling descriptor " +
                               std::string(e.message));
        }
    }
    else if (rank == 5)
    {
        poolAttrs<3> pAttrs = (*attrsPtr).poolAttrs3d;
        try
        {
            auto avgpoolDescF = mkldnn::pooling_forward::desc(
                mkldnn::prop_kind::forward_training,
                (pAttrs.includePaddingInAvgComputation
                     ? mkldnn::algorithm::pooling_avg_include_padding
                     : mkldnn::algorithm::pooling_avg_exclude_padding),
                diffSrcDesc,
                diffDstDesc,
                mkldnn::memory::dims{
                    pAttrs.windowStrides[0], pAttrs.windowStrides[1], pAttrs.windowStrides[2]},
                mkldnn::memory::dims{
                    pAttrs.windowShape[0], pAttrs.windowShape[1], pAttrs.windowShape[2]},
                mkldnn::memory::dims{pAttrs.padBelow[0], pAttrs.padBelow[1], pAttrs.padBelow[2]},
                mkldnn::memory::dims{pAttrs.padAbove[0], pAttrs.padAbove[1], pAttrs.padAbove[2]});
            auto avgpoolDescB = mkldnn::pooling_backward::desc(
                (pAttrs.includePaddingInAvgComputation
                     ? mkldnn::algorithm::pooling_avg_include_padding
                     : mkldnn::algorithm::pooling_avg_exclude_padding),
                diffSrcDesc,
                diffDstDesc,
                mkldnn::memory::dims{
                    pAttrs.windowStrides[0], pAttrs.windowStrides[1], pAttrs.windowStrides[2]},
                mkldnn::memory::dims{
                    pAttrs.windowShape[0], pAttrs.windowShape[1], pAttrs.windowShape[2]},
                mkldnn::memory::dims{pAttrs.padBelow[0], pAttrs.padBelow[1], pAttrs.padBelow[2]},
                mkldnn::memory::dims{pAttrs.padAbove[0], pAttrs.padAbove[1], pAttrs.padAbove[2]});
            auto avgpoolPdF =
                mkldnn::pooling_forward::primitive_desc(avgpoolDescF, attr, cpuEngine);
            avgpoolPdB =
                mkldnn::pooling_backward::primitive_desc(avgpoolDescB, attr, cpuEngine, avgpoolPdF);
        }
        catch (const mkldnn::error& e)
        {
            throw ngraph_error("Could not create mkldnn avg pooling descriptor " +
                               std::string(e.message));
        }
    }

    mkldnn::pooling_backward avgpool(avgpoolPdB);
    mkldnn::memory in = convertLayoutIfDiff(
        diffDstDescOrigin, avgpoolPdB.diff_dst_desc(), memRefInput->allocatedPtr, cpuEngine);
    mkldnn::memory out;
    bool needConvert = false;
    if (!compareMkldnnMdFormats(diffSrcDescOrigin, avgpoolPdB.diff_src_desc()))
    {
        out = mkldnn::memory(avgpoolPdB.diff_src_desc(), cpuEngine);
        needConvert = true;
    }
    else
    {
        out = mkldnn::memory(avgpoolPdB.diff_src_desc(), cpuEngine, memRefOutput->allocatedPtr);
    }
    std::unordered_map<int, mkldnn::memory> execArgs = {{MKLDNN_ARG_DIFF_DST, in},
                                                        {MKLDNN_ARG_DIFF_SRC, out}};

    mkldnn::stream s(cpuEngine);
    try
    {
        avgpool.execute(s, execArgs);
        s.wait();
    }
    catch (const mkldnn::error& e)
    {
        throw ngraph_error("Could not run mkdnn primitive " + std::string(e.message));
    }

    if (needConvert)
    {
        convertOutputLayout(out, diffSrcDescOrigin, memRefOutput->allocatedPtr, cpuEngine);
    }
}

/// Callback for AvgPool and MaxPool
static void __mlir_mkldnn_pooling(size_t rank,
                                  StaticMemRef* memRefInput,
                                  StaticMemRef* memRefOutput,
                                  opAttrs* attrsPtr,
                                  OpType type)
{
    mkldnn::memory::dims dims(rank);
    mkldnn::memory::dims strides(rank);
    mkldnn::memory::dims outDims(rank);
    mkldnn::memory::dims outStrides(rank);
    for (auto i = 0; i < rank; i++)
    {
        dims[i] = memRefInput->shapeAndStrides[i];
        strides[i] = memRefInput->shapeAndStrides[rank + i];
        outDims[i] = memRefOutput->shapeAndStrides[i];
        outStrides[i] = memRefOutput->shapeAndStrides[rank + i];
    }

    // build mkldnn primitive and execute
    auto requiredFormat = rank == 4 ? mkldnn::memory::FORMAT::nchw : mkldnn::memory::FORMAT::ncdhw;
    mkldnn::memory::data_type dtype = mkldnn::memory::data_type::f32;
    auto inputDesc = mkldnn::memory::desc(dims, dtype, requiredFormat);
    auto resultDesc = mkldnn::memory::desc(outDims, dtype, requiredFormat);
    auto inputDescOrigin = mkldnn::memory::desc(dims, dtype, strides);
    auto resultDescOrigin = mkldnn::memory::desc(outDims, dtype, outStrides);
    mkldnn::primitive_attr attr;
    mkldnn::engine cpuEngine(mkldnn::engine::kind::cpu, 0);
    mkldnn::pooling_forward::primitive_desc poolPd;
    if (rank == 4)
    {
        poolAttrs<2> pAttrs = (*attrsPtr).poolAttrs2d;
        mkldnn::algorithm alg = type == OpType::MAXPOOL
                                    ? mkldnn::algorithm::pooling_max
                                    : (pAttrs.includePaddingInAvgComputation
                                           ? mkldnn::algorithm::pooling_avg_include_padding
                                           : mkldnn::algorithm::pooling_avg_exclude_padding);
        try
        {
            auto poolDesc = mkldnn::pooling_forward::desc(
                mkldnn::prop_kind::forward_inference,
                alg,
                inputDesc,
                resultDesc,
                mkldnn::memory::dims{pAttrs.windowStrides[0], pAttrs.windowStrides[1]},
                mkldnn::memory::dims{pAttrs.windowShape[0], pAttrs.windowShape[1]},
                mkldnn::memory::dims{pAttrs.padBelow[0], pAttrs.padBelow[1]},
                mkldnn::memory::dims{pAttrs.padAbove[0], pAttrs.padAbove[1]});
            poolPd = mkldnn::pooling_forward::primitive_desc(poolDesc, attr, cpuEngine);
        }
        catch (const mkldnn::error& e)
        {
            throw ngraph_error("Could not create mkldnn pooling descriptor " +
                               std::string(e.message));
        }
    }
    else if (rank == 5)
    {
        poolAttrs<3> pAttrs = (*attrsPtr).poolAttrs3d;
        mkldnn::algorithm alg = type == OpType::MAXPOOL
                                    ? mkldnn::algorithm::pooling_max
                                    : (pAttrs.includePaddingInAvgComputation
                                           ? mkldnn::algorithm::pooling_avg_include_padding
                                           : mkldnn::algorithm::pooling_avg_exclude_padding);
        try
        {
            auto poolDesc = mkldnn::pooling_forward::desc(
                mkldnn::prop_kind::forward_inference,
                alg,
                inputDesc,
                resultDesc,
                mkldnn::memory::dims{
                    pAttrs.windowStrides[0], pAttrs.windowStrides[1], pAttrs.windowStrides[2]},
                mkldnn::memory::dims{
                    pAttrs.windowShape[0], pAttrs.windowShape[1], pAttrs.windowShape[2]},
                mkldnn::memory::dims{pAttrs.padBelow[0], pAttrs.padBelow[1], pAttrs.padBelow[2]},
                mkldnn::memory::dims{pAttrs.padAbove[0], pAttrs.padAbove[1], pAttrs.padAbove[2]});
            poolPd = mkldnn::pooling_forward::primitive_desc(poolDesc, attr, cpuEngine);
        }
        catch (const mkldnn::error& e)
        {
            throw ngraph_error("Could not create mkldnn pooing descriptor " +
                               std::string(e.message));
        }
    }

    mkldnn::pooling_forward pool(poolPd);
    mkldnn::memory in = convertLayoutIfDiff(
        inputDescOrigin, poolPd.src_desc(), memRefInput->allocatedPtr, cpuEngine);
    mkldnn::memory out;
    bool needConvert = false;
    if (!compareMkldnnMdFormats(resultDescOrigin, poolPd.dst_desc()))
    {
        out = mkldnn::memory(poolPd.dst_desc(), cpuEngine);
        needConvert = true;
    }
    else
    {
        out = mkldnn::memory(poolPd.dst_desc(), cpuEngine, memRefOutput->allocatedPtr);
    }
    std::unordered_map<int, mkldnn::memory> execArgs = {{MKLDNN_ARG_SRC, in},
                                                        {MKLDNN_ARG_DST, out}};

    mkldnn::stream s(cpuEngine);
    try
    {
        pool.execute(s, execArgs);
        s.wait();
    }
    catch (const mkldnn::error& e)
    {
        throw ngraph_error("Could not run mkdnn primitive " + std::string(e.message));
    }

    if (needConvert)
    {
        convertOutputLayout(out, resultDescOrigin, memRefOutput->allocatedPtr, cpuEngine);
    }
}

/// Callback for Softmax
static void __mlir_mkldnn_softmax(size_t rank,
                                  StaticMemRef* memRefInput,
                                  StaticMemRef* memRefOutput,
                                  opAttrs* attrsPtr)
{
    mkldnn::memory::dims dims(rank);
    mkldnn::memory::dims strides(rank);
    for (auto i = 0; i < rank; i++)
    {
        dims[i] = memRefInput->shapeAndStrides[i];
        strides[i] = memRefInput->shapeAndStrides[rank + i];
    }
    auto softmaxAxis = (*attrsPtr).intAttr;

    // build mkldnn primitive and execute
    mkldnn::memory::data_type dtype = mkldnn::memory::data_type::f32;
    auto inputDesc = mkldnn::memory::desc(dims, dtype, strides);
    mkldnn::softmax_forward::primitive_desc softmaxPd;
    mkldnn::engine cpuEngine(mkldnn::engine::kind::cpu, 0);
    try
    {
        auto softmaxDesc = mkldnn::softmax_forward::desc(
            mkldnn::prop_kind::forward_scoring, inputDesc, softmaxAxis);
        mkldnn::primitive_attr attr;
        softmaxPd = mkldnn::softmax_forward::primitive_desc(softmaxDesc, attr, cpuEngine);
    }
    catch (const mkldnn::error& e)
    {
        throw ngraph_error("Could not create mkldnn softmax descriptor " + std::string(e.message));
    }
    mkldnn::softmax_forward softmax(softmaxPd);

    mkldnn::memory in{softmaxPd.src_desc(), cpuEngine, memRefInput->allocatedPtr};
    mkldnn::memory out{softmaxPd.dst_desc(), cpuEngine, memRefOutput->allocatedPtr};

    std::unordered_map<int, mkldnn::memory> execArgs = {{MKLDNN_ARG_SRC, in},
                                                        {MKLDNN_ARG_DST, out}};

    mkldnn::stream s(cpuEngine);
    try
    {
        softmax.execute(s, execArgs);
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
                               opAttrs* attrsPtr)
{
    gemmAttrs gAttrs = (*attrsPtr).gemmAttrs2d;
    ;
    cblas::cblas_sgemm(cblas::Layout::RowMajor,
                       gAttrs.transposeA ? cblas::Transpose::Transpose : cblas::Transpose::None,
                       gAttrs.transposeB ? cblas::Transpose::Transpose : cblas::Transpose::None,
                       gAttrs.m,
                       gAttrs.n,
                       gAttrs.k,
                       1.0f,
                       reinterpret_cast<float*>(memRefmatA->allocatedPtr),
                       std::max<size_t>(1, gAttrs.lda),
                       reinterpret_cast<float*>(memRefmatB->allocatedPtr),
                       std::max<size_t>(1, gAttrs.ldb),
                       0.0f,
                       reinterpret_cast<float*>(memRefmatC->allocatedPtr),
                       std::max<size_t>(1, gAttrs.ldc));
}

/// Callback for Gemm
static void __mlir_cblas_sgemm_with_bias(StaticMemRef* memRefmatA,
                                         StaticMemRef* memRefmatB,
                                         StaticMemRef* memRefmatC,
                                         StaticMemRef* memRefmatOut,
                                         opAttrs* attrsPtr)
{
    gemmAttrs gAttrs = (*attrsPtr).gemmAttrs2d;
    auto transposeA = gAttrs.transposeA;
    auto transposeB = gAttrs.transposeB;
    auto m = gAttrs.m;
    auto n = gAttrs.n;
    auto k = gAttrs.k;
    auto lda = gAttrs.lda;
    auto ldb = gAttrs.ldb;
    auto ldc = gAttrs.ldc;
    auto alpha = gAttrs.alpha;
    auto beta = gAttrs.beta;
    auto broadcastHint = gAttrs.broadcastHint;

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

    if (broadcastHint == BroadcastType::ROW)
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
    else if (broadcastHint == BroadcastType::COLUMN)
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
    else if (broadcastHint == BroadcastType::ROWCOLUMN)
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
    else if (broadcastHint == BroadcastType::NONE)
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
    else
    {
        NGRAPH_UNREACHABLE("Unsupported broadcast");
    }
}

extern "C" void
    _mlir_ciface_callback_1_input(void* input, void* output, void* attrsPtr, OpType type)
{
    auto unrankedMemRefInput = reinterpret_cast<UnrankedMemRef*>(input);
    auto unrankedMemRefOutput = reinterpret_cast<UnrankedMemRef*>(output);

    if (type == OpType::SOFTMAX)
    {
        __mlir_mkldnn_softmax(unrankedMemRefInput->rank,
                              unrankedMemRefInput->memRefDescPtr,
                              unrankedMemRefOutput->memRefDescPtr,
                              static_cast<opAttrs*>(attrsPtr));
    }
    else if (type == OpType::AVGPOOL || type == OpType::MAXPOOL)
    {
        __mlir_mkldnn_pooling(unrankedMemRefInput->rank,
                              unrankedMemRefInput->memRefDescPtr,
                              unrankedMemRefOutput->memRefDescPtr,
                              static_cast<opAttrs*>(attrsPtr),
                              type);
    }
    else if (type == OpType::AVGPOOLBACKPROP)
    {
        __mlir_mkldnn_avgpoolbackprop(unrankedMemRefInput->rank,
                                      unrankedMemRefInput->memRefDescPtr,
                                      unrankedMemRefOutput->memRefDescPtr,
                                      static_cast<opAttrs*>(attrsPtr));
    }
    else
    {
        NGRAPH_UNREACHABLE("Unsupported type");
    }
}

extern "C" void _mlir_ciface_callback_2_inputs(
    void* input0, void* input1, void* output, void* attrsPtr, OpType type)
{
    auto unrankedMemRefInput0 = reinterpret_cast<UnrankedMemRef*>(input0);
    auto unrankedMemRefInput1 = reinterpret_cast<UnrankedMemRef*>(input1);
    auto unrankedMemRefOutput = reinterpret_cast<UnrankedMemRef*>(output);

    if (type == OpType::MATMUL)
    {
        __mlir_cblas_sgemm(unrankedMemRefInput0->memRefDescPtr,
                           unrankedMemRefInput1->memRefDescPtr,
                           unrankedMemRefOutput->memRefDescPtr,
                           static_cast<opAttrs*>(attrsPtr));
    }
    else if (type == OpType::MAXPOOLBACKPROP)
    {
        __mlir_mkldnn_maxpoolbackprop(unrankedMemRefInput0->rank,
                                      unrankedMemRefInput0->memRefDescPtr,
                                      unrankedMemRefInput1->memRefDescPtr,
                                      unrankedMemRefOutput->memRefDescPtr,
                                      static_cast<opAttrs*>(attrsPtr));
    }
    else
    {
        NGRAPH_UNREACHABLE("Unsupported type");
    }
}

extern "C" void _mlir_ciface_callback_3_inputs(
    void* input0, void* input1, void* input2, void* output, void* attrsPtr, OpType type)
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
                                     static_cast<opAttrs*>(attrsPtr));
    }
    else if (type == OpType::CONVOLUTIONBIAS)
    {
        __mlir_mkldnn_convbias(unrankedMemRefInput0->rank,
                               unrankedMemRefInput0->memRefDescPtr,
                               unrankedMemRefInput1->memRefDescPtr,
                               unrankedMemRefInput2->memRefDescPtr,
                               unrankedMemRefOutput->memRefDescPtr,
                               static_cast<opAttrs*>(attrsPtr));
    }
    else
    {
        NGRAPH_UNREACHABLE("Unsupported type");
    }
}
