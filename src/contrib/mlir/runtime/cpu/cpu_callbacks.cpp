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
// Follows nGraph naming convention for public APIs only, else MLIR naming
// convention.

#include "callback_utils.hpp"
#include "contrib/mlir/backend/cpu/cpu_backend.hpp"
#include "cpu_runtime.hpp"
#include "ngraph/check.hpp"

#include "ngraph/runtime/cpu/cpu_kernels.hpp"
#include "ngraph/runtime/cpu/dnnl_utils.hpp"

using namespace ngraph;
using namespace ngraph::runtime::ngmlir;

static bool inline compareMkldnnDims(dnnl_dims_t& arr1, dnnl_dims_t& arr2, size_t size)
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

static bool compareMkldnnStridesOrder(dnnl_dims_t& strides1, dnnl_dims_t& strides2, size_t size)
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

static bool compareMkldnnMdFormats(const dnnl::memory::desc& lhs, const dnnl::memory::desc& rhs)
{
    dnnl_memory_desc_t md1 = lhs.data, md2 = rhs.data;

    if (md1.format_kind != md2.format_kind)
    {
        return false;
    }

    if (md1.format_kind != static_cast<dnnl_format_kind_t>(dnnl::memory::format_kind::blocked))
    {
        // dnnl not implemented yet
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

static dnnl::memory convertLayoutIfDiff(const dnnl::memory::desc& lhs,
                                        const dnnl::memory::desc& rhs,
                                        void* ptr,
                                        dnnl::engine cpuEngine)
{
    if (!compareMkldnnMdFormats(lhs, rhs))
    {
        dnnl::memory reorderIn = {lhs, cpuEngine, ptr};
        dnnl::memory reorderOut = {rhs, cpuEngine};
        dnnl::reorder convert(reorderIn, reorderOut);
        std::unordered_map<int, dnnl::memory> execArgs = {{DNNL_ARG_SRC, reorderIn},
                                                          {DNNL_ARG_DST, reorderOut}};
        dnnl::stream s(cpuEngine);
        try
        {
            convert.execute(s, execArgs);
            s.wait();
        }
        catch (const dnnl::error& e)
        {
            throw ngraph_error("Could not run mkdnn primitive " + std::string(e.message));
        }
        return reorderOut;
    }
    else
    {
        return dnnl::memory{lhs, cpuEngine, ptr};
    }
}

static void convertOutputLayout(dnnl::memory& reorderIn,
                                const dnnl::memory::desc& rhs,
                                void* ptr,
                                dnnl::engine cpuEngine)
{
    dnnl::memory reorderOut = {rhs, cpuEngine, ptr};
    dnnl::reorder convert(reorderIn, reorderOut);
    std::unordered_map<int, dnnl::memory> execArgs = {{DNNL_ARG_SRC, reorderIn},
                                                      {DNNL_ARG_DST, reorderOut}};
    dnnl::stream s(cpuEngine);
    try
    {
        convert.execute(s, execArgs);
        s.wait();
    }
    catch (const dnnl::error& e)
    {
        throw ngraph_error("Could not run mkdnn primitive " + std::string(e.message));
    }
}

static dnnl::algorithm getConvAlgo()
{
#if defined(NGRAPH_CPU_CONV_AUTO_ENABLE)
    return dnnl::algorithm::convolution_auto;
#else
    return dnnl::algorithm::convolution_direct;
#endif
}

/// Callback for ConvBias
static void __mlir_dnnl_convbias(size_t rank,
                                 StaticMemRef* memRefData,
                                 StaticMemRef* memRefWeights,
                                 StaticMemRef* memRefBias,
                                 StaticMemRef* memRefOutput,
                                 opAttrs* attrsPtr)
{
    dnnl::memory::dims dataDims(rank);
    dnnl::memory::dims dataStrides(rank);
    dnnl::memory::dims weightsDims(rank);
    dnnl::memory::dims weightsStrides(rank);
    dnnl::memory::dims biasDims(1);
    dnnl::memory::dims biasStrides(1);
    dnnl::memory::dims resultDims(rank);
    dnnl::memory::dims resultStrides(rank);
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

    // build dnnl primitive and execute
    dnnl::algorithm alg = getConvAlgo();
    dnnl::memory::data_type dtype = dnnl::memory::data_type::f32;
    auto dataDesc = dnnl::memory::desc(dataDims, dtype, dnnl::memory::FORMAT::any);
    auto dataDescOrigin = dnnl::memory::desc(dataDims, dtype, dataStrides);
    auto weightsDesc = dnnl::memory::desc(weightsDims, dtype, dnnl::memory::FORMAT::any);
    auto weightsDescOrigin = dnnl::memory::desc(weightsDims, dtype, weightsStrides);
    auto biasDesc = dnnl::memory::desc(biasDims, dtype, dnnl::memory::FORMAT::any);
    auto resultDesc = dnnl::memory::desc(resultDims, dtype, dnnl::memory::FORMAT::any);
    auto resultDescOrigin = dnnl::memory::desc(resultDims, dtype, resultStrides);

    dnnl::primitive_attr attr;
    dnnl::engine cpuEngine(dnnl::engine::kind::cpu, 0);
    dnnl::convolution_forward::primitive_desc convPd;
    dnnl::post_ops ops;
    const float opsScale = 1.f;
    const float opsAlpha = -0.f; // relu negative slope
    const float opsBeta = 0.f;
    ops.append_eltwise(opsScale, dnnl::algorithm::eltwise_relu, opsAlpha, opsBeta);
    if (rank == 3)
    {
        auto convAttrs = (*attrsPtr).convAttrs1d;
        try
        {
            auto convDesc =
                dnnl::convolution_forward::desc(dnnl::prop_kind::forward_inference,
                                                alg,
                                                dataDesc,
                                                weightsDesc,
                                                biasDesc,
                                                resultDesc,
                                                dnnl::memory::dims{convAttrs.windowStrides[0]},
                                                dnnl::memory::dims{convAttrs.windowDilation[0] - 1},
                                                dnnl::memory::dims{convAttrs.padBelow[0]},
                                                dnnl::memory::dims{convAttrs.padAbove[0]});
            if (convAttrs.withRelu)
            {
                attr.set_post_ops(ops);
            }
            convPd = dnnl::convolution_forward::primitive_desc(convDesc, attr, cpuEngine);
        }
        catch (const dnnl::error& e)
        {
            throw ngraph_error("Could not create dnnl conv descriptor " + std::string(e.message));
        }
    }
    else if (rank == 4)
    {
        auto convAttrs = (*attrsPtr).convAttrs2d;
        try
        {
            auto convDesc = dnnl::convolution_forward::desc(
                dnnl::prop_kind::forward_inference,
                alg,
                dataDesc,
                weightsDesc,
                biasDesc,
                resultDesc,
                dnnl::memory::dims{convAttrs.windowStrides[0], convAttrs.windowStrides[1]},
                dnnl::memory::dims{convAttrs.windowDilation[0] - 1,
                                   convAttrs.windowDilation[1] - 1},
                dnnl::memory::dims{convAttrs.padBelow[0], convAttrs.padBelow[1]},
                dnnl::memory::dims{convAttrs.padAbove[0], convAttrs.padAbove[1]});
            if (convAttrs.withRelu)
            {
                attr.set_post_ops(ops);
            }
            convPd = dnnl::convolution_forward::primitive_desc(convDesc, attr, cpuEngine);
        }
        catch (const dnnl::error& e)
        {
            throw ngraph_error("Could not create dnnl conv descriptor " + std::string(e.message));
        }
    }
    else if (rank == 5)
    {
        auto convAttrs = (*attrsPtr).convAttrs3d;
        try
        {
            auto convDesc = dnnl::convolution_forward::desc(
                dnnl::prop_kind::forward_inference,
                alg,
                dataDesc,
                weightsDesc,
                biasDesc,
                resultDesc,
                dnnl::memory::dims{convAttrs.windowStrides[0],
                                   convAttrs.windowStrides[1],
                                   convAttrs.windowStrides[2]},
                dnnl::memory::dims{convAttrs.windowDilation[0] - 1,
                                   convAttrs.windowDilation[1] - 1,
                                   convAttrs.windowDilation[2] - 1},
                dnnl::memory::dims{
                    convAttrs.padBelow[0], convAttrs.padBelow[1], convAttrs.padBelow[2]},
                dnnl::memory::dims{
                    convAttrs.padAbove[0], convAttrs.padAbove[1], convAttrs.padAbove[2]});
            if (convAttrs.withRelu)
            {
                attr.set_post_ops(ops);
            }
            convPd = dnnl::convolution_forward::primitive_desc(convDesc, attr, cpuEngine);
        }
        catch (const dnnl::error& e)
        {
            throw ngraph_error("Could not create dnnl conv descriptor " + std::string(e.message));
        }
    }

    dnnl::convolution_forward conv(convPd);
    dnnl::memory data =
        convertLayoutIfDiff(dataDescOrigin, convPd.src_desc(), memRefData->allocatedPtr, cpuEngine);
    dnnl::memory weights = convertLayoutIfDiff(
        weightsDescOrigin, convPd.weights_desc(), memRefWeights->allocatedPtr, cpuEngine);
    dnnl::memory bias{convPd.bias_desc(), cpuEngine, memRefBias->allocatedPtr};
    dnnl::memory out;
    bool needConvert = false;
    if (!compareMkldnnMdFormats(resultDescOrigin, convPd.dst_desc()))
    {
        out = dnnl::memory(convPd.dst_desc(), cpuEngine);
        needConvert = true;
    }
    else
    {
        out = dnnl::memory(convPd.dst_desc(), cpuEngine, memRefOutput->allocatedPtr);
    }

    std::unordered_map<int, dnnl::memory> execArgs = {{DNNL_ARG_SRC, data},
                                                      {DNNL_ARG_WEIGHTS, weights},
                                                      {DNNL_ARG_BIAS, bias},
                                                      {DNNL_ARG_DST, out}};

    dnnl::stream s(cpuEngine);
    try
    {
        conv.execute(s, execArgs);
        s.wait();
    }
    catch (const dnnl::error& e)
    {
        throw ngraph_error("Could not run mkdnn primitive " + std::string(e.message));
    }

    if (needConvert)
    {
        convertOutputLayout(out, resultDescOrigin, memRefOutput->allocatedPtr, cpuEngine);
    }
}

/// Callback for MaxPoolBackprop
static void __mlir_dnnl_maxpoolbackprop(size_t rank,
                                        StaticMemRef* memRefSrc,
                                        StaticMemRef* memRefDelta,
                                        StaticMemRef* memRefOutput,
                                        opAttrs* attrsPtr)
{
    dnnl::memory::dims srcDims(rank);
    dnnl::memory::dims srcStrides(rank);
    dnnl::memory::dims deltaDims(rank);
    dnnl::memory::dims deltaStrides(rank);
    dnnl::memory::dims outDims(rank);
    dnnl::memory::dims outStrides(rank);
    for (auto i = 0; i < rank; i++)
    {
        srcDims[i] = memRefSrc->shapeAndStrides[i];
        srcStrides[i] = memRefSrc->shapeAndStrides[rank + i];
        deltaDims[i] = memRefDelta->shapeAndStrides[i];
        deltaStrides[i] = memRefDelta->shapeAndStrides[rank + i];
        outDims[i] = memRefOutput->shapeAndStrides[i];
        outStrides[i] = memRefOutput->shapeAndStrides[rank + i];
    }

    // build dnnl primitive and execute
    auto requiredFormat = rank == 4 ? dnnl::memory::FORMAT::nchw : dnnl::memory::FORMAT::ncdhw;
    dnnl::memory::data_type dtype = dnnl::memory::data_type::f32;
    auto diffDstDesc = dnnl::memory::desc(deltaDims, dtype, requiredFormat);
    auto diffSrcDesc = dnnl::memory::desc(outDims, dtype, requiredFormat);
    auto srcDescOrigin = dnnl::memory::desc(srcDims, dtype, srcStrides);
    auto diffDstDescOrigin = dnnl::memory::desc(deltaDims, dtype, deltaStrides);
    auto diffSrcDescOrigin = dnnl::memory::desc(outDims, dtype, outStrides);

    dnnl::primitive_attr attr;
    dnnl::engine cpuEngine(dnnl::engine::kind::cpu, 0);
    dnnl::pooling_forward::primitive_desc maxpoolPdF;
    dnnl::pooling_backward::primitive_desc maxpoolPdB;
    if (rank == 4)
    {
        poolAttrs<2> pAttrs = (*attrsPtr).poolAttrs2d;
        try
        {
            auto maxpoolDescF = dnnl::pooling_forward::desc(
                dnnl::prop_kind::forward_training,
                dnnl::algorithm::pooling_max,
                diffSrcDesc,
                diffDstDesc,
                dnnl::memory::dims{pAttrs.windowStrides[0], pAttrs.windowStrides[1]},
                dnnl::memory::dims{pAttrs.windowShape[0], pAttrs.windowShape[1]},
                dnnl::memory::dims{pAttrs.padBelow[0], pAttrs.padBelow[1]},
                dnnl::memory::dims{pAttrs.padAbove[0], pAttrs.padAbove[1]});
            auto maxpoolDescB = dnnl::pooling_backward::desc(
                dnnl::algorithm::pooling_max,
                diffSrcDesc,
                diffDstDesc,
                dnnl::memory::dims{pAttrs.windowStrides[0], pAttrs.windowStrides[1]},
                dnnl::memory::dims{pAttrs.windowShape[0], pAttrs.windowShape[1]},
                dnnl::memory::dims{pAttrs.padBelow[0], pAttrs.padBelow[1]},
                dnnl::memory::dims{pAttrs.padAbove[0], pAttrs.padAbove[1]});
            maxpoolPdF = dnnl::pooling_forward::primitive_desc(maxpoolDescF, attr, cpuEngine);
            maxpoolPdB =
                dnnl::pooling_backward::primitive_desc(maxpoolDescB, attr, cpuEngine, maxpoolPdF);
        }
        catch (const dnnl::error& e)
        {
            throw ngraph_error("Could not create dnnl max pooling descriptor " +
                               std::string(e.message));
        }
    }
    else if (rank == 5)
    {
        poolAttrs<3> pAttrs = (*attrsPtr).poolAttrs3d;
        try
        {
            auto maxpoolDescF = dnnl::pooling_forward::desc(
                dnnl::prop_kind::forward_training,
                dnnl::algorithm::pooling_max,
                diffSrcDesc,
                diffDstDesc,
                dnnl::memory::dims{
                    pAttrs.windowStrides[0], pAttrs.windowStrides[1], pAttrs.windowStrides[2]},
                dnnl::memory::dims{
                    pAttrs.windowShape[0], pAttrs.windowShape[1], pAttrs.windowShape[2]},
                dnnl::memory::dims{pAttrs.padBelow[0], pAttrs.padBelow[1], pAttrs.padBelow[2]},
                dnnl::memory::dims{pAttrs.padAbove[0], pAttrs.padAbove[1], pAttrs.padAbove[2]});
            auto maxpoolDescB = dnnl::pooling_backward::desc(
                dnnl::algorithm::pooling_max,
                diffSrcDesc,
                diffDstDesc,
                dnnl::memory::dims{
                    pAttrs.windowStrides[0], pAttrs.windowStrides[1], pAttrs.windowStrides[2]},
                dnnl::memory::dims{
                    pAttrs.windowShape[0], pAttrs.windowShape[1], pAttrs.windowShape[2]},
                dnnl::memory::dims{pAttrs.padBelow[0], pAttrs.padBelow[1], pAttrs.padBelow[2]},
                dnnl::memory::dims{pAttrs.padAbove[0], pAttrs.padAbove[1], pAttrs.padAbove[2]});
            maxpoolPdF = dnnl::pooling_forward::primitive_desc(maxpoolDescF, attr, cpuEngine);
            maxpoolPdB =
                dnnl::pooling_backward::primitive_desc(maxpoolDescB, attr, cpuEngine, maxpoolPdF);
        }
        catch (const dnnl::error& e)
        {
            throw ngraph_error("Could not create dnnl max pooling descriptor " +
                               std::string(e.message));
        }
    }

    dnnl::pooling_forward maxpoolF(maxpoolPdF);
    dnnl::memory srcMem = convertLayoutIfDiff(
        srcDescOrigin, maxpoolPdB.diff_src_desc(), memRefSrc->allocatedPtr, cpuEngine);
    dnnl::memory dstMem{maxpoolPdB.diff_dst_desc(), cpuEngine};
    dnnl::memory workspace{maxpoolPdF.workspace_desc(), cpuEngine};

    dnnl::pooling_backward maxpoolB(maxpoolPdB);
    dnnl::memory diffDst = convertLayoutIfDiff(
        diffDstDescOrigin, maxpoolPdB.diff_dst_desc(), memRefDelta->allocatedPtr, cpuEngine);
    dnnl::memory diffSrc;
    bool needConvert = false;
    if (!compareMkldnnMdFormats(diffSrcDescOrigin, maxpoolPdB.diff_src_desc()))
    {
        diffSrc = dnnl::memory(maxpoolPdB.diff_src_desc(), cpuEngine);
        needConvert = true;
    }
    else
    {
        diffSrc = dnnl::memory(maxpoolPdB.diff_src_desc(), cpuEngine, memRefOutput->allocatedPtr);
    }

    std::unordered_map<int, dnnl::memory> execArgsF = {
        {DNNL_ARG_SRC, srcMem}, {DNNL_ARG_WORKSPACE, workspace}, {DNNL_ARG_DST, dstMem}};
    std::unordered_map<int, dnnl::memory> execArgsB = {{DNNL_ARG_DIFF_DST, diffDst},
                                                       {DNNL_ARG_WORKSPACE, workspace},
                                                       {DNNL_ARG_DIFF_SRC, diffSrc}};

    dnnl::stream s(cpuEngine);
    try
    {
        maxpoolF.execute(s, execArgsF);
        s.wait();
        maxpoolB.execute(s, execArgsB);
        s.wait();
    }
    catch (const dnnl::error& e)
    {
        throw ngraph_error("Could not run mkdnn primitive " + std::string(e.message));
    }

    if (needConvert)
    {
        convertOutputLayout(diffSrc, diffSrcDescOrigin, memRefOutput->allocatedPtr, cpuEngine);
    }
}

/// Callback for AvgPoolBackprop
static void __mlir_dnnl_avgpoolbackprop(size_t rank,
                                        StaticMemRef* memRefInput,
                                        StaticMemRef* memRefOutput,
                                        opAttrs* attrsPtr)
{
    dnnl::memory::dims dims(rank);
    dnnl::memory::dims strides(rank);
    dnnl::memory::dims outDims(rank);
    dnnl::memory::dims outStrides(rank);
    for (auto i = 0; i < rank; i++)
    {
        dims[i] = memRefInput->shapeAndStrides[i];
        strides[i] = memRefInput->shapeAndStrides[rank + i];
        outDims[i] = memRefOutput->shapeAndStrides[i];
        outStrides[i] = memRefOutput->shapeAndStrides[rank + i];
    }

    // build dnnl primitive and execute
    auto requiredFormat = rank == 4 ? dnnl::memory::FORMAT::nchw : dnnl::memory::FORMAT::ncdhw;
    dnnl::memory::data_type dtype = dnnl::memory::data_type::f32;
    auto diffDstDesc = dnnl::memory::desc(dims, dtype, requiredFormat);
    auto diffSrcDesc = dnnl::memory::desc(outDims, dtype, requiredFormat);
    auto diffDstDescOrigin = dnnl::memory::desc(dims, dtype, strides);
    auto diffSrcDescOrigin = dnnl::memory::desc(outDims, dtype, outStrides);
    dnnl::primitive_attr attr;
    dnnl::engine cpuEngine(dnnl::engine::kind::cpu, 0);
    dnnl::pooling_backward::primitive_desc avgpoolPdB;
    if (rank == 4)
    {
        poolAttrs<2> pAttrs = (*attrsPtr).poolAttrs2d;
        try
        {
            auto avgpoolDescF = dnnl::pooling_forward::desc(
                dnnl::prop_kind::forward_training,
                (pAttrs.includePaddingInAvgComputation
                     ? dnnl::algorithm::pooling_avg_include_padding
                     : dnnl::algorithm::pooling_avg_exclude_padding),
                diffSrcDesc,
                diffDstDesc,
                dnnl::memory::dims{pAttrs.windowStrides[0], pAttrs.windowStrides[1]},
                dnnl::memory::dims{pAttrs.windowShape[0], pAttrs.windowShape[1]},
                dnnl::memory::dims{pAttrs.padBelow[0], pAttrs.padBelow[1]},
                dnnl::memory::dims{pAttrs.padAbove[0], pAttrs.padAbove[1]});
            auto avgpoolDescB = dnnl::pooling_backward::desc(
                (pAttrs.includePaddingInAvgComputation
                     ? dnnl::algorithm::pooling_avg_include_padding
                     : dnnl::algorithm::pooling_avg_exclude_padding),
                diffSrcDesc,
                diffDstDesc,
                dnnl::memory::dims{pAttrs.windowStrides[0], pAttrs.windowStrides[1]},
                dnnl::memory::dims{pAttrs.windowShape[0], pAttrs.windowShape[1]},
                dnnl::memory::dims{pAttrs.padBelow[0], pAttrs.padBelow[1]},
                dnnl::memory::dims{pAttrs.padAbove[0], pAttrs.padAbove[1]});
            auto avgpoolPdF = dnnl::pooling_forward::primitive_desc(avgpoolDescF, attr, cpuEngine);
            avgpoolPdB =
                dnnl::pooling_backward::primitive_desc(avgpoolDescB, attr, cpuEngine, avgpoolPdF);
        }
        catch (const dnnl::error& e)
        {
            throw ngraph_error("Could not create dnnl avg pooling descriptor " +
                               std::string(e.message));
        }
    }
    else if (rank == 5)
    {
        poolAttrs<3> pAttrs = (*attrsPtr).poolAttrs3d;
        try
        {
            auto avgpoolDescF = dnnl::pooling_forward::desc(
                dnnl::prop_kind::forward_training,
                (pAttrs.includePaddingInAvgComputation
                     ? dnnl::algorithm::pooling_avg_include_padding
                     : dnnl::algorithm::pooling_avg_exclude_padding),
                diffSrcDesc,
                diffDstDesc,
                dnnl::memory::dims{
                    pAttrs.windowStrides[0], pAttrs.windowStrides[1], pAttrs.windowStrides[2]},
                dnnl::memory::dims{
                    pAttrs.windowShape[0], pAttrs.windowShape[1], pAttrs.windowShape[2]},
                dnnl::memory::dims{pAttrs.padBelow[0], pAttrs.padBelow[1], pAttrs.padBelow[2]},
                dnnl::memory::dims{pAttrs.padAbove[0], pAttrs.padAbove[1], pAttrs.padAbove[2]});
            auto avgpoolDescB = dnnl::pooling_backward::desc(
                (pAttrs.includePaddingInAvgComputation
                     ? dnnl::algorithm::pooling_avg_include_padding
                     : dnnl::algorithm::pooling_avg_exclude_padding),
                diffSrcDesc,
                diffDstDesc,
                dnnl::memory::dims{
                    pAttrs.windowStrides[0], pAttrs.windowStrides[1], pAttrs.windowStrides[2]},
                dnnl::memory::dims{
                    pAttrs.windowShape[0], pAttrs.windowShape[1], pAttrs.windowShape[2]},
                dnnl::memory::dims{pAttrs.padBelow[0], pAttrs.padBelow[1], pAttrs.padBelow[2]},
                dnnl::memory::dims{pAttrs.padAbove[0], pAttrs.padAbove[1], pAttrs.padAbove[2]});
            auto avgpoolPdF = dnnl::pooling_forward::primitive_desc(avgpoolDescF, attr, cpuEngine);
            avgpoolPdB =
                dnnl::pooling_backward::primitive_desc(avgpoolDescB, attr, cpuEngine, avgpoolPdF);
        }
        catch (const dnnl::error& e)
        {
            throw ngraph_error("Could not create dnnl avg pooling descriptor " +
                               std::string(e.message));
        }
    }

    dnnl::pooling_backward avgpool(avgpoolPdB);
    dnnl::memory in = convertLayoutIfDiff(
        diffDstDescOrigin, avgpoolPdB.diff_dst_desc(), memRefInput->allocatedPtr, cpuEngine);
    dnnl::memory out;
    bool needConvert = false;
    if (!compareMkldnnMdFormats(diffSrcDescOrigin, avgpoolPdB.diff_src_desc()))
    {
        out = dnnl::memory(avgpoolPdB.diff_src_desc(), cpuEngine);
        needConvert = true;
    }
    else
    {
        out = dnnl::memory(avgpoolPdB.diff_src_desc(), cpuEngine, memRefOutput->allocatedPtr);
    }
    std::unordered_map<int, dnnl::memory> execArgs = {{DNNL_ARG_DIFF_DST, in},
                                                      {DNNL_ARG_DIFF_SRC, out}};

    dnnl::stream s(cpuEngine);
    try
    {
        avgpool.execute(s, execArgs);
        s.wait();
    }
    catch (const dnnl::error& e)
    {
        throw ngraph_error("Could not run mkdnn primitive " + std::string(e.message));
    }

    if (needConvert)
    {
        convertOutputLayout(out, diffSrcDescOrigin, memRefOutput->allocatedPtr, cpuEngine);
    }
}

/// Callback for AvgPool and MaxPool
static void __mlir_dnnl_pooling(size_t rank,
                                StaticMemRef* memRefInput,
                                StaticMemRef* memRefOutput,
                                opAttrs* attrsPtr,
                                OpType type)
{
    dnnl::memory::dims dims(rank);
    dnnl::memory::dims strides(rank);
    dnnl::memory::dims outDims(rank);
    dnnl::memory::dims outStrides(rank);
    for (auto i = 0; i < rank; i++)
    {
        dims[i] = memRefInput->shapeAndStrides[i];
        strides[i] = memRefInput->shapeAndStrides[rank + i];
        outDims[i] = memRefOutput->shapeAndStrides[i];
        outStrides[i] = memRefOutput->shapeAndStrides[rank + i];
    }

    // build dnnl primitive and execute
    auto requiredFormat = rank == 4 ? dnnl::memory::FORMAT::nchw : dnnl::memory::FORMAT::ncdhw;
    dnnl::memory::data_type dtype = dnnl::memory::data_type::f32;
    auto inputDesc = dnnl::memory::desc(dims, dtype, requiredFormat);
    auto resultDesc = dnnl::memory::desc(outDims, dtype, requiredFormat);
    auto inputDescOrigin = dnnl::memory::desc(dims, dtype, strides);
    auto resultDescOrigin = dnnl::memory::desc(outDims, dtype, outStrides);
    dnnl::primitive_attr attr;
    dnnl::engine cpuEngine(dnnl::engine::kind::cpu, 0);
    dnnl::pooling_forward::primitive_desc poolPd;
    if (rank == 4)
    {
        poolAttrs<2> pAttrs = (*attrsPtr).poolAttrs2d;
        dnnl::algorithm alg = type == OpType::MAXPOOL
                                  ? dnnl::algorithm::pooling_max
                                  : (pAttrs.includePaddingInAvgComputation
                                         ? dnnl::algorithm::pooling_avg_include_padding
                                         : dnnl::algorithm::pooling_avg_exclude_padding);
        try
        {
            auto poolDesc = dnnl::pooling_forward::desc(
                dnnl::prop_kind::forward_inference,
                alg,
                inputDesc,
                resultDesc,
                dnnl::memory::dims{pAttrs.windowStrides[0], pAttrs.windowStrides[1]},
                dnnl::memory::dims{pAttrs.windowShape[0], pAttrs.windowShape[1]},
                dnnl::memory::dims{pAttrs.padBelow[0], pAttrs.padBelow[1]},
                dnnl::memory::dims{pAttrs.padAbove[0], pAttrs.padAbove[1]});
            poolPd = dnnl::pooling_forward::primitive_desc(poolDesc, attr, cpuEngine);
        }
        catch (const dnnl::error& e)
        {
            throw ngraph_error("Could not create dnnl pooling descriptor " +
                               std::string(e.message));
        }
    }
    else if (rank == 5)
    {
        poolAttrs<3> pAttrs = (*attrsPtr).poolAttrs3d;
        dnnl::algorithm alg = type == OpType::MAXPOOL
                                  ? dnnl::algorithm::pooling_max
                                  : (pAttrs.includePaddingInAvgComputation
                                         ? dnnl::algorithm::pooling_avg_include_padding
                                         : dnnl::algorithm::pooling_avg_exclude_padding);
        try
        {
            auto poolDesc = dnnl::pooling_forward::desc(
                dnnl::prop_kind::forward_inference,
                alg,
                inputDesc,
                resultDesc,
                dnnl::memory::dims{
                    pAttrs.windowStrides[0], pAttrs.windowStrides[1], pAttrs.windowStrides[2]},
                dnnl::memory::dims{
                    pAttrs.windowShape[0], pAttrs.windowShape[1], pAttrs.windowShape[2]},
                dnnl::memory::dims{pAttrs.padBelow[0], pAttrs.padBelow[1], pAttrs.padBelow[2]},
                dnnl::memory::dims{pAttrs.padAbove[0], pAttrs.padAbove[1], pAttrs.padAbove[2]});
            poolPd = dnnl::pooling_forward::primitive_desc(poolDesc, attr, cpuEngine);
        }
        catch (const dnnl::error& e)
        {
            throw ngraph_error("Could not create dnnl pooing descriptor " + std::string(e.message));
        }
    }

    dnnl::pooling_forward pool(poolPd);
    dnnl::memory in = convertLayoutIfDiff(
        inputDescOrigin, poolPd.src_desc(), memRefInput->allocatedPtr, cpuEngine);
    dnnl::memory out;
    bool needConvert = false;
    if (!compareMkldnnMdFormats(resultDescOrigin, poolPd.dst_desc()))
    {
        out = dnnl::memory(poolPd.dst_desc(), cpuEngine);
        needConvert = true;
    }
    else
    {
        out = dnnl::memory(poolPd.dst_desc(), cpuEngine, memRefOutput->allocatedPtr);
    }
    std::unordered_map<int, dnnl::memory> execArgs = {{DNNL_ARG_SRC, in}, {DNNL_ARG_DST, out}};

    dnnl::stream s(cpuEngine);
    try
    {
        pool.execute(s, execArgs);
        s.wait();
    }
    catch (const dnnl::error& e)
    {
        throw ngraph_error("Could not run mkdnn primitive " + std::string(e.message));
    }

    if (needConvert)
    {
        convertOutputLayout(out, resultDescOrigin, memRefOutput->allocatedPtr, cpuEngine);
    }
}

/// Callback for Softmax
static void __mlir_dnnl_softmax(size_t rank,
                                StaticMemRef* memRefInput,
                                StaticMemRef* memRefOutput,
                                opAttrs* attrsPtr)
{
    dnnl::memory::dims dims(rank);
    dnnl::memory::dims strides(rank);
    for (auto i = 0; i < rank; i++)
    {
        dims[i] = memRefInput->shapeAndStrides[i];
        strides[i] = memRefInput->shapeAndStrides[rank + i];
    }
    auto softmaxAxis = (*attrsPtr).intAttr;

    // build dnnl primitive and execute
    dnnl::memory::data_type dtype = dnnl::memory::data_type::f32;
    auto inputDesc = dnnl::memory::desc(dims, dtype, strides);
    dnnl::softmax_forward::primitive_desc softmaxPd;
    dnnl::engine cpuEngine(dnnl::engine::kind::cpu, 0);
    try
    {
        auto softmaxDesc =
            dnnl::softmax_forward::desc(dnnl::prop_kind::forward_scoring, inputDesc, softmaxAxis);
        dnnl::primitive_attr attr;
        softmaxPd = dnnl::softmax_forward::primitive_desc(softmaxDesc, attr, cpuEngine);
    }
    catch (const dnnl::error& e)
    {
        throw ngraph_error("Could not create dnnl softmax descriptor " + std::string(e.message));
    }
    dnnl::softmax_forward softmax(softmaxPd);

    dnnl::memory in{softmaxPd.src_desc(), cpuEngine, memRefInput->allocatedPtr};
    dnnl::memory out{softmaxPd.dst_desc(), cpuEngine, memRefOutput->allocatedPtr};

    std::unordered_map<int, dnnl::memory> execArgs = {{DNNL_ARG_SRC, in}, {DNNL_ARG_DST, out}};

    dnnl::stream s(cpuEngine);
    try
    {
        softmax.execute(s, execArgs);
        s.wait();
    }
    catch (const dnnl::error& e)
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
    dnnl_sgemm(
                       gAttrs.transposeA ? 't' : 'n',
                       gAttrs.transposeB ? 't' : 'n',
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

    dnnl_sgemm(
                       transposeA ? 't' : 'n',
                       transposeB ? 't' : 'n',
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
        dnnl_sgemm(
                           'n',
                           'n',
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
        dnnl_sgemm(
                           'n',
                           'n',
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
        dnnl_sgemm(
                           'n',
                           'n',
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
        dnnl_sgemm(
                           'n',
                           'n',
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
        __mlir_dnnl_softmax(unrankedMemRefInput->rank,
                            unrankedMemRefInput->memRefDescPtr,
                            unrankedMemRefOutput->memRefDescPtr,
                            static_cast<opAttrs*>(attrsPtr));
    }
    else if (type == OpType::AVGPOOL || type == OpType::MAXPOOL)
    {
        __mlir_dnnl_pooling(unrankedMemRefInput->rank,
                            unrankedMemRefInput->memRefDescPtr,
                            unrankedMemRefOutput->memRefDescPtr,
                            static_cast<opAttrs*>(attrsPtr),
                            type);
    }
    else if (type == OpType::AVGPOOLBACKPROP)
    {
        __mlir_dnnl_avgpoolbackprop(unrankedMemRefInput->rank,
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
        __mlir_dnnl_maxpoolbackprop(unrankedMemRefInput0->rank,
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
        __mlir_dnnl_convbias(unrankedMemRefInput0->rank,
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
