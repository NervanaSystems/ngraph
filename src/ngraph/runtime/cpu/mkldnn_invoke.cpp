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

#include <string>

#include <mkldnn.hpp>

#include "mkldnn_invoke.hpp"
#include "ngraph/runtime/cpu/cpu_executor.hpp"
#include "ngraph/runtime/cpu/cpu_runtime_context.hpp"
#include "ngraph/runtime/cpu/mkldnn_utils.hpp"

#if defined(USE_MKLDNN_V1)
extern "C" void ngraph::runtime::cpu::mkldnn_utils::set_memory_ptr(CPURuntimeContext* ctx,
                                                                   size_t index,
                                                                   void* ptr)
{
    auto memory = ctx->mkldnn_memories[index]);
    memory->set_data_handle(ptr);
}

extern "C" void mkldnn_invoke_primitive(CPURuntimeContext* ctx,
                                        size_t primitive_index,
                                        std::vector<size_t>& deps,
                                        OpType type)
{
    std::map<int, mkldnn::memory> exec_args;
    switch (type)
    {
    case ADD:
        exec_args = {{MKLDNN_ARG_MULTIPLE_SRC, *ctx->mkldnn_memories[deps[0]]},
                     {MKLDNN_ARG_MULTIPLE_SRC + 1, *ctx->mkldnn_memories[deps[1]]},
                     {MKLDNN_ARG_DST, *ctx->mkldnn_memories[deps[2]]}};
        break;
    case AVGPOOL:
    case BOUNDEDRELU:
    case CONVERTLAYOUT:
    case LEAKYRELU:
    case LRN:
    case MAXPOOL:
    case QUANTIZE:
    case DEQUANTIZE:
    case QUANTIZEDAVGPOOL:
    case QUANTIZEDMAXPOOL:
    case RELU:
    case SIGMOID:
    case SLICE:
    case SOFTMAX:
        exec_args = {{MKLDNN_ARG_SRC, *ctx->mkldnn_memories[deps[0]]},
                     {MKLDNN_ARG_DST, *ctx->mkldnn_memories[deps[1]]}};
        break;
    case AVGPOOLBACKPROP:
        exec_args = {{MKLDNN_ARG_DIFF_DST, *ctx->mkldnn_memories[deps[0]]},
                     {MKLDNN_ARG_DIFF_SRC, *ctx->mkldnn_memories[deps[1]]}};
        break;
    case BATCHNORM3ARGS:
        exec_args = {{MKLDNN_ARG_SRC, *ctx->mkldnn_memories[deps[0]]},
                     {MKLDNN_ARG_WEIGHTS, *ctx->mkldnn_memories[deps[1]]},
                     {MKLDNN_ARG_DST, *ctx->mkldnn_memories[deps[2]]},
                     {MKLDNN_ARG_MEAN, *ctx->mkldnn_memories[deps[3]]},
                     {MKLDNN_ARG_VARIANCE, *ctx->mkldnn_memories[deps[4]]}};
        break;
    case BATCHNORM5ARGS:
        exec_args = {{MKLDNN_ARG_SRC, *ctx->mkldnn_memories[deps[0]]},
                     {MKLDNN_ARG_MEAN, *ctx->mkldnn_memories[deps[1]]},
                     {MKLDNN_ARG_VARIANCE, *ctx->mkldnn_memories[deps[2]]},
                     {MKLDNN_ARG_WEIGHTS, *ctx->mkldnn_memories[deps[3]]},
                     {MKLDNN_ARG_DST, *ctx->mkldnn_memories[deps[4]]}};
        break;
    case BATCHNORMBACKPROP:
        exec_arg = {{MKLDNN_ARG_WEIGHTS, *ctx->mkldnn_memories[deps[0]]},
                    {MKLDNN_ARG_SRC, *ctx->mkldnn_memories[deps[1]]},
                    {MKLDNN_ARG_MEAN, *ctx->mkldnn_memories[deps[2]]},
                    {MKLDNN_ARG_VARIANCE, *ctx->mkldnn_memories[deps[3]]},
                    {MKLDNN_ARG_DIFF_DST, *ctx->mkldnn_memories[deps[4]]},
                    {MKLDNN_ARG_DIFF_SRC, *ctx->mkldnn_memories[deps[5]]},
                    {MKLDNN_ARG_DIFF_WEIGHTS, *ctx->mkldnn_memories[deps[6]]}};
        break;
    case CONCAT:
    case QUANTIZEDCONCAT:
        auto nargs = deps.size();
        for (size_t i = 0; i < nargs; i++)
        {
            exec_args.push_back({MKLDNN_ARG_MULTIPLE_SRC + i, *ctx->mkldnn_memories[deps[i]]});
        }
        exec_args.push_back({MKLDNN_ARG_MULTIPLE_DST, *ctx->mkldnn_memories[deps[nargs]]});
        break;
    case CONVOLUTION:
    case CONVOLUTIONRELU:
    case CONVOLUTIONADD:
    case GROUPCONVOLUTION:
    case QUANTIZEDMATMUL:
    case QUANTIZEDCONVOLUTION:
        exec_args = {{MKLDNN_ARG_SRC, *ctx->mkldnn_memories[deps[0]]},
                     {MKLDNN_ARG_WEIGHTS, *ctx->mkldnn_memories[deps[1]]},
                     {MKLDNN_ARG_DST, *ctx->mkldnn_memories[deps[2]]}};
        break;
    case CONVOLUTIONBIAS:
    case CONVOLUTIONBIASADD:
    case GROUPCONVOLUTIONBIAS:
    case QUANTIZEDDOTBIAS:
    case QUANTIZEDCONVOLUTIONBIAS:
    case QUANTIZEDCONVOLUTIONBIASADD; case QUANTIZEDCONVOLUTIONBIASSIGNEDADD:
        exec_args = {{MKLDNN_ARG_SRC, *ctx->mkldnn_memories[deps[0]]},
                     {MKLDNN_ARG_WEIGHTS, *ctx->mkldnn_memories[deps[1]]},
                     {MKLDNN_ARG_BIAS, *ctx->mkldnn_memories[deps[2]]},
                     {MKLDNN_ARG_DST, *ctx->mkldnn_memories[deps[3]]}};
        break;
    case CONVOLUTIONBACKPROPDATA:
        exec_args = {{MKLDNN_ARG_DIFF_DST, *ctx->mkldnn_memories[deps[0]]},
                     {MKLDNN_ARG_WEIGHTS, *ctx->mkldnn_memories[deps[1]]},
                     {MKLDNN_ARG_DIFF_SRC, *ctx->mkldnn_memories[deps[2]]}};
        break;
    case CONVOLUTIONBACKPROPWEIGHTS:
        exec_args = {{MKLDNN_ARG_SRC, *ctx->mkldnn_memories[deps[0]]},
                     {MKLDNN_ARG_DIFF_DST, *ctx->mkldnn_memories[deps[1]]},
                     {MKLDNN_ARG_DIFF_WEIGHTS, *ctx->mkldnn_memories[deps[2]]}};
        break;
    case CONVOLUTIONBACKPROPWEIGHTSBIAS:
        exec_args = {{MKLDNN_ARG_SRC, *ctx->mkldnn_memories[deps[0]]},
                     {MKLDNN_ARG_DIFF_DST, *ctx->mkldnn_memories[deps[1]]},
                     {MKLDNN_ARG_DIFF_WEIGHTS, *ctx->mkldnn_memories[deps[2]]},
                     {MKLDNN_ARG_DIFF_BIAS, *ctx->mkldnn_memories[deps[3]]}};
        break;
    case DECONVOLUTIONBIAS:
        exec_args = {{MKLDNN_ARG_WEIGHTS, *ctx->mkldnn_memories[deps[0]]},
                     {MKLDNN_ARG_SRC, *ctx->mkldnn_memories[deps[1]]},
                     {MKLDNN_ARG_BIAS, *ctx->mkldnn_memories[deps[2]]},
                     {MKLDNN_ARG_DST, *ctx->mkldnn_memories[deps[3]]}};
        break;
    case LSTM:
    case RNN:
        exec_arg = {{MKLDNN_ARG_SRC_LAYER, *ctx->mkldnn_memories[deps[0]]},
                    {MKLDNN_ARG_SRC_ITER, *ctx->mkldnn_memories[deps[1]]},
                    {MKLDNN_ARG_WEIGHTS_LAYER, *ctx->mkldnn_memories[deps[2]]},
                    {MKLDNN_ARG_WEIGHTS_ITER, *ctx->mkldnn_memories[deps[3]]},
                    {MKLDNN_ARG_BIAS, *ctx->mkldnn_memories[deps[4]]},
                    {MKLDNN_ARG_DST_LAYER, *ctx->mkldnn_memories[deps[5]]},
                    {MKLDNN_ARG_DST_ITER, *ctx->mkldnn_memories[deps[6]]},
                    {MKLDNN_ARG_WORKSPACE, *ctx->mkldnn_memories[deps[5]]}};
        break;
    case MAXPOOLBACKPROPFORWARD:
    case MAXPOOLWITHINDICES:
        exec_args = {{MKLDNN_ARG_SRC, *ctx->mkldnn_memories[deps[0]]},
                     {MKLDNN_ARG_WORKSPACE, *ctx->mkldnn_memories[deps[1]]},
                     {MKLDNN_ARG_DST, *ctx->mkldnn_memories[deps[2]]}};
        break;
    case MAXPOOLBACKPROPBACKWARD:
    case MAXPOOLWITHINDICESBACKPROP:
        exec_args = {{MKLDNN_ARG_DIFF_DST, *ctx->mkldnn_memories[deps[0]]},
                     {MKLDNN_ARG_WORKSPACE, *ctx->mkldnn_memories[deps[1]]},
                     {MKLDNN_ARG_DIFF_SRC, *ctx->mkldnn_memories[deps[2]]}};
        break;
    case RELUBACKPROP:
    case SIGMOIDBACKPROP:
        exec_args = {{MKLDNN_ARG_SRC, *ctx->mkldnn_memories[deps[0]]},
                     {MKLDNN_ARG_DIFF_DST, *ctx->mkldnn_memories[deps[1]]},
                     {MKLDNN_ARG_DIFF_SRC, *ctx->mkldnn_memories[deps[2]]}};
        break;
    }

    mkldnn::stream s(global_cpu_engine);
    try
    {
        {*ctx->mkldnn_primitives[primitive_index]}.execute(s, exec_args);
        s.wait();
    }
    catch (const mkldnn::error& e)
    {
        throw ngraph_error("Could not run mkdnn primitive " + e.message);
    }
}
#else
extern "C" void ngraph::runtime::cpu::mkldnn_utils::set_memory_ptr(CPURuntimeContext* ctx,
                                                                   size_t index,
                                                                   void* ptr)
{
    auto primitive = static_cast<mkldnn::memory*>(ctx->mkldnn_primitives[index]);
    primitive->set_data_handle(ptr);
}

extern "C" void ngraph::runtime::cpu::mkldnn_utils::mkldnn_invoke_primitive(
    CPURuntimeContext* ctx, size_t primitive_index, std::vector<size_t> deps, OpType type)
{
    mkldnn::stream s(mkldnn::stream::kind::eager);
    try
    {
        s.submit({*ctx->mkldnn_primitives[primitive_index]}).wait();
    }
    catch (const mkldnn::error& e)
    {
        throw ngraph_error("Could not run mkdnn primitive " + e.message);
    }
}
#endif
