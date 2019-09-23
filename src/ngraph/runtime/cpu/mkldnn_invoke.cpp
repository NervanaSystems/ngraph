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
#include "ngraph/runtime/aligned_buffer.hpp"
#include "ngraph/runtime/cpu/cpu_executor.hpp"
#include "ngraph/runtime/cpu/cpu_runtime_context.hpp"
#include "ngraph/runtime/cpu/mkldnn_utils.hpp"

#if MKLDNN_VERSION_MAJOR < 1
extern "C" void ngraph::runtime::cpu::mkldnn_utils::set_memory_ptr(CPURuntimeContext* ctx,
                                                                   size_t index,
                                                                   void* ptr)
{
    auto primitive = static_cast<mkldnn::memory*>(ctx->mkldnn_primitives[index]);
    primitive->set_data_handle(ptr);
}

extern "C" void
    ngraph::runtime::cpu::mkldnn_utils::mkldnn_invoke_primitive(CPURuntimeContext* ctx,
                                                                size_t primitive_index,
                                                                std::vector<size_t>& /* deps */,
                                                                OpType /* type */)
{
    mkldnn::stream s(mkldnn::stream::kind::eager);
    try
    {
        s.submit({*ctx->mkldnn_primitives[primitive_index]}).wait();
    }
    catch (const mkldnn::error& e)
    {
        throw ngraph_error("Could not run mkdnn primitive " + MKLDNN_ERROR_MESSAGE);
    }
}
#else
extern "C" void ngraph::runtime::cpu::mkldnn_utils::set_memory_ptr(CPURuntimeContext* ctx,
                                                                   size_t index,
                                                                   void* ptr)
{
    auto memory = ctx->mkldnn_memories[index];
    memory->set_data_handle(ptr);
}

extern "C" void ngraph::runtime::cpu::mkldnn_utils::mkldnn_invoke_primitive(
    CPURuntimeContext* ctx, size_t primitive_index, std::vector<size_t>& deps, OpType type)
{
    std::unordered_map<int, mkldnn::memory> exec_args;
    size_t nargs;
    switch (type)
    {
    case OpType::ADD:
        exec_args = {{MKLDNN_ARG_MULTIPLE_SRC, *ctx->mkldnn_memories[deps[0]]},
                     {MKLDNN_ARG_MULTIPLE_SRC + 1, *ctx->mkldnn_memories[deps[1]]},
                     {MKLDNN_ARG_DST, *ctx->mkldnn_memories[deps[2]]}};
        break;
    case OpType::AVGPOOL:
    case OpType::BOUNDEDRELU:
    case OpType::CONVERTLAYOUT:
    case OpType::LEAKYRELU:
    case OpType::LRN:
    case OpType::MAXPOOL:
    case OpType::QUANTIZE:
    case OpType::DEQUANTIZE:
    case OpType::QUANTIZEDAVGPOOL:
    case OpType::QUANTIZEDMAXPOOL:
    case OpType::RELU:
    case OpType::SIGMOID:
    case OpType::SLICE:
    case OpType::SOFTMAX:
        exec_args = {{MKLDNN_ARG_SRC, *ctx->mkldnn_memories[deps[0]]},
                     {MKLDNN_ARG_DST, *ctx->mkldnn_memories[deps[1]]}};
        break;
    case OpType::AVGPOOLBACKPROP:
        exec_args = {{MKLDNN_ARG_DIFF_DST, *ctx->mkldnn_memories[deps[0]]},
                     {MKLDNN_ARG_DIFF_SRC, *ctx->mkldnn_memories[deps[1]]}};
        break;
    case OpType::BATCHNORM3ARGS:
        exec_args = {{MKLDNN_ARG_SRC, *ctx->mkldnn_memories[deps[0]]},
                     {MKLDNN_ARG_WEIGHTS, *ctx->mkldnn_memories[deps[1]]},
                     {MKLDNN_ARG_DST, *ctx->mkldnn_memories[deps[2]]},
                     {MKLDNN_ARG_MEAN, *ctx->mkldnn_memories[deps[3]]},
                     {MKLDNN_ARG_VARIANCE, *ctx->mkldnn_memories[deps[4]]}};
        break;
    case OpType::BATCHNORM5ARGS:
        exec_args = {{MKLDNN_ARG_SRC, *ctx->mkldnn_memories[deps[0]]},
                     {MKLDNN_ARG_MEAN, *ctx->mkldnn_memories[deps[1]]},
                     {MKLDNN_ARG_VARIANCE, *ctx->mkldnn_memories[deps[2]]},
                     {MKLDNN_ARG_WEIGHTS, *ctx->mkldnn_memories[deps[3]]},
                     {MKLDNN_ARG_DST, *ctx->mkldnn_memories[deps[4]]}};
        break;
    case OpType::BATCHNORMBACKPROP:
        exec_args = {{MKLDNN_ARG_WEIGHTS, *ctx->mkldnn_memories[deps[0]]},
                     {MKLDNN_ARG_SRC, *ctx->mkldnn_memories[deps[1]]},
                     {MKLDNN_ARG_MEAN, *ctx->mkldnn_memories[deps[2]]},
                     {MKLDNN_ARG_VARIANCE, *ctx->mkldnn_memories[deps[3]]},
                     {MKLDNN_ARG_DIFF_DST, *ctx->mkldnn_memories[deps[4]]},
                     {MKLDNN_ARG_DIFF_SRC, *ctx->mkldnn_memories[deps[5]]},
                     {MKLDNN_ARG_DIFF_WEIGHTS, *ctx->mkldnn_memories[deps[6]]}};
        break;
    case OpType::CONCAT:
    case OpType::QUANTIZEDCONCAT:
        nargs = deps.size() - 1;
        for (size_t i = 0; i < nargs; i++)
        {
            exec_args.insert({MKLDNN_ARG_MULTIPLE_SRC + i, *ctx->mkldnn_memories[deps[i]]});
        }
        exec_args.insert({MKLDNN_ARG_DST, *ctx->mkldnn_memories[deps[nargs]]});
        break;
    case OpType::CONVOLUTION:
    case OpType::CONVOLUTIONRELU:
    case OpType::CONVOLUTIONADD:
    case OpType::GROUPCONVOLUTION:
    case OpType::QUANTIZEDMATMUL:
    case OpType::QUANTIZEDCONVOLUTION:
    case OpType::QUANTIZEDCONVOLUTIONRELU:
        exec_args = {{MKLDNN_ARG_SRC, *ctx->mkldnn_memories[deps[0]]},
                     {MKLDNN_ARG_WEIGHTS, *ctx->mkldnn_memories[deps[1]]},
                     {MKLDNN_ARG_DST, *ctx->mkldnn_memories[deps[2]]}};
        break;
    case OpType::CONVOLUTIONBIAS:
    case OpType::CONVOLUTIONBIASADD:
    case OpType::GROUPCONVOLUTIONBIAS:
    case OpType::QUANTIZEDDOTBIAS:
    case OpType::QUANTIZEDCONVOLUTIONBIAS:
    case OpType::QUANTIZEDCONVOLUTIONBIASADD:
    case OpType::QUANTIZEDCONVOLUTIONBIASSIGNEDADD:
        exec_args = {{MKLDNN_ARG_SRC, *ctx->mkldnn_memories[deps[0]]},
                     {MKLDNN_ARG_WEIGHTS, *ctx->mkldnn_memories[deps[1]]},
                     {MKLDNN_ARG_BIAS, *ctx->mkldnn_memories[deps[2]]},
                     {MKLDNN_ARG_DST, *ctx->mkldnn_memories[deps[3]]}};
        break;
    case OpType::CONVOLUTIONBACKPROPDATA:
        exec_args = {{MKLDNN_ARG_DIFF_DST, *ctx->mkldnn_memories[deps[1]]},
                     {MKLDNN_ARG_WEIGHTS, *ctx->mkldnn_memories[deps[0]]},
                     {MKLDNN_ARG_DIFF_SRC, *ctx->mkldnn_memories[deps[2]]}};
        break;
    case OpType::CONVOLUTIONBACKPROPWEIGHTS:
        exec_args = {{MKLDNN_ARG_SRC, *ctx->mkldnn_memories[deps[0]]},
                     {MKLDNN_ARG_DIFF_DST, *ctx->mkldnn_memories[deps[1]]},
                     {MKLDNN_ARG_DIFF_WEIGHTS, *ctx->mkldnn_memories[deps[2]]}};
        break;
    case OpType::CONVOLUTIONBACKPROPWEIGHTSBIAS:
        exec_args = {{MKLDNN_ARG_SRC, *ctx->mkldnn_memories[deps[0]]},
                     {MKLDNN_ARG_DIFF_DST, *ctx->mkldnn_memories[deps[1]]},
                     {MKLDNN_ARG_DIFF_WEIGHTS, *ctx->mkldnn_memories[deps[2]]},
                     {MKLDNN_ARG_DIFF_BIAS, *ctx->mkldnn_memories[deps[3]]}};
        break;
    case OpType::DECONVOLUTIONBIAS:
        exec_args = {{MKLDNN_ARG_WEIGHTS, *ctx->mkldnn_memories[deps[0]]},
                     {MKLDNN_ARG_SRC, *ctx->mkldnn_memories[deps[1]]},
                     {MKLDNN_ARG_BIAS, *ctx->mkldnn_memories[deps[2]]},
                     {MKLDNN_ARG_DST, *ctx->mkldnn_memories[deps[3]]}};
        break;
    case OpType::LSTM:
    case OpType::RNN:
        exec_args = {{MKLDNN_ARG_SRC_LAYER, *ctx->mkldnn_memories[deps[0]]},
                     {MKLDNN_ARG_SRC_ITER, *ctx->mkldnn_memories[deps[1]]},
                     {MKLDNN_ARG_SRC_ITER_C, *ctx->mkldnn_memories[deps[2]]},
                     {MKLDNN_ARG_WEIGHTS_LAYER, *ctx->mkldnn_memories[deps[3]]},
                     {MKLDNN_ARG_WEIGHTS_ITER, *ctx->mkldnn_memories[deps[4]]},
                     {MKLDNN_ARG_BIAS, *ctx->mkldnn_memories[deps[5]]},
                     {MKLDNN_ARG_DST_LAYER, *ctx->mkldnn_memories[deps[6]]},
                     {MKLDNN_ARG_DST_ITER, *ctx->mkldnn_memories[deps[7]]},
                     {MKLDNN_ARG_DST_ITER_C, *ctx->mkldnn_memories[deps[8]]},
                     {MKLDNN_ARG_WORKSPACE, *ctx->mkldnn_memories[deps[9]]}};
        break;
    case OpType::MAXPOOLBACKPROPFORWARD:
    case OpType::MAXPOOLWITHINDICES:
        exec_args = {{MKLDNN_ARG_SRC, *ctx->mkldnn_memories[deps[0]]},
                     {MKLDNN_ARG_WORKSPACE, *ctx->mkldnn_memories[deps[2]]},
                     {MKLDNN_ARG_DST, *ctx->mkldnn_memories[deps[1]]}};
        break;
    case OpType::MAXPOOLBACKPROPBACKWARD:
    case OpType::MAXPOOLWITHINDICESBACKPROP:
        exec_args = {{MKLDNN_ARG_DIFF_DST, *ctx->mkldnn_memories[deps[0]]},
                     {MKLDNN_ARG_WORKSPACE, *ctx->mkldnn_memories[deps[1]]},
                     {MKLDNN_ARG_DIFF_SRC, *ctx->mkldnn_memories[deps[2]]}};
        break;
    case OpType::RELUBACKPROP:
    case OpType::SIGMOIDBACKPROP:
        exec_args = {{MKLDNN_ARG_SRC, *ctx->mkldnn_memories[deps[0]]},
                     {MKLDNN_ARG_DIFF_DST, *ctx->mkldnn_memories[deps[1]]},
                     {MKLDNN_ARG_DIFF_SRC, *ctx->mkldnn_memories[deps[2]]}};
        break;
    }

    mkldnn::memory scratchpad(*ctx->mkldnn_scratchpad_mds[primitive_index],
                              executor::global_cpu_engine,
                              ctx->scratchpad_buffer->get_ptr());
    exec_args.insert({MKLDNN_ARG_SCRATCHPAD, scratchpad});

    mkldnn::stream s(executor::global_cpu_engine);
    try
    {
        (*ctx->mkldnn_primitives[primitive_index]).execute(s, exec_args);
        s.wait();
    }
    catch (const mkldnn::error& e)
    {
        throw ngraph_error("Could not run mkdnn primitive " + MKLDNN_ERROR_MESSAGE);
    }
}
#endif
