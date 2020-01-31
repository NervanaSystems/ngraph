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

#pragma once

#include <cstddef>

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            struct CPURuntimeContext;

            namespace mkldnn_utils
            {
                enum class OpType
                {
                    ADD,
                    AVGPOOL,
                    AVGPOOLBACKPROP,
                    BATCHNORM3ARGS,
                    BATCHNORM5ARGS,
                    BATCHNORMBACKPROP,
                    BOUNDEDRELU,
                    CONCAT,
                    CONVERTLAYOUT,
                    CONVOLUTION,
                    CONVOLUTIONRELU,
                    CONVOLUTIONADD,
                    CONVOLUTIONBIAS,
                    CONVOLUTIONBIASADD,
                    CONVOLUTIONBACKPROPDATA,
                    CONVOLUTIONBACKPROPWEIGHTS,
                    CONVOLUTIONBACKPROPWEIGHTSBIAS,
                    GELU,
                    GELUBACKPROP,
                    GROUPCONVOLUTION,
                    GROUPCONVOLUTIONBIAS,
                    DECONVOLUTIONBIAS,
                    LEAKYRELU,
                    LRN,
                    LSTM,
                    MAXPOOL,
                    MAXPOOLBACKPROPFORWARD,
                    MAXPOOLBACKPROPBACKWARD,
                    MAXPOOLWITHINDICES,
                    MAXPOOLWITHINDICESBACKPROP,
                    QUANTIZE,
                    DEQUANTIZE,
                    QUANTIZEDAVGPOOL,
                    QUANTIZEDMAXPOOL,
                    QUANTIZEDCONCAT,
                    QUANTIZEDDOTBIAS,
                    QUANTIZEDMATMUL,
                    QUANTIZEDCONVOLUTION,
                    QUANTIZEDCONVOLUTIONBIAS,
                    QUANTIZEDCONVOLUTIONBIASADD,
                    QUANTIZEDCONVOLUTIONBIASSIGNEDADD,
                    QUANTIZEDCONVOLUTIONRELU,
                    RELU,
                    RELUBACKPROP,
                    RNN,
                    SIGMOID,
                    SIGMOIDBACKPROP,
                    SLICE,
                    SOFTMAX
                };
                extern "C" void set_memory_ptr(CPURuntimeContext* ctx, size_t index, void* ptr);
                extern "C" void mkldnn_invoke_primitive(CPURuntimeContext* ctx,
                                                        size_t primitive_index,
                                                        std::vector<size_t>& deps,
                                                        OpType type,
                                                        size_t scratchpad_size = 0);
            }
        }
    }
}
