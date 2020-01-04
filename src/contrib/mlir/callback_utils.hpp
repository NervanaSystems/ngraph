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

namespace ngraph
{
    namespace runtime
    {
        namespace ngmlir
        {
            enum class OpType
            {
                ADD = 0,
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
                GEMM,
                GROUPCONVOLUTION,
                GROUPCONVOLUTIONBIAS,
                DECONVOLUTIONBIAS,
                LEAKYRELU,
                LRN,
                LSTM,
                MATMUL,
                MAXPOOL,
                MAXPOOLBACKPROP,
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

            template <int N>
            struct poolAttrs
            {
                bool includePaddingInAvgComputation;
                int64_t windowShape[N];
                int64_t windowStrides[N];
                int64_t padBelow[N];
                int64_t padAbove[N];
            };

            struct gemmAttrs
            {
                bool transposeA;
                bool transposeB;
                int64_t m;
                int64_t n;
                int64_t k;
                int64_t lda;
                int64_t ldb;
                int64_t ldc;
                float alpha;
                float beta;
                int64_t broadcastHint;
            };
        } // namespace ngmlir
    }     // namespace runtime
} // namespace ngraph
