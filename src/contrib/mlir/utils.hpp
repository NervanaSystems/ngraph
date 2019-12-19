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

// NOTE: This file follows nGraph format style and MLIR naming convention since it does
// not expose public API to the rest of nGraph codebase and heavily depends on MLIR API.
#pragma once

#include <mlir/IR/Module.h>

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

            /// Common nGraph dialect initialization code. Used by nGraph compiler and tools that
            /// require nGraph dialect initialization.
            void initializeNGraphMLIR();

            /// Helper to dump MLIR module into llvm::dbgs prepended by the message \p msg.
            void dumpMlirModule(const std::string msg, mlir::ModuleOp module);
        } // namespace ngmlir
    }     // namespace runtime
} // namespace ngraph
