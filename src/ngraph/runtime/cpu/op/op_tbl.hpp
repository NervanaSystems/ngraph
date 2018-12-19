//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
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

#include "ngraph/op/op_tbl.hpp"

NGRAPH_OP(BatchDot, ngraph::op)
NGRAPH_OP(BatchNormInferenceRelu, ngraph::op)
NGRAPH_OP(BatchNormTrainingRelu, ngraph::op)
NGRAPH_OP(BoundedRelu, ngraph::op)
NGRAPH_OP(ConvertLayout, ngraph::runtime::cpu::op)
NGRAPH_OP(ConvolutionAdd, ngraph::op)
NGRAPH_OP(ConvolutionBiasAdd, ngraph::op)
NGRAPH_OP(ConvolutionBiasBackpropFiltersBias, ngraph::op)
NGRAPH_OP(ConvolutionBias, ngraph::op)
NGRAPH_OP(ConvolutionRelu, ngraph::op)
NGRAPH_OP(GroupConvolutionBias, ngraph::op)
NGRAPH_OP(GroupConvolution, ngraph::op)
NGRAPH_OP(LeakyRelu, ngraph::op)
NGRAPH_OP(LoopKernel, ngraph::runtime::cpu::op)
NGRAPH_OP(Lstm, ngraph::op)
NGRAPH_OP(MatmulBias, ngraph::op)
NGRAPH_OP(MaxPoolWithIndicesBackprop, ngraph::op)
NGRAPH_OP(MaxPoolWithIndices, ngraph::op)
NGRAPH_OP(QuantizedConvolutionBiasAdd, ngraph::op)
NGRAPH_OP(QuantizedConvolutionBias, ngraph::op)
NGRAPH_OP(QuantizedConvolutionBiasSignedAdd, ngraph::op)
NGRAPH_OP(QuantizedConvolutionRelu, ngraph::op)
NGRAPH_OP(Rnn, ngraph::op)
NGRAPH_OP(SigmoidMultiplyBackprop, ngraph::op)
NGRAPH_OP(SigmoidMultiply, ngraph::op)
NGRAPH_OP(UpdateSlice, ngraph::op)
