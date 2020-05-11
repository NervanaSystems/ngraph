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

#include <pybind11/pybind11.h>

#include "pyngraph/ops/regmodule_pyngraph_op.hpp"

namespace py = pybind11;

void regmodule_pyngraph_op(py::module m_op)
{
    regclass_pyngraph_op_AllReduce(m_op);
    regclass_pyngraph_op_ArgMax(m_op);
    regclass_pyngraph_op_ArgMin(m_op);
    regclass_pyngraph_op_AvgPool(m_op);
    regclass_pyngraph_op_AvgPoolBackprop(m_op);
    regclass_pyngraph_op_BatchNormInference(m_op);
    regclass_pyngraph_op_BatchNormTraining(m_op);
    regclass_pyngraph_op_BatchNormTrainingBackprop(m_op);
    regclass_pyngraph_op_Broadcast(m_op);
    regclass_pyngraph_op_BroadcastDistributed(m_op);
    regclass_pyngraph_op_Constant(m_op);
    regclass_pyngraph_op_Convert(m_op);
    regclass_pyngraph_op_Convolution(m_op);
    regclass_pyngraph_op_ConvolutionBackpropData(m_op);
    regclass_pyngraph_op_ConvolutionBackpropFilters(m_op);
    regclass_pyngraph_op_DepthToSpace(m_op);
    regclass_pyngraph_op_Dequantize(m_op);
    regclass_pyngraph_op_Dot(m_op);
    regclass_pyngraph_op_Gelu(m_op);
    regclass_pyngraph_op_Gemm(m_op);
    regclass_pyngraph_op_GetOutputElement(m_op);
    regclass_pyngraph_op_GRN(m_op);
    regclass_pyngraph_op_GroupConvolution(m_op);
    regclass_pyngraph_op_HardSigmoid(m_op);
    regclass_pyngraph_op_Max(m_op);
    regclass_pyngraph_op_Maximum(m_op);
    regclass_pyngraph_op_MaxPool(m_op);
    regclass_pyngraph_op_MaxPoolBackprop(m_op);
    regclass_pyngraph_op_Min(m_op);
    regclass_pyngraph_op_MVN(m_op);
    regclass_pyngraph_op_Parameter(m_op);
    regclass_pyngraph_op_Passthrough(m_op);
    regclass_pyngraph_op_Product(m_op);
    regclass_pyngraph_op_Quantize(m_op);
    regclass_pyngraph_op_QuantizedConvolution(m_op);
    regclass_pyngraph_op_QuantizedDot(m_op);
    regclass_pyngraph_op_ReplaceSlice(m_op);
    regclass_pyngraph_op_RNNCell(m_op);
    regclass_pyngraph_op_ScaleShift(m_op);
    regclass_pyngraph_op_ShuffleChannels(m_op);
    regclass_pyngraph_op_Slice(m_op);
    regclass_pyngraph_op_Softmax(m_op);
    regclass_pyngraph_op_SpaceToDepth(m_op);
    regclass_pyngraph_op_Result(m_op);
    regclass_pyngraph_op_Unsqueeze(m_op);
}
