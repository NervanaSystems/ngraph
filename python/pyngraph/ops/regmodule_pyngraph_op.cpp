/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "pyngraph/ops/regmodule_pyngraph_op.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

void regmodule_pyngraph_op(py::module m_op)
{
    regclass_pyngraph_op_Abs(m_op);
    regclass_pyngraph_op_Acos(m_op);
    regclass_pyngraph_op_Asin(m_op);
    regclass_pyngraph_op_Atan(m_op);
    regclass_pyngraph_op_AvgPool(m_op);
    regclass_pyngraph_op_AvgPoolBackprop(m_op);
    regclass_pyngraph_op_Cos(m_op);
    regclass_pyngraph_op_Cosh(m_op);
    regclass_pyngraph_op_Add(m_op);
    regclass_pyngraph_op_And(m_op);
    regclass_pyngraph_op_Broadcast(m_op);
    regclass_pyngraph_op_Ceiling(m_op);
    regclass_pyngraph_op_Concat(m_op);
    regclass_pyngraph_op_Constant(m_op);
    regclass_pyngraph_op_Convert(m_op);
    regclass_pyngraph_op_Convolution(m_op);
    regclass_pyngraph_op_ConvolutionBackpropData(m_op);
    regclass_pyngraph_op_ConvolutionBackpropFilters(m_op);
    regclass_pyngraph_op_Divide(m_op);
    regclass_pyngraph_op_Dot(m_op);
    regclass_pyngraph_op_Equal(m_op);
    regclass_pyngraph_op_Exp(m_op);
    regclass_pyngraph_op_Floor(m_op);
    regclass_pyngraph_op_Greater(m_op);
    regclass_pyngraph_op_GreaterEq(m_op);
    regclass_pyngraph_op_Less(m_op);
    regclass_pyngraph_op_LessEq(m_op);
    regclass_pyngraph_op_Log(m_op);
    regclass_pyngraph_op_LRN(m_op);
    regclass_pyngraph_op_MaxPool(m_op);
    regclass_pyngraph_op_MaxPoolBackprop(m_op);
    regclass_pyngraph_op_Maximum(m_op);
    regclass_pyngraph_op_Minimum(m_op);
    regclass_pyngraph_op_Multiply(m_op);
    regclass_pyngraph_op_Negative(m_op);
    regclass_pyngraph_op_Not(m_op);
    regclass_pyngraph_op_NotEqual(m_op);
    regclass_pyngraph_op_Pad(m_op);
    regclass_pyngraph_op_ParameterVector(m_op);
    regclass_pyngraph_op_Parameter(m_op);
    regclass_pyngraph_op_Power(m_op);
    regclass_pyngraph_op_OneHot(m_op);
    // regclass_pyngraph_op_Op(m_op);
    regclass_pyngraph_op_Or(m_op);
    regclass_pyngraph_op_Reduce(m_op);
    regclass_pyngraph_op_ReplaceSlice(m_op);
    regclass_pyngraph_op_Reshape(m_op);
    regclass_pyngraph_op_Reverse(m_op);
    regclass_pyngraph_op_Select(m_op);
    regclass_pyngraph_op_Sign(m_op);
    regclass_pyngraph_op_Sin(m_op);
    regclass_pyngraph_op_Sinh(m_op);
    regclass_pyngraph_op_Slice(m_op);
    regclass_pyngraph_op_Sqrt(m_op);
    regclass_pyngraph_op_Subtract(m_op);
    regclass_pyngraph_op_Sum(m_op);
    regclass_pyngraph_op_Tan(m_op);
    regclass_pyngraph_op_Tanh(m_op);
    regclass_pyngraph_op_Relu(m_op);
    regclass_pyngraph_op_ReluBackprop(m_op);
    regclass_pyngraph_op_Max(m_op);
    regclass_pyngraph_op_Product(m_op);
    regclass_pyngraph_op_AllReduce(m_op);
    regclass_pyngraph_op_FunctionCall(m_op);
    regclass_pyngraph_op_GetOutputElement(m_op);
    regclass_pyngraph_op_Min(m_op);
    regclass_pyngraph_op_BatchNorm(m_op);
    regclass_pyngraph_op_BatchNormBackprop(m_op);
    regclass_pyngraph_op_Softmax(m_op);
}
