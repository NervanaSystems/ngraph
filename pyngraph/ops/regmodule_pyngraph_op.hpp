// ----------------------------------------------------------------------------
// Copyright 2017 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------

#include <pybind11/pybind11.h>
#include "pyngraph/ops/add.hpp"
#include "pyngraph/ops/broadcast.hpp"
#include "pyngraph/ops/constant.hpp"
#include "pyngraph/ops/convert.hpp"
#include "pyngraph/ops/divide.hpp"
#include "pyngraph/ops/dot.hpp"
#include "pyngraph/ops/exp.hpp"
#include "pyngraph/ops/greater.hpp"
#include "pyngraph/ops/log.hpp"
#include "pyngraph/ops/maximum.hpp"
#include "pyngraph/ops/minimum.hpp"
#include "pyngraph/ops/multiply.hpp"
#include "pyngraph/ops/op.hpp"
#include "pyngraph/ops/parameter.hpp"
#include "pyngraph/ops/reduce.hpp"
#include "pyngraph/ops/reshape.hpp"
#include "pyngraph/ops/subtract.hpp"
#include "pyngraph/ops/sum.hpp"

namespace py = pybind11;

void regmodule_pyngraph_op(py::module m){
    m_op = m.def_submodule("op", "module pyngraph.op");
    regclass_pyngraph_op_Add(m_op);
    regclass_pyngraph_op_Broadcast(m_op);
    regclass_pyngraph_op_ConstantBase(m_op);
    regclass_pyngraph_op_Float32Constant(m_op);
    regclass_pyngraph_op_Float64Constant(m_op);
    regclass_pyngraph_op_Int8Constant(m_op);
    regclass_pyngraph_op_Int16Constant(m_op);
    regclass_pyngraph_op_Int32Constant(m_op);
    regclass_pyngraph_op_Int64Constant(m_op);
    regclass_pyngraph_op_UInt8Constant(m_op);
    regclass_pyngraph_op_UInt16Constant(m_op);
    regclass_pyngraph_op_UInt32Constant(m_op);
    regclass_pyngraph_op_UInt64Constant(m_op);
    regclass_pyngraph_op_Convert(m_op);
    regclass_pyngraph_op_Divide(m_op);
    regclass_pyngraph_op_Dot(m_op);
    regclass_pyngraph_op_Exp(m_op);
    regclass_pyngraph_op_Greater(m_op);
    regclass_pyngraph_op_Log(m_op);
    regclass_pyngraph_op_Maximum(m_op);
    regclass_pyngraph_op_Minimum(m_op);
    regclass_pyngraph_op_Multiply(m_op);
    regclass_pyngraph_op_RequiresTensorViewArgs(m_op);
    regclass_pyngraph_op_UnaryElementwise(m_op);
    regclass_pyngraph_op_UnaryElementwiseArithmetic(m_op);
    regclass_pyngraph_op_BinaryElementwise(m_op);
    regclass_pyngraph_op_BinaryElementwiseComparison(m_op);
    regclass_pyngraph_op_BinaryElementwiseArithmetic(m_op);
    regclass_pyngraph_op_Parameter(m_op);
    regclass_pyngraph_op_Reduce(m_op);
    regclass_pyngraph_op_Reshape(m_op);
    regclass_pyngraph_op_Subtract(m_op);
    regclass_pyngraph_op_Sum(m_op);
}
