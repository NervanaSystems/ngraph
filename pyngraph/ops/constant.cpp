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
#include <pybind11/stl.h>
//#include <string>
#include "ngraph/ops/constant.hpp"
#include "pyngraph/ops/constant.hpp"

namespace py = pybind11;

void regclass_pyngraph_op_ConstantBase(py::module m){
    py::class_<ngraph::op::ConstantBase, std::shared_ptr<ngraph::op::ConstantBase>, ngraph::Node> constantBase(m, "ConstantBase");
}
void regclass_pyngraph_op_Float32Constant(py::module m){
    py::class_<ngraph::op::Float32Constant, std::shared_ptr<ngraph::op::Float32Constant>, ngraph::op::ConstantBase> float32constant(m, "Float32Constant");
    float32constant.def(py::init<const ngraph::Shape&, std::shared_ptr<ngraph::runtime::ParameterizedTensorView<ngraph::element::Float32>>& >());
}
/*
void regclass_pyngraph_op_Float64Constant(py::module m){
    py::class_<ngraph::op::Float64Constant, std::shared_ptr<ngraph::op::Float64Constant>, ngraph::op::ConstantBase> float64constant(m, "Float64Constant");
    float64constant.def(py::init<const ngraph::Shape&, std::shared_ptr<ngraph::runtime::ParameterizedTensorView<ngraph::element::Float64>>& >());
}
*/
void regclass_pyngraph_op_Int8Constant(py::module m){
    py::class_<ngraph::op::Int8Constant, std::shared_ptr<ngraph::op::Int8Constant>, ngraph::op::ConstantBase> int8constant(m, "Int8Constant");
    int8constant.def(py::init<const ngraph::Shape&, std::shared_ptr<ngraph::runtime::ParameterizedTensorView<ngraph::element::Int8>>& >());
}
/*
void regclass_pyngraph_op_Int16Constant(py::module m){
    py::class_<ngraph::op::Int16Constant, std::shared_ptr<ngraph::op::Int16Constant>, ngraph::op::ConstantBase> int16constant(m, "Int16Constant");
    int16constant.def(py::init<const ngraph::Shape&, std::shared_ptr<ngraph::runtime::ParameterizedTensorView<ngraph::element::Int16>>& >());
}
*/
void regclass_pyngraph_op_Int32Constant(py::module m){
    py::class_<ngraph::op::Int32Constant, std::shared_ptr<ngraph::op::Int32Constant>, ngraph::op::ConstantBase> int32constant(m, "Int32Constant");
    int32constant.def(py::init<const ngraph::Shape&, std::shared_ptr<ngraph::runtime::ParameterizedTensorView<ngraph::element::Int32>>& >());
}
void regclass_pyngraph_op_Int64Constant(py::module m){
    py::class_<ngraph::op::Int64Constant, std::shared_ptr<ngraph::op::Int64Constant>, ngraph::op::ConstantBase> int64constant(m, "Int64Constant");
    int64constant.def(py::init<const ngraph::Shape&, std::shared_ptr<ngraph::runtime::ParameterizedTensorView<ngraph::element::Int64>>& >());
}
void regclass_pyngraph_op_UInt8Constant(py::module m){
    py::class_<ngraph::op::UInt8Constant, std::shared_ptr<ngraph::op::UInt8Constant>, ngraph::op::ConstantBase> uint8constant(m, "UInt8Constant");
    uint8constant.def(py::init<const ngraph::Shape&, std::shared_ptr<ngraph::runtime::ParameterizedTensorView<ngraph::element::UInt8>>& >());
}
/*
void regclass_pyngraph_op_UInt16Constant(py::module m){
    py::class_<ngraph::op::UInt16Constant, std::shared_ptr<ngraph::op::UInt16Constant>, ngraph::op::ConstantBase> uint16constant(m, "UInt16Constant");
    uint16constant.def(py::init<const ngraph::Shape&, std::shared_ptr<ngraph::runtime::ParameterizedTensorView<ngraph::element::UInt16>>& >());
}
*/
void regclass_pyngraph_op_UInt32Constant(py::module m){
    py::class_<ngraph::op::UInt32Constant, std::shared_ptr<ngraph::op::UInt32Constant>, ngraph::op::ConstantBase> uint32constant(m, "UInt32Constant");
    uint32constant.def(py::init<const ngraph::Shape&, std::shared_ptr<ngraph::runtime::ParameterizedTensorView<ngraph::element::UInt32>>& >());
}
void regclass_pyngraph_op_UInt64Constant(py::module m){
    py::class_<ngraph::op::UInt64Constant, std::shared_ptr<ngraph::op::UInt64Constant>, ngraph::op::ConstantBase> uint64constant(m, "UInt64Constant");
    uint64constant.def(py::init<const ngraph::Shape&, std::shared_ptr<ngraph::runtime::ParameterizedTensorView<ngraph::element::UInt64>>& >());
}

