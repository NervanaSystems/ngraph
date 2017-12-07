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
#include "ngraph/types/element_type.hpp"
#include "pyngraph/types/element_type.hpp"
#include "ngraph/ops/parameter.hpp"

namespace py = pybind11;

void regclass_pyngraph_Type(py::module m){
    py::class_<ngraph::element::Type> type(m, "Type");
}
void regclass_pyngraph_Bool(py::module m)
{
    py::class_<ngraph::element::Bool, ngraph::element::Type> tbool(m, "Bool");
    tbool.def_static("element_type", &ngraph::element::Bool::element_type,
                     py::return_value_policy::reference);
}
void regclass_pyngraph_Float32(py::module m)
{
    py::class_<ngraph::element::Float32, ngraph::element::Type> tfloat32(m, "Float32");
    tfloat32.def_static("element_type", &ngraph::element::Float32::element_type,
                     py::return_value_policy::reference);
}
void regclass_pyngraph_Float64(py::module m)
{
    py::class_<ngraph::element::Float64, ngraph::element::Type> tfloat64(m, "Float64");
    tfloat64.def_static("element_type", &ngraph::element::Float64::element_type,
                     py::return_value_policy::reference);
}
void regclass_pyngraph_Int8(py::module m)
{
    py::class_<ngraph::element::Int8, ngraph::element::Type> tint8(m, "Int8");
    tint8.def_static("element_type", &ngraph::element::Int8::element_type,
                     py::return_value_policy::reference);
}
/*
void regclass_pyngraph_Int16(py::module m)
{
    py::class_<ngraph::element::Int16, ngraph::element::Type> tint16(m, "Int16");
    tint16.def_static("element_type", &ngraph::element::Int16::element_type,
                     py::return_value_policy::reference);
}
*/
void regclass_pyngraph_Int32(py::module m)
{
    py::class_<ngraph::element::Int32, ngraph::element::Type> tint32(m, "Int32");
    tint32.def_static("element_type", &ngraph::element::Int32::element_type,
                     py::return_value_policy::reference);
}
void regclass_pyngraph_Int64(py::module m)
{
    py::class_<ngraph::element::Int64, ngraph::element::Type> tint64(m, "Int64");
    tint64.def_static("element_type", &ngraph::element::Int64::element_type,
                     py::return_value_policy::reference);
}
void regclass_pyngraph_UInt8(py::module m)
{
    py::class_<ngraph::element::UInt8, ngraph::element::Type> tuint8(m, "UInt8");
    tuint8.def_static("element_type", &ngraph::element::UInt8::element_type,
                     py::return_value_policy::reference);
}
/*
void regclass_pyngraph_UInt16(py::module m)
{
    py::class_<ngraph::element::UInt16, ngraph::element::Type> tuint16(m, "UInt16");
    tuint16.def_static("element_type", &ngraph::element::UInt16::element_type,
                     py::return_value_policy::reference);
}
*/
void regclass_pyngraph_UInt32(py::module m)
{
    py::class_<ngraph::element::UInt32, ngraph::element::Type> tuint32(m, "UInt32");
    tuint32.def_static("element_type", &ngraph::element::UInt32::element_type,
                     py::return_value_policy::reference);
}
void regclass_pyngraph_UInt64(py::module m)
{
    py::class_<ngraph::element::UInt64, ngraph::element::Type> tuint64(m, "UInt64");
    tuint64.def_static("element_type", &ngraph::element::UInt64::element_type,
                     py::return_value_policy::reference);
}

