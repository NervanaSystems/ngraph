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

const ngraph::element::Type& boolean() { return ngraph::element::boolean; }
const ngraph::element::Type& f32() { return ngraph::element::f32; }
const ngraph::element::Type& f64() { return ngraph::element::f64; }
const ngraph::element::Type& i8() { return ngraph::element::i8; }
const ngraph::element::Type& i16() { return ngraph::element::i16; }
const ngraph::element::Type& i32() { return ngraph::element::i32; }
const ngraph::element::Type& i64() { return ngraph::element::i64; }
const ngraph::element::Type& u8() { return ngraph::element::u8; }
const ngraph::element::Type& u16() { return ngraph::element::u16; }
const ngraph::element::Type& u32() { return ngraph::element::u32; }
const ngraph::element::Type& u64() { return ngraph::element::u64; }

void regclass_pyngraph_Type(py::module m){
    py::class_<ngraph::element::Type, std::shared_ptr<ngraph::element::Type>> type(m, "Type");
    type.def_static("boolean", &boolean, py::return_value_policy::reference);
    type.def_static("f32", &f32, py::return_value_policy::reference);
    type.def_static("f64", &f64, py::return_value_policy::reference);
    type.def_static("i8", &i8, py::return_value_policy::reference);
    type.def_static("i16", &i16, py::return_value_policy::reference);
    type.def_static("i32", &i32, py::return_value_policy::reference);
    type.def("i64", &i64, py::return_value_policy::reference);
    type.def("u8", &u8, py::return_value_policy::reference);
    type.def("u16", &u16, py::return_value_policy::reference);
    type.def("u32", &u32, py::return_value_policy::reference);
    type.def("u64", &u64, py::return_value_policy::reference);
}
