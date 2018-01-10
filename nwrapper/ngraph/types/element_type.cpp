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
#include <string>
#include "ngraph/types/element_type.hpp"
#include "ngraph/ops/parameter.hpp"

namespace py = pybind11;
namespace ngraph {
namespace element {

PYBIND11_MODULE(Type, mod) {

    py::class_<Type, std::shared_ptr<Type>> type(mod, "Type");

    type.def(py::init<>());

    type.def("c_type_string", &Type::c_type_string);
    type.def("size", &Type::size);
    type.def("hash", &Type::hash);
    type.def("is_real", &Type::is_real);
    type.def("is_signed", &Type::is_signed);
    type.def("bitwidth", &Type::bitwidth);
    type.def_static("get_known_types", &Type::get_known_types);
    type.def("get_is_real", &Type::get_is_real);

    mod.attr("boolean") = element::boolean;
    mod.attr("f32")     = element::f32;
    mod.attr("f64")     = element::f64;
    mod.attr("i8")      = element::i8;
    mod.attr("i16")     = element::i16;
    mod.attr("i32")     = element::i32;
    mod.attr("i64")     = element::i64;
    mod.attr("u8")      = element::u8;
    mod.attr("u16")     = element::u16;
    mod.attr("u32")     = element::u32;
    mod.attr("u64")     = element::u64;
}

}}  // ngraph

