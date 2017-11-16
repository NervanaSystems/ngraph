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
#include "ngraph/types/type.hpp"

namespace py = pybind11;
namespace ngraph {

PYBIND11_PLUGIN(clsTensorViewType) {

    py::module mod("clsTensorViewType");
    py::module mod_1("clsValueType");
    py::class_<ValueType, std::shared_ptr<ValueType>> clsValueType(mod_1, "clsValueType"); 
    py::class_<TensorViewType, std::shared_ptr<TensorViewType>, ValueType> clsTensorViewType(mod, "clsTensorViewType");

    clsTensorViewType.def(py::init<const element::Type&, const Shape&>());
    clsTensorViewType.def("get_shape", &TensorViewType::get_shape);

    return mod.ptr();

}

}  // ngraph
