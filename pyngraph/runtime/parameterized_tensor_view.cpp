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
//#include <pybind11/numpy.h>
//#include <string>
#include "ngraph/runtime/parameterized_tensor_view.hpp"
#include "pyngraph/runtime/parameterized_tensor_view.hpp"

namespace py = pybind11;

void regclass_pyngraph_runtime_ParameterizedTensorView(py::module m) {
    using PTVFloat32 = ngraph::runtime::ParameterizedTensorView<ngraph::element::Float32>;
    py::class_<PTVFloat32, std::shared_ptr<PTVFloat32>, ngraph::runtime::TensorView> ptvfloat32(m, "ParameterizedTensorViewFloat32");
    ptvfloat32.def("write", (void (PTVFloat32::*) (const void*, size_t, size_t)) &PTVFloat32::write);
    ptvfloat32.def("read", &PTVFloat32::read);
    // TODO: add other types as well
}
