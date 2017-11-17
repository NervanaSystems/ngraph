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
#include <pybind11/numpy.h>
#include <string>
#include "ngraph/runtime/parameterized_tensor_view.hpp"

namespace py = pybind11;
namespace ngraph {
namespace runtime {
namespace {

template <typename ET>
static void declareParameterizedTensorView(py::module & mod, std::string const & suffix) {
    using Class = ParameterizedTensorView<ET>;
    using PyClass = py::class_<Class, std::shared_ptr<Class>, TensorView>;

    PyClass cls(mod, ("ParameterizedTensorView" + suffix).c_str());
#if 0
    cls.def("write", [] (py::array_t<float, py::array::c_style> a) {
      py::buffer_info a_info = a.request();
      &Class.write(a_info.ptr, 0, a_info.ndim);
    };
#endif
}

}

PYBIND11_PLUGIN(clsParameterizedTensorView) {

    py::module mod1("clsTensorView");
    py::class_<TensorView, std::shared_ptr<TensorView>> clsTensorView(mod1, "TensorView");
    py::module mod("clsParameterizedTensorView");

    py::module::import("wrapper.ngraph.types.clsTraitedType");

    declareParameterizedTensorView<ngraph::element::TraitedType<float>>(mod, "F");

    return mod.ptr();
}

}}  // ngraph
