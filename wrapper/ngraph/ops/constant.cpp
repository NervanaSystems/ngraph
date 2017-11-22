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
#include "ngraph/ops/constant.hpp"

namespace py = pybind11;
namespace ngraph {
namespace op {
namespace {

template <typename T>
static void declareParameterizedConstant(py::module & mod, std::string const & suffix) {
    using Class = ParameterizedConstant<T>;
    using PyClass = py::class_<Class, std::shared_ptr<Class>, ConstantBase>;

    PyClass cls(mod, ("ParameterizedConstant" + suffix).c_str());
    cls.def(py::init<const ngraph::Shape&, std::shared_ptr<ngraph::runtime::ParameterizedTensorView<T>>& >());

}

}

PYBIND11_MODULE(clsParameterizedConstant, mod) {

    py::module::import("wrapper.ngraph.clsNode");
    py::module::import("wrapper.ngraph.runtime.clsParameterizedTensorView");
    using ET = ngraph::element::TraitedType<float>;

    py::class_<ConstantBase, std::shared_ptr<ConstantBase>, Node> clsConstantBase(mod, "ConstantBase");

    declareParameterizedConstant<ET>(mod, "F");
}

}}  // ngraph

