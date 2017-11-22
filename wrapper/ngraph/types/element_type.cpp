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
namespace {

template <typename T>
static void declareTraitedType(py::module & mod, std::string const & suffix) {
    using Class = TraitedType<T>;
    using PyClass = py::class_<Class, std::shared_ptr<Class>, Type>;

    PyClass cls(mod, ("TraitedType" + suffix).c_str());

    //cls.def(py::init<>());
    cls.def_static("element_type", &Class::element_type,
                   py::return_value_policy::reference);
    cls.def_static("read", (T (*) (const std::string&)) &Class::read);
    cls.def_static("read", (std::vector<T> (*) (const std::vector<std::string>&)) &Class::read);
    cls.def("make_primary_tensor_view", &Class::make_primary_tensor_view);
}

}

PYBIND11_MODULE(TraitedType, mod) {

    py::class_<Type, std::shared_ptr<Type>> type(mod, "Type");

    declareTraitedType<float>(mod, "F");
    declareTraitedType<double>(mod, "D");
    declareTraitedType<int>(mod, "I");
}

}}  // ngraph

