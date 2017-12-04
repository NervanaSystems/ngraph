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
#include <pybind11/operators.h>
#include <string>
#include "ngraph/node.hpp"
#include "ngraph/ops/parameter.hpp"

namespace py = pybind11;
namespace ngraph {

PYBIND11_MODULE(Parameter, mod) {

    py::module::import("nwrapper.ngraph.types.TraitedType");
    py::module::import("nwrapper.ngraph.Node");
    py::class_<op::Parameter, std::shared_ptr<op::Parameter>, Node> parameter(mod, "Parameter");

    parameter.def(py::init<const ngraph::element::Type&, const ngraph::Shape& >());
    parameter.def("description", &op::Parameter::description);
}

}  // ngraph
