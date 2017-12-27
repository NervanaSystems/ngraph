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
#include "ngraph/function.hpp"
#include "ngraph/types/type.hpp"
#include "ngraph/common.hpp"

namespace py = pybind11;
namespace ngraph {

PYBIND11_MODULE(Function, mod) {

    py::module::import("nwrapper.ngraph.ops.Parameter");
    py::module::import("nwrapper.ngraph.types.TensorViewType");

    py::class_<Function, std::shared_ptr<Function>> function(mod, "Function");

    function.def(py::init<const std::shared_ptr<Node>&, const std::shared_ptr<const ValueType>&,
                          const std::vector<std::shared_ptr<op::Parameter>>&, const std::string&>());
    function.def(py::init<const Nodes&, const std::vector<std::shared_ptr<const ValueType>>&,
                          const std::vector<std::shared_ptr<op::Parameter>>&, const std::string&>());
    function.def("get_result_type", &Function::get_result_type);
    function.def("get_results", &Function::get_results);
}

}  // ngraph
