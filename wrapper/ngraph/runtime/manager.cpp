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
#include "ngraph/runtime/manager.hpp"
#include "ngraph/runtime/ngvm/ngvm_manager.hpp"
#include "ngraph/function.hpp"
#include "ngraph/runtime/external_function.hpp"
#include "ngraph/runtime/backend.hpp"

namespace py = pybind11;
namespace ngraph {
namespace runtime {

PYBIND11_MODULE(Manager, mod) {

    py::module::import("wrapper.ngraph.Function");
    py::module::import("wrapper.ngraph.runtime.ExternalFunction");
    py::module::import("wrapper.ngraph.runtime.Backend");

    py::class_<Manager, std::shared_ptr<Manager>> manager(mod, "Manager");
    py::class_<ngvm::NGVMManager, std::shared_ptr<ngvm::NGVMManager>, Manager> ngvmManager(mod, "NGVMManager");

    manager.def_static("get", &Manager::get);
    manager.def("compile", &Manager::compile);
    manager.def("allocate_backend", &Manager::allocate_backend);
}

}}  // ngraph
