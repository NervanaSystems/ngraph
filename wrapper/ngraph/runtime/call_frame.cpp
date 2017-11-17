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
#include "ngraph/runtime/call_frame.hpp"
#include "ngraph/runtime/ngvm/call_frame.hpp" 

namespace py = pybind11;
namespace ngraph {
namespace runtime {

PYBIND11_PLUGIN(clsCallFrame) {

    py::module mod("clsCallFrame");
    py::module mod_1("clsNGVMCallFrame");

    py::class_<CallFrame, std::shared_ptr<CallFrame>> clsCallFrame(mod, "CallFrame");
    py::class_<ngvm::CallFrame, std::shared_ptr<ngvm::CallFrame>, CallFrame> clsNGVMCallFrame(mod_1, "clsNGVMCallFrame");

    clsCallFrame.def("call", [](const std::vector<std::shared_ptr<ngraph::runtime::Value>>& inputs,
                              const std::vector<std::shared_ptr<ngraph::runtime::Value>>& outputs) {
                      (inputs, outputs);
                     }, py::is_operator());

    return mod.ptr();

}

}}  // ngraph

