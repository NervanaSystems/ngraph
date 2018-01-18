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
//#include <string>
#include "ngraph/runtime/external_function.hpp"
#include "pyngraph/runtime/external_function.hpp"

namespace py = pybind11;

void regclass_pyngraph_runtime_ExternalFunction(py::module m){
    py::class_<ngraph::runtime::ExternalFunction, std::shared_ptr<ngraph::runtime::ExternalFunction>> externalFunction(m, "ExternalFunction");
}
