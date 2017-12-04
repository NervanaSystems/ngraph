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
#include "ngraph/ops/exp.hpp"

namespace py = pybind11;
namespace ngraph {
namespace op {

PYBIND11_MODULE(Exp, mod) {

    py::module::import("nwrapper.ngraph.ops.Op");

    py::class_<Exp, std::shared_ptr<Exp>, UnaryElementwiseArithmetic> exp(mod, "Exp");
    exp.def(py::init<const std::shared_ptr<ngraph::Node>& >());
}

}}  // ngraph
