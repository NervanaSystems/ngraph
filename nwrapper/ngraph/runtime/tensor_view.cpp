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
#include "ngraph/runtime/tensor_view.hpp"
#include "ngraph/runtime/value.hpp"

namespace py = pybind11;
namespace ngraph {
namespace runtime {

PYBIND11_MODULE(TensorView, mod) {

    py::class_<Value, std::shared_ptr<Value>> value(mod, "Value");
    py::class_<TensorView, std::shared_ptr<TensorView>, Value> tensorView(mod, "TensorView");
    tensorView.def("write", (void (TensorView::*) (const void*, size_t, size_t)) &TensorView::write);
    tensorView.def("read", &TensorView::read);
}

}}  // ngraph
