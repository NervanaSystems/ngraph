//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ngraph/descriptor/tensor.hpp"
#include "ngraph/runtime/tensor_view.hpp"
#include "pyngraph/runtime/tensor_view.hpp"

namespace py = pybind11;

void regclass_pyngraph_runtime_TensorView(py::module m)
{
    py::class_<ngraph::runtime::TensorView, std::shared_ptr<ngraph::runtime::TensorView>>
        tensorView(m, "TensorView");
    tensorView.doc() = "ngraph.impl.runtime.TensorView wraps ngraph::runtime::TensorView";
    tensorView.def("write",
                   (void (ngraph::runtime::TensorView::*)(const void*, size_t, size_t)) &
                       ngraph::runtime::TensorView::write);
    tensorView.def("read", &ngraph::runtime::TensorView::read);

    tensorView.def_property_readonly("shape", &ngraph::runtime::TensorView::get_shape);
    tensorView.def_property_readonly("element_count",
                                     &ngraph::runtime::TensorView::get_element_count);
    tensorView.def_property_readonly("element_type", [](const ngraph::runtime::TensorView& self) {
        return self.get_tensor().get_element_type();
    });
}
