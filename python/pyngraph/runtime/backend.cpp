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

#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "pyngraph/runtime/backend.hpp"

namespace py = pybind11;

void regclass_pyngraph_runtime_Backend(py::module m)
{
    py::class_<ngraph::runtime::Backend, std::unique_ptr<ngraph::runtime::Backend>> backend(
        m, "Backend");
    backend.doc() = "ngraph.impl.runtime.Backend wraps ngraph::runtime::Backend";
    backend.def_static("create", &ngraph::runtime::Backend::create);
    backend.def_static("get_registered_devices", &ngraph::runtime::Backend::get_registered_devices);
    backend.def("create_tensor",
                (std::shared_ptr<ngraph::runtime::Tensor>(ngraph::runtime::Backend::*)(
                    const ngraph::element::Type&, const ngraph::Shape&)) &
                    ngraph::runtime::Backend::create_tensor);
    backend.def("compile",
                (void (ngraph::runtime::Backend::*)(std::shared_ptr<ngraph::Function>)) &
                    ngraph::runtime::Backend::compile);
    backend.def("call",
                (void (ngraph::runtime::Backend::*)(
                    std::shared_ptr<ngraph::Function>,
                    const std::vector<std::shared_ptr<ngraph::runtime::Tensor>>&,
                    const std::vector<std::shared_ptr<ngraph::runtime::Tensor>>&)) &
                    ngraph::runtime::Backend::call);
    backend.def("remove_compiled_function",
                (void (ngraph::runtime::Backend::*)(std::shared_ptr<ngraph::Function>)) &
                    ngraph::runtime::Backend::remove_compiled_function);
    backend.def("enable_performance_data",
                (void (ngraph::runtime::Backend::*)(std::shared_ptr<ngraph::Function>, bool)) &
                    ngraph::runtime::Backend::enable_performance_data);
    backend.def("get_performance_data",
                (std::vector<ngraph::runtime::PerformanceCounter>(ngraph::runtime::Backend::*)(
                    std::shared_ptr<ngraph::Function>)) &
                    ngraph::runtime::Backend::get_performance_data);
}
