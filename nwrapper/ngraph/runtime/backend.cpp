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
#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/call_frame.hpp"
#include "ngraph/runtime/external_function.hpp"
#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/ngvm/ngvm_backend.hpp"
//#include "ngraph/runtime/parameterized_tensor_view.hpp"
#include "ngraph/runtime/tensor_view.hpp"

namespace py = pybind11;
namespace ngraph {
namespace runtime {

PYBIND11_MODULE(Backend, mod) {

    py::module::import("nwrapper.ngraph.runtime.CallFrame");
//    py::module::import("nwrapper.ngraph.runtime.ParameterizedTensorView");
    py::module::import("nwrapper.ngraph.runtime.TensorView");

    using ET = ngraph::element::TraitedType<float>;

    py::class_<Backend, std::shared_ptr<Backend>> backend(mod, "Backend");

    backend.def("make_call_frame", &Backend::make_call_frame);
    backend.def("make_primary_tensor_view", (std::shared_ptr<ngraph::runtime::TensorView> (Backend::*) (const ngraph::element::Type& , const ngraph::Shape& )) &Backend::make_primary_tensor_view);
    backend.def("make_primary_tensor_view", (std::shared_ptr<ngraph::runtime::TensorView> (Backend::*) (const ngraph::Shape& )) &Backend::make_primary_tensor_view<ET>);
    //backend.def("make_parameterized_tensor_view", (std::shared_ptr<ParameterizedTensorView<ET>> (Backend::*) (const ngraph::Shape& )) &Backend::make_parameterized_tensor_view);
    //backend.def("make_parameterized_tensor_view", (std::shared_ptr<ParameterizedTensorView<ET>> (Backend::*) (const NDArrayBase<ET::type>& )) &Backend::make_parameterized_tensor_view);
}

}}  // ngraph
