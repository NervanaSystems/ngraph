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
#include "ngraph/runtime/utils.hpp"

namespace py = pybind11;
namespace ngraph {
namespace runtime{

PYBIND11_MODULE(Utils, mod) {

    py::module::import("wrapper.ngraph.runtime.ParameterizedTensorView");
    using ET = ngraph::element::TraitedType<float>;    

    mod.def("make_tensor", (std::shared_ptr<ngraph::runtime::ParameterizedTensorView<ET>> (*) (const ngraph::Shape&)) &make_tensor);
    mod.def("make_tensor", (std::shared_ptr<ngraph::runtime::ParameterizedTensorView<ET>> (*) (const Shape& , const std::vector<typename ET::type>& )) &make_tensor);
}

}}  // ngraph
