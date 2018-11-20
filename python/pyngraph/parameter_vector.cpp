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

#include "ngraph/op/parameter.hpp" // ngraph::op::Parameter
#include "ngraph/parameter_vector.hpp"
#include "pyngraph/ops/parameter.hpp"
#include "pyngraph/parameter_vector.hpp"

namespace py = pybind11;

void regclass_pyngraph_ParameterVector(py::module m)
{
    py::class_<ngraph::ParameterVector, std::shared_ptr<ngraph::ParameterVector>> parameter_vector(
        m, "ParameterVector");
    parameter_vector.doc() = "ngraph.impl.ParameterVector wraps ngraph::ParameterVector";
    parameter_vector.def(
        py::init<const std::initializer_list<std::shared_ptr<ngraph::op::Parameter>>&>());
    parameter_vector.def(py::init<const std::vector<std::shared_ptr<ngraph::op::Parameter>>&>());
    parameter_vector.def(py::init<const ngraph::ParameterVector&>());
    parameter_vector.def("__len__", [](const ngraph::ParameterVector& v) { return v.size(); });
    parameter_vector.def("__getitem__",
                         [](const ngraph::ParameterVector& v, int key) { return v[key]; });
    parameter_vector.def(
        "__iter__",
        [](ngraph::ParameterVector& v) { return py::make_iterator(v.begin(), v.end()); },
        py::keep_alive<0, 1>()); /* Keep vector alive while iterator is used */
}
