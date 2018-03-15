/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "ngraph/axis_set.hpp"      //ngraph::AxisSet
#include "pyngraph/axis_set.hpp"

namespace py = pybind11;

void regclass_pyngraph_AxisSet(py::module m) {

    py::class_<ngraph::AxisSet, std::shared_ptr<ngraph::AxisSet>> axis_set(m, "AxisSet");
    axis_set.doc() = "ngraph.AxisSet wraps ngraph::AxisSet";
    axis_set.def(py::init<const std::initializer_list<size_t>& >());
    axis_set.def(py::init<const std::set<size_t>& >());
    axis_set.def(py::init<const ngraph::AxisSet& >());
}
