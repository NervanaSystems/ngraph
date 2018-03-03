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
#include "ngraph/shape.hpp"      //ngraph::Shape
#include "ngraph/ops/parameter.hpp" //ngraph::op::Parameter
#include "pyngraph/shape.hpp"

namespace py = pybind11;

void regclass_pyngraph_Shape(py::module m) {

    py::class_<ngraph::Shape, std::shared_ptr<ngraph::Shape>> shape(m, "Shape");
    shape.def(py::init<const std::initializer_list<size_t>& >());
    shape.def(py::init<const std::vector<size_t>& >());
    shape.def(py::init<const ngraph::Shape& >());
}
