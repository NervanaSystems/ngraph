//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#include <iterator>
#include <sstream>
#include <string>

#include "ngraph/dimension.hpp"
#include "ngraph/partial_shape.hpp"
#include "pyngraph/partial_shape.hpp"

namespace py = pybind11;

void regclass_pyngraph_PartialShape(py::module m)
{
    py::class_<ngraph::PartialShape, std::shared_ptr<ngraph::PartialShape>> shape(m,
                                                                                  "PartialShape");
    shape.doc() = "ngraph.impl.Shape wraps ngraph::PartialShape";
    shape.def(py::init<std::initializer_list<ngraph::Dimension>&>());
    shape.def(py::init<const std::initializer_list<ngraph::Dimension>&>());
    shape.def(py::init<const ngraph::Shape&>());
    shape.def("is_static", &ngraph::PartialShape::is_static);
    shape.def("is_dynamic", &ngraph::PartialShape::is_dynamic);
}
