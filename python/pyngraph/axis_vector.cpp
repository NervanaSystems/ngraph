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

#include "ngraph/axis_vector.hpp" // ngraph::AxisVector
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "pyngraph/axis_vector.hpp"

namespace py = pybind11;

void regclass_pyngraph_AxisVector(py::module m)
{
    py::class_<ngraph::AxisVector, std::shared_ptr<ngraph::AxisVector>> axis_vector(m,
                                                                                    "AxisVector");
    axis_vector.doc() = "ngraph.impl.AxisVector wraps ngraph::AxisVector";
    axis_vector.def(py::init<const std::initializer_list<size_t>&>());
    axis_vector.def(py::init<const std::vector<size_t>&>());
    axis_vector.def(py::init<const ngraph::AxisVector&>());
}
