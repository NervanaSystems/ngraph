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

#include "ngraph/coordinate_diff.hpp" //ngraph::CoordinateDiff
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "pyngraph/coordinate_diff.hpp"

namespace py = pybind11;

void regclass_pyngraph_CoordinateDiff(py::module m)
{
    py::class_<ngraph::CoordinateDiff, std::shared_ptr<ngraph::CoordinateDiff>> coordinate_diff(
        m, "CoordinateDiff");
    coordinate_diff.doc() = "ngraph.CoordinateDiff wraps ngraph::CoordinateDiff";
    coordinate_diff.def(py::init<const std::initializer_list<ptrdiff_t>&>());
    coordinate_diff.def(py::init<const std::vector<ptrdiff_t>&>());
    coordinate_diff.def(py::init<const ngraph::CoordinateDiff&>());
}
