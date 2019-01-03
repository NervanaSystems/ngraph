//*****************************************************************************
// Copyright 2017-2019 Intel Corporation
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

#include <iterator>
#include <sstream>
#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ngraph/shape.hpp" // ngraph::Shape
#include "pyngraph/shape.hpp"

namespace py = pybind11;

void regclass_pyngraph_Shape(py::module m)
{
    py::class_<ngraph::Shape, std::shared_ptr<ngraph::Shape>> shape(m, "Shape");
    shape.doc() = "ngraph.impl.Shape wraps ngraph::Shape";
    shape.def(py::init<const std::initializer_list<size_t>&>());
    shape.def(py::init<const std::vector<size_t>&>());
    shape.def(py::init<const ngraph::Shape&>());
    shape.def("__len__", [](const ngraph::Shape& v) { return v.size(); });
    shape.def("__getitem__", [](const ngraph::Shape& v, int key) { return v[key]; });

    shape.def("__iter__",
              [](ngraph::Shape& v) { return py::make_iterator(v.begin(), v.end()); },
              py::keep_alive<0, 1>()); /* Keep vector alive while iterator is used */

    shape.def("__str__", [](const ngraph::Shape& self) -> std::string {
        std::stringstream shape_ss;
        std::copy(self.begin(), self.end(), std::ostream_iterator<int>(shape_ss, ", "));
        std::string shape_str = shape_ss.str();
        return "[" + shape_str.substr(0, shape_str.size() - 2) + "]";
    });

    shape.def("__repr__", [](const ngraph::Shape& self) -> std::string {
        std::string class_name = py::cast(self).get_type().attr("__name__").cast<std::string>();
        std::string shape_str = py::cast(self).attr("__str__")().cast<std::string>();
        return "<" + class_name + ": (" + shape_str + ")>";
    });
}
