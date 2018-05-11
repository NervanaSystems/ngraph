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
//#include <string>
#include "ngraph/node.hpp"        // ngraph::Node
#include "ngraph/op/add.hpp"      // ngraph::op::Add
#include "ngraph/op/divide.hpp"   // ngraph::op::Divide
#include "ngraph/op/multiply.hpp" // ngraph::op::Multiply
#include "ngraph/op/subtract.hpp" // ngraph::op::Subtract
#include "pyngraph/node.hpp"

namespace py = pybind11;

void regclass_pyngraph_Node(py::module m)
{
    py::class_<ngraph::Node, std::shared_ptr<ngraph::Node>> node(m, "Node");
    node.doc() = "ngraph.impl.Node wraps ngraph::Node";
    node.def("__add__",
             [](const std::shared_ptr<ngraph::Node>& a, const std::shared_ptr<ngraph::Node> b) {
                 return a + b;
             },
             py::is_operator());
    node.def("__sub__",
             [](const std::shared_ptr<ngraph::Node>& a, const std::shared_ptr<ngraph::Node> b) {
                 return a - b;
             },
             py::is_operator());
    node.def("__mul__",
             [](const std::shared_ptr<ngraph::Node>& a, const std::shared_ptr<ngraph::Node> b) {
                 return a * b;
             },
             py::is_operator());
    node.def("__div__",
             [](const std::shared_ptr<ngraph::Node>& a, const std::shared_ptr<ngraph::Node> b) {
                 return a / b;
             },
             py::is_operator());
    node.def("__truediv__",
             [](const std::shared_ptr<ngraph::Node>& a, const std::shared_ptr<ngraph::Node> b) {
                 return a / b;
             },
             py::is_operator());

    node.def("__repr__", [](const ngraph::Node& self) {
        std::string class_name = py::cast(self).get_type().attr("__name__").cast<std::string>();
        std::string shape = py::cast(self.get_shape()).attr("__str__")().cast<std::string>();
        return "<" + class_name + ": '" + self.get_name() + "' (" + shape + ")>";
    });

    node.def("get_output_size", &ngraph::Node::get_output_size);
    node.def("get_output_element_type", &ngraph::Node::get_output_element_type);
    node.def("get_element_type", &ngraph::Node::get_element_type);
    node.def("get_output_shape", &ngraph::Node::get_output_shape);
    node.def("get_shape", &ngraph::Node::get_shape);
    node.def("get_argument", &ngraph::Node::get_argument);
    node.def("get_unique_name", &ngraph::Node::get_name);

    node.def_property("name", &ngraph::Node::get_friendly_name, &ngraph::Node::set_name);
    node.def_property_readonly("shape", &ngraph::Node::get_shape);
}
