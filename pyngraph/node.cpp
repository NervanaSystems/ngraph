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
//#include <string>
#include "ngraph/node.hpp"          // ngraph::Node
#include "ngraph/types/type.hpp"    // ngraph::ValueType
#include "ngraph/ops/add.hpp"       // ngraph::op::Add
#include "ngraph/ops/multiply.hpp"  // ngraph::op::Multiply
#include "ngraph/ops/divide.hpp"    // ngraph::op::Divide
#include "ngraph/ops/subtract.hpp"  // ngraph::op::Subtract
#include "pyngraph/node.hpp"

namespace py = pybind11;

void regclass_pyngraph_Node(py::module m){

    py::class_<ngraph::Node, std::shared_ptr<ngraph::Node>> node(m, "Node");

    node.def("__add__", [](const std::shared_ptr<ngraph::Node>& a, const std::shared_ptr<ngraph::Node> b) {
                return a + b;
               }, py::is_operator());
    node.def("__sub__", [](const std::shared_ptr<ngraph::Node>& a, const std::shared_ptr<ngraph::Node> b) {
                return a - b;
               }, py::is_operator());
    node.def("__mul__", [](const std::shared_ptr<ngraph::Node>& a, const std::shared_ptr<ngraph::Node> b) {
                return a * b;
               }, py::is_operator());
    node.def("__truediv__", [](const std::shared_ptr<ngraph::Node>& a, const std::shared_ptr<ngraph::Node> b) {
                return a / b;
               }, py::is_operator());

    node.def("get_shape", &ngraph::Node::get_shape);
    node.def("get_value_type", (std::shared_ptr<const ngraph::ValueType> (ngraph::Node::*)()) &ngraph::Node::get_value_type);
    node.def("get_value_type", (const std::shared_ptr<const ngraph::ValueType> (ngraph::Node::*)() const) &ngraph::Node::get_value_type);
}

