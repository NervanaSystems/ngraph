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
#include "ngraph/node.hpp"
#include "ngraph/ops/add.hpp"
#include "ngraph/ops/multiply.hpp"
#include "ngraph/ops/divide.hpp"

namespace py = pybind11;
namespace ngraph {

PYBIND11_MODULE(Node, mod) {

    py::class_<Node, std::shared_ptr<Node>> node(mod, "Node");
 
    node.def("__add__", [](const std::shared_ptr<ngraph::Node>& a, const std::shared_ptr<ngraph::Node> b) {
                return a + b;
               }, py::is_operator());
    node.def("__mul__", [](const std::shared_ptr<ngraph::Node>& a, const std::shared_ptr<ngraph::Node> b) {
                return a * b;
               }, py::is_operator());
    node.def("__truediv__", [](const std::shared_ptr<ngraph::Node>& a, const std::shared_ptr<ngraph::Node> b) {
                return a/b;
               }, py::is_operator());
}

}  // ngraph
