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

#include "ngraph/node.hpp" // ngraph::Node
#include "ngraph/node_vector.hpp"
#include "pyngraph/node.hpp"
#include "pyngraph/node_vector.hpp"

namespace py = pybind11;

void regclass_pyngraph_NodeVector(py::module m)
{
    py::class_<ngraph::NodeVector, std::shared_ptr<ngraph::NodeVector>> node_vector(m,
                                                                                    "NodeVector");
    node_vector.doc() = "ngraph.impl.NodeVector wraps ngraph::NodeVector";
    node_vector.def(py::init<const std::initializer_list<std::shared_ptr<ngraph::Node>>&>());
    node_vector.def(py::init<const std::vector<std::shared_ptr<ngraph::Node>>&>());
    node_vector.def(py::init<const ngraph::NodeVector&>());
}
