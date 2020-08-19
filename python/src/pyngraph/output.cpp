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

#include "ngraph/node_output.hpp"  // ngraph::Output<Node>
#include "ngraph/op/parameter.hpp" // ngraph::op::v0::Parameter
#include "pyngraph/output.hpp"

namespace py = pybind11;

static const char* CAPSULE_NAME = "ngraph_function";

void regclass_pyngraph_Output(py::module m)
{
    py::class_<ngraph::Output<ngraph::Node>> output(m, "Output");
    output.doc() = "ngraph.impl.Output wraps ngraph::Output<Node>";
    output.def("get_index", &ngraph::Output<ngraph::Node>::get_index);
    output.def("get_node", &ngraph::Output<ngraph::Node>::get_node_shared_ptr);
}
