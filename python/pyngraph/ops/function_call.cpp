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

#include "ngraph/function.hpp"
#include "ngraph/op/function_call.hpp"
#include "pyngraph/ops/function_call.hpp"

namespace py = pybind11;

void regclass_pyngraph_op_FunctionCall(py::module m)
{
    py::class_<ngraph::op::FunctionCall, std::shared_ptr<ngraph::op::FunctionCall>, ngraph::Node>
        function_call(m, "FunctionCall");
    function_call.doc() = "ngraph.impl.op.FunctionCall wraps ngraph::op::FunctionCall";
    function_call.def(py::init<std::shared_ptr<ngraph::Function>, const ngraph::NodeVector&>());
}
