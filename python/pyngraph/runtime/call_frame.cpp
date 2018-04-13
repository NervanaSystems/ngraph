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
#include "pyngraph/runtime/call_frame.hpp"

namespace py = pybind11;

void regclass_pyngraph_runtime_CallFrame(py::module m)
{
    py::class_<ngraph::runtime::CallFrame, std::shared_ptr<ngraph::runtime::CallFrame>> callFrame(
        m, "CallFrame");
    callFrame.doc() = "ngraph.impl.runtime.CallFrame wraps ngraph::runtime::CallFrame";
    callFrame.def("call", &ngraph::runtime::CallFrame::call);
}
