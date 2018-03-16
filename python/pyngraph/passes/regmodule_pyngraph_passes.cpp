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

#include "pyngraph/passes/regmodule_pyngraph_passes.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

void regmodule_pyngraph_passes(py::module m)
{
    py::module m_passes = m.def_submodule("passes", "Package ngraph.passes wraps ngraph::passes");
    regclass_pyngraph_passes_Manager(m_passes);
}
