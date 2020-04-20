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

#include "ngraph/op/fused/clamp.hpp"
#include "pyngraph/ops/fused/clamp.hpp"

namespace py = pybind11;

void regclass_pyngraph_op_Clamp(py::module m)
{
    py::class_<ngraph::op::Clamp, std::shared_ptr<ngraph::op::Clamp>, ngraph::op::Op> clamp(
        m, "Clamp");
    clamp.doc() = "ngraph.impl.op.Clamp wraps ngraph::op::Clamp";
    clamp.def(py::init<const std::shared_ptr<ngraph::Node>&, const double, const double>());
}
