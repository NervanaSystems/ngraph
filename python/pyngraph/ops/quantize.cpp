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

#include "ngraph/op/quantize.hpp"
#include "pyngraph/ops/quantize.hpp"

namespace py = pybind11;

void regclass_pyngraph_op_Quantize(py::module m)
{
    py::class_<ngraph::op::Quantize, std::shared_ptr<ngraph::op::Quantize>, ngraph::op::Op>
        quantize(m, "Quantize");
    quantize.doc() = "ngraph.impl.op.Quantize wraps ngraph::op::Quantize";
    quantize.def(py::init<const std::shared_ptr<ngraph::Node>&,
                          const std::shared_ptr<ngraph::Node>&,
                          const std::shared_ptr<ngraph::Node>&,
                          const ngraph::element::Type&,
                          const ngraph::AxisSet&,
                          ngraph::op::Quantize::RoundMode>());
    py::enum_<ngraph::op::Quantize::RoundMode>(quantize, "RoundMode", py::arithmetic())
        .value("ROUND_NEAREST_TOWARD_INFINITY",
               ngraph::op::Quantize::RoundMode::ROUND_NEAREST_TOWARD_INFINITY)
        .value("ROUND_NEAREST_TOWARD_ZERO",
               ngraph::op::Quantize::RoundMode::ROUND_NEAREST_TOWARD_ZERO)
        .value("ROUND_NEAREST_UPWARD", ngraph::op::Quantize::RoundMode::ROUND_NEAREST_UPWARD)
        .value("ROUND_NEAREST_DOWNWARD", ngraph::op::Quantize::RoundMode::ROUND_NEAREST_DOWNWARD)
        .value("ROUND_NEAREST_TOWARD_EVEN",
               ngraph::op::Quantize::RoundMode::ROUND_NEAREST_TOWARD_EVEN)
        .value("ROUND_TOWARD_INFINITY", ngraph::op::Quantize::RoundMode::ROUND_TOWARD_INFINITY)
        .value("ROUND_TOWARD_ZERO", ngraph::op::Quantize::RoundMode::ROUND_TOWARD_ZERO)
        .value("ROUND_UP", ngraph::op::Quantize::RoundMode::ROUND_UP)
        .value("ROUND_DOWN", ngraph::op::Quantize::RoundMode::ROUND_DOWN)
        .export_values();
}
