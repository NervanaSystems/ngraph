//*****************************************************************************
// Copyright 2017-2019 Intel Corporation
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

#include "ngraph/op/fused/group_conv.hpp"
#include "pyngraph/ops/fused/group_conv.hpp"

namespace py = pybind11;

void regclass_pyngraph_op_GroupConvolution(py::module m)
{
    py::class_<ngraph::op::GroupConvolution,
               std::shared_ptr<ngraph::op::GroupConvolution>,
               ngraph::op::Op>
        groupconvolution(m, "GroupConvolution");
    groupconvolution.doc() = "ngraph.impl.op.GroupConvolution wraps ngraph::op::GroupConvolution";
    groupconvolution.def(py::init<const std::shared_ptr<ngraph::Node>&,
                                  const std::shared_ptr<ngraph::Node>&,
                                  const ngraph::Strides&,
                                  const ngraph::Strides&,
                                  const ngraph::CoordinateDiff&,
                                  const ngraph::CoordinateDiff&,
                                  const ngraph::Strides&,
                                  const size_t,
                                  const ngraph::op::PadType&>());
    py::enum_<ngraph::op::PadType>(groupconvolution, "PadType", py::arithmetic())
        .value("EXPLICIT", ngraph::op::PadType::EXPLICIT)
        .value("SAME_LOWER", ngraph::op::PadType::SAME_LOWER)
        .value("SAME_UPPER", ngraph::op::PadType::SAME_UPPER)
        .value("VALID", ngraph::op::PadType::VALID)
        .value("AUTO", ngraph::op::PadType::AUTO)
        .value("NOTSET", ngraph::op::PadType::NOTSET)
        .export_values();
}
