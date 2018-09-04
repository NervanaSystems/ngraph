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

#include "ngraph/op/relu.hpp"
#include "pyngraph/ops/relu.hpp"

namespace py = pybind11;

void regclass_pyngraph_op_Relu(py::module m)
{
    py::class_<ngraph::op::Relu,
               std::shared_ptr<ngraph::op::Relu>,
               ngraph::op::util::UnaryElementwiseArithmetic>
        relu(m, "Relu");
    relu.doc() = "ngraph.impl.op.Relu wraps ngraph::op::Relu";
    relu.def(py::init<std::shared_ptr<ngraph::Node>&>());
}

void regclass_pyngraph_op_ReluBackprop(py::module m)
{
    py::class_<ngraph::op::ReluBackprop, std::shared_ptr<ngraph::op::ReluBackprop>, ngraph::op::Op>
        relu_backprop(m, "ReluBackprop");
    relu_backprop.def(py::init<std::shared_ptr<ngraph::Node>&, std::shared_ptr<ngraph::Node>&>());
}
