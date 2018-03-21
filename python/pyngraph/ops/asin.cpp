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

#include "ngraph/op/asin.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "pyngraph/ops/asin.hpp"

namespace py = pybind11;

void regclass_pyngraph_op_Asin(py::module m)
{
    py::class_<ngraph::op::Asin,
               std::shared_ptr<ngraph::op::Asin>,
               ngraph::op::util::UnaryElementwiseArithmetic>
        asin(m, "Asin");
    asin.doc() = "ngraph.impl.op.Asin wraps ngraph::op::Asin";
    asin.def(py::init<const std::shared_ptr<ngraph::Node>&>());
}
