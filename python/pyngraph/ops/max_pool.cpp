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

#include "ngraph/op/max_pool.hpp"
#include "ngraph/shape.hpp"
#include "pyngraph/ops/max_pool.hpp"

namespace py = pybind11;

void regclass_pyngraph_op_MaxPool(py::module m)
{
    py::class_<ngraph::op::MaxPool, std::shared_ptr<ngraph::op::MaxPool>, ngraph::op::Op> max_pool(
        m, "MaxPool");
    max_pool.def(py::init<const std::shared_ptr<ngraph::Node>&,
                          const ngraph::Shape&,
                          const ngraph::Strides&,
                          const ngraph::Shape&,
                          const ngraph::Shape&>());
    max_pool.def(py::init<const std::shared_ptr<ngraph::Node>&,
                          const ngraph::Shape&,
                          const ngraph::Strides&>());
    max_pool.def(py::init<const std::shared_ptr<ngraph::Node>&, const ngraph::Shape&>());
}

void regclass_pyngraph_op_MaxPoolBackprop(py::module m)
{
    py::class_<ngraph::op::MaxPoolBackprop,
               std::shared_ptr<ngraph::op::MaxPoolBackprop>,
               ngraph::op::Op>
        max_pool_backprop(m, "MaxPoolBackprop");
    max_pool_backprop.doc() = "ngraph.impl.op.MaxPoolBackprop wraps ngraph::op::MaxPoolBackprop";
    max_pool_backprop.def(py::init<const std::shared_ptr<ngraph::Node>&,
                                   const std::shared_ptr<ngraph::Node>&,
                                   const ngraph::Shape&,
                                   const ngraph::Strides&,
                                   const ngraph::Shape&,
                                   const ngraph::Shape&,
                                   const std::shared_ptr<ngraph::op::MaxPool>&>());
}
