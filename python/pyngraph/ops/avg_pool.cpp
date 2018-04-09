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

#include "ngraph/op/avg_pool.hpp"
#include "ngraph/shape.hpp"
#include "pyngraph/ops/avg_pool.hpp"

namespace py = pybind11;

void regclass_pyngraph_op_AvgPool(py::module m)
{
    py::class_<ngraph::op::AvgPool,
               std::shared_ptr<ngraph::op::AvgPool>,
               ngraph::op::util::RequiresTensorViewArgs>
        avg_pool(m, "AvgPool");
    avg_pool.doc() = "ngraph.impl.op.AvgPool wraps ngraph::op::AvgPool";
    avg_pool.def(py::init<const std::shared_ptr<ngraph::Node>&,
                          const ngraph::Shape&,
                          const ngraph::Strides&,
                          const ngraph::Shape&,
                          const ngraph::Shape&,
                          bool>());
    avg_pool.def(py::init<const std::shared_ptr<ngraph::Node>&,
                          const ngraph::Shape&,
                          const ngraph::Strides&>());
    avg_pool.def(py::init<const std::shared_ptr<ngraph::Node>&, const ngraph::Shape&>());
}

void regclass_pyngraph_op_AvgPoolBackprop(py::module m)
{
    py::class_<ngraph::op::AvgPoolBackprop,
               std::shared_ptr<ngraph::op::AvgPoolBackprop>,
               ngraph::op::util::RequiresTensorViewArgs>
        avg_pool_backprop(m, "AvgPoolBackprop");
    avg_pool_backprop.doc() = "ngraph.impl.op.AvgPoolBackprop wraps ngraph::op::AvgPoolBackprop";
    avg_pool_backprop.def(py::init<const ngraph::Shape&,
                                   const std::shared_ptr<ngraph::Node>&,
                                   const ngraph::Shape&,
                                   const ngraph::Strides&,
                                   const ngraph::Shape&,
                                   const ngraph::Shape&,
                                   bool>());
}
