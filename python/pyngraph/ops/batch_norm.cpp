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

#include "ngraph/op/batch_norm.hpp"
#include "pyngraph/ops/batch_norm.hpp"

namespace py = pybind11;

void regclass_pyngraph_op_BatchNorm(py::module m)
{
    py::class_<ngraph::op::BatchNorm,
               std::shared_ptr<ngraph::op::BatchNorm>,
               ngraph::op::util::RequiresTensorViewArgs>
        batch_norm(m, "BatchNorm");
    batch_norm.doc() = "ngraph.impl.op.BatchNorm wraps ngraph::op::BatchNorm";
    batch_norm.def(py::init<double,
                            const std::shared_ptr<ngraph::Node>&,
                            const std::shared_ptr<ngraph::Node>&,
                            const std::shared_ptr<ngraph::Node>&>());
}

void regclass_pyngraph_op_BatchNormBackprop(py::module m)
{
    py::class_<ngraph::op::BatchNormBackprop,
               std::shared_ptr<ngraph::op::BatchNormBackprop>,
               ngraph::op::util::RequiresTensorViewArgs>
        batch_norm_backprop(m, "BatchNormBackprop");
    batch_norm_backprop.doc() =
        "ngraph.impl.op.BatchNormBackprop wraps ngraph::op::BatchNormBackprop";
    batch_norm_backprop.def(py::init<double,
                                     const std::shared_ptr<ngraph::Node>&,
                                     const std::shared_ptr<ngraph::Node>&,
                                     const std::shared_ptr<ngraph::Node>&,
                                     const std::shared_ptr<ngraph::Node>&,
                                     const std::shared_ptr<ngraph::Node>&,
                                     const std::shared_ptr<ngraph::Node>&>());
}
