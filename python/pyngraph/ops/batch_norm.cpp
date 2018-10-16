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

#include "ngraph/op/batch_norm.hpp"
#include "pyngraph/ops/batch_norm.hpp"

namespace py = pybind11;

void regclass_pyngraph_op_BatchNormTraining(py::module m)
{
    py::class_<ngraph::op::BatchNormTraining,
               std::shared_ptr<ngraph::op::BatchNormTraining>,
               ngraph::op::Op>
        batch_norm_training(m, "BatchNormTraining");
    batch_norm_training.doc() =
        "ngraph.impl.op.BatchNormTraining wraps ngraph::op::BatchNormTraining";
    batch_norm_training.def(py::init<double,
                                     const std::shared_ptr<ngraph::Node>&,
                                     const std::shared_ptr<ngraph::Node>&,
                                     const std::shared_ptr<ngraph::Node>&>());
}

void regclass_pyngraph_op_BatchNormInference(py::module m)
{
    py::class_<ngraph::op::BatchNormInference,
               std::shared_ptr<ngraph::op::BatchNormInference>,
               ngraph::op::Op>
        batch_norm_inference(m, "BatchNormInference");
    batch_norm_inference.doc() =
        "ngraph.impl.op.BatchNormInference wraps ngraph::op::BatchNormInference";

    batch_norm_inference.def(py::init<double,
                                      const std::shared_ptr<ngraph::Node>&,
                                      const std::shared_ptr<ngraph::Node>&,
                                      const std::shared_ptr<ngraph::Node>&,
                                      const std::shared_ptr<ngraph::Node>&,
                                      const std::shared_ptr<ngraph::Node>&>());
}

void regclass_pyngraph_op_BatchNormTrainingBackprop(py::module m)
{
    py::class_<ngraph::op::BatchNormTrainingBackprop,
               std::shared_ptr<ngraph::op::BatchNormTrainingBackprop>,
               ngraph::op::Op>
        batch_norm_training_backprop(m, "BatchNormTrainingBackprop");
    batch_norm_training_backprop.doc() =
        "ngraph.impl.op.BatchNormTrainingBackprop wraps ngraph::op::BatchNormTrainingBackprop";
    batch_norm_training_backprop.def(py::init<double,
                                              const std::shared_ptr<ngraph::Node>&,
                                              const std::shared_ptr<ngraph::Node>&,
                                              const std::shared_ptr<ngraph::Node>&,
                                              const std::shared_ptr<ngraph::Node>&,
                                              const std::shared_ptr<ngraph::Node>&,
                                              const std::shared_ptr<ngraph::Node>&>());
}
