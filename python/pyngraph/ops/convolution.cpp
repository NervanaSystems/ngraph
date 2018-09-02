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

#include "ngraph/op/convolution.hpp"
#include "ngraph/shape.hpp"
#include "pyngraph/ops/convolution.hpp"

namespace py = pybind11;

void regclass_pyngraph_op_Convolution(py::module m)
{
    py::class_<ngraph::op::Convolution, std::shared_ptr<ngraph::op::Convolution>, ngraph::op::Op>
        convolution(m, "Convolution");
    convolution.doc() = "ngraph.impl.op.Convolution wraps ngraph::op::Convolution";
    convolution.def(py::init<const std::shared_ptr<ngraph::Node>&,
                             const std::shared_ptr<ngraph::Node>&,
                             const ngraph::Strides&,
                             const ngraph::Strides&,
                             const ngraph::CoordinateDiff&,
                             const ngraph::CoordinateDiff&,
                             const ngraph::Strides&>());

    convolution.def(py::init<const std::shared_ptr<ngraph::Node>&,
                             const std::shared_ptr<ngraph::Node>&,
                             const ngraph::Strides&,
                             const ngraph::Strides&,
                             const ngraph::CoordinateDiff&,
                             const ngraph::CoordinateDiff&>());

    convolution.def(py::init<const std::shared_ptr<ngraph::Node>&,
                             const std::shared_ptr<ngraph::Node>&,
                             const ngraph::Strides&,
                             const ngraph::Strides&>());

    convolution.def(py::init<const std::shared_ptr<ngraph::Node>&,
                             const std::shared_ptr<ngraph::Node>&,
                             const ngraph::Strides&>());

    convolution.def(
        py::init<const std::shared_ptr<ngraph::Node>&, const std::shared_ptr<ngraph::Node>&>());
}

void regclass_pyngraph_op_ConvolutionBackpropData(py::module m)
{
    py::class_<ngraph::op::ConvolutionBackpropData,
               std::shared_ptr<ngraph::op::ConvolutionBackpropData>,
               ngraph::op::Op>
        convolutionBackpropData(m, "ConvolutionBackpropData");

    convolutionBackpropData.def(py::init<const ngraph::Shape&,
                                         const std::shared_ptr<ngraph::Node>&,
                                         const std::shared_ptr<ngraph::Node>&,
                                         const ngraph::Strides&,
                                         const ngraph::Strides&,
                                         const ngraph::CoordinateDiff&,
                                         const ngraph::CoordinateDiff&,
                                         const ngraph::Strides&>());
}

void regclass_pyngraph_op_ConvolutionBackpropFilters(py::module m)
{
    py::class_<ngraph::op::ConvolutionBackpropFilters,
               std::shared_ptr<ngraph::op::ConvolutionBackpropFilters>,
               ngraph::op::Op>
        convolutionBackpropFilters(m, "ConvolutionBackpropFilters");

    convolutionBackpropFilters.def(py::init<const std::shared_ptr<ngraph::Node>&,
                                            const ngraph::Shape&,
                                            const std::shared_ptr<ngraph::Node>&,
                                            const ngraph::Strides&,
                                            const ngraph::Strides&,
                                            const ngraph::CoordinateDiff&,
                                            const ngraph::CoordinateDiff&,
                                            const ngraph::Strides&>());
}
