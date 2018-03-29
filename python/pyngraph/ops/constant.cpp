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
//#include <string>
#include "ngraph/op/constant.hpp"
#include "pyngraph/ops/constant.hpp"

namespace py = pybind11;

void regclass_pyngraph_op_Constant(py::module m)
{
    py::class_<ngraph::op::Constant, std::shared_ptr<ngraph::op::Constant>, ngraph::Node> constant(
        m, "Constant");
    constant.doc() = "ngraph.impl.op.Constant wraps ngraph::op::Constant";
    constant.def(
        py::init<const ngraph::element::Type&, const ngraph::Shape&, const std::vector<char>&>());
    constant.def(
        py::init<const ngraph::element::Type&, const ngraph::Shape&, const std::vector<float>&>());
    constant.def(
        py::init<const ngraph::element::Type&, const ngraph::Shape&, const std::vector<double>&>());
    constant.def(
        py::init<const ngraph::element::Type&, const ngraph::Shape&, const std::vector<int8_t>&>());
    constant.def(py::init<const ngraph::element::Type&,
                          const ngraph::Shape&,
                          const std::vector<int16_t>&>());
    constant.def(py::init<const ngraph::element::Type&,
                          const ngraph::Shape&,
                          const std::vector<int32_t>&>());
    constant.def(py::init<const ngraph::element::Type&,
                          const ngraph::Shape&,
                          const std::vector<int64_t>&>());
    constant.def(py::init<const ngraph::element::Type&,
                          const ngraph::Shape&,
                          const std::vector<uint8_t>&>());
    constant.def(py::init<const ngraph::element::Type&,
                          const ngraph::Shape&,
                          const std::vector<uint16_t>&>());
    constant.def(py::init<const ngraph::element::Type&,
                          const ngraph::Shape&,
                          const std::vector<uint32_t>&>());
    constant.def(py::init<const ngraph::element::Type&,
                          const ngraph::Shape&,
                          const std::vector<uint64_t>&>());
}
