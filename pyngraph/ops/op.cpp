// ----------------------------------------------------------------------------
// Copyright 2017 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
//#include <string>
#include "ngraph/ops/op.hpp"
#include "pyngraph/ops/op.hpp"

namespace py = pybind11;


//    py::module::import("wrapper.ngraph.Node");

void regclass_pyngraph_op_RequiresTensorViewArgs(py::module m){
    py::class_<ngraph::op::RequiresTensorViewArgs, std::shared_ptr<ngraph::op::RequiresTensorViewArgs>,
        ngraph::Node> requiresTensorViewArgs(m, "RequiresTensorViewArgs");
}

void regclass_pyngraph_op_UnaryElementwise(py::module m){
    py::class_<ngraph::op::UnaryElementwise, std::shared_ptr<ngraph::op::UnaryElementwise>,
        ngraph::op::RequiresTensorViewArgs> unaryElementwise(m, "UnaryElementwise");
}
void regclass_pyngraph_op_UnaryElementwiseArithmetic(py::module m){
    py::class_<ngraph::op::UnaryElementwiseArithmetic, std::shared_ptr<ngraph::op::UnaryElementwiseArithmetic>,
        ngraph::op::UnaryElementwise> unaryElementwiseArithmetic(m, "UnaryElementwiseArithmetic");
}
void regclass_pyngraph_op_BinaryElementwise(py::module m){
    py::class_<ngraph::op::BinaryElementwise, std::shared_ptr<ngraph::op::BinaryElementwise>,
        ngraph::op::RequiresTensorViewArgs> binaryElementwise(m, "BinaryElementwise");
}
void regclass_pyngraph_op_BinaryElementwiseComparison(py::module m){
    py::class_<ngraph::op::BinaryElementwiseComparison, std::shared_ptr<ngraph::op::BinaryElementwiseComparison>,
        ngraph::op::BinaryElementwise> binaryElementwiseComparison(m, "BinaryElementwiseComparison");
}
void regclass_pyngraph_op_BinaryElementwiseArithmetic(py::module m){
    py::class_<ngraph::op::BinaryElementwiseArithmetic, std::shared_ptr<ngraph::op::BinaryElementwiseArithmetic>,
        ngraph::op::BinaryElementwise> binaryElementwiseArithmetic(m, "BinaryElementwiseArithmetic");
}


