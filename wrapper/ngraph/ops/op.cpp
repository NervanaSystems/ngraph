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
#include <string>
#include "ngraph/ops/op.hpp"

namespace py = pybind11;
namespace ngraph {
namespace op {

PYBIND11_MODULE(clsOp, mod) {

    py::module::import("wrapper.ngraph.clsNode");

    py::class_<Builtin, std::shared_ptr<Builtin>, Node> clsBuiltin(mod, "Builtin");
    py::class_<UnaryElementwiseBuiltin, std::shared_ptr<UnaryElementwiseBuiltin>,
        Builtin> clsUnaryElementwiseBuiltin(mod, "UnaryElementwiseBuiltin");
    py::class_<UnaryElementwiseArithmetic, std::shared_ptr<UnaryElementwiseArithmetic>, 
        UnaryElementwiseBuiltin> clsUnaryElementwiseArithmetic(mod, "UnaryElementwiseArithmetic"); 
    py::class_<BinaryElementwiseBuiltin, std::shared_ptr<BinaryElementwiseBuiltin>,
        Builtin> clsBinaryElementwiseBuiltin(mod, "BinaryElementwiseBuiltin");
    py::class_<BinaryElementwiseArithmetic, std::shared_ptr<BinaryElementwiseArithmetic>,
        BinaryElementwiseBuiltin> clsBinaryElementwiseArithmetic(mod, "BinaryElementwiseArithmetic");    

}

}}  // ngraph
