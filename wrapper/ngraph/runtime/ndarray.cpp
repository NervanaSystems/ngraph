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
#include "ngraph/runtime/ndarray.hpp"

namespace py = pybind11;
namespace ngraph {
namespace runtime {

template <typename T>
static void declareNDArrayBase(py::module & mod, std::string const & suffix) {
    using NDArrayBaseClass = NDArrayBase<T>;

    py::class_<NDArrayBaseClass, std::shared_ptr<NDArrayBaseClass>> clsNDArrayBase(mod,
                                                             ("clsNDArrayBase" + suffix).c_str());

    clsNDArrayBase.def("get_shape", &NDArrayBaseClass::get_shape);
    clsNDArrayBase.def("begin", &NDArrayBaseClass::begin);
    clsNDArrayBase.def("end", &NDArrayBaseClass::end);
    //clsNDArrayBase.def("get_vector", &NDArrayBaseClass::get_vector);
}

template <typename T, size_t N>
static void declareNDArray(py::module & mod, std::string const & suffix) {
    using NDArrayBaseClass = NDArrayBase<T>;
    using NDArrayClass = NDArray<T, N>;

    py::class_<NDArrayClass, std::shared_ptr<NDArrayClass>, NDArrayBaseClass> clsNDArray(mod,
                                                      ("clsNDArray" + suffix + std::to_string(N)).c_str());
    //clsNDArray.def(py::init<std::vector
}

PYBIND11_PLUGIN(clsNDArray) {

    py::module clsNDArrayBase("clsNDArrayBase");
    declareNDArrayBase<float>(clsNDArrayBase, "F");
    declareNDArrayBase<double>(clsNDArrayBase, "D");
    declareNDArrayBase<int>(clsNDArrayBase, "I");

    py::module clsNDArray("clsNDArray");
    declareNDArray<float, 1>(clsNDArrayBase, "F");
    declareNDArray<double, 1>(clsNDArrayBase, "D");
    declareNDArray<int, 1>(clsNDArrayBase, "I");
}
}}  // ngraph
