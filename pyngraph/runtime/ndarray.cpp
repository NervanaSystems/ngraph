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
//#include <pybind11/stl.h>
//#include <string>
#include "ngraph/runtime/ndarray.hpp"
#include "pyngraph/runtime/ndarray.hpp"

namespace py = pybind11;

void regclass_pyngraph_runtime_NDArray(py::module m) {
    // TODO: implement later
    /*
    using ZeroDArrayFloat32 = NDArray< >;
    using OneDArrayFloat32 = NDArray< >;
    using TwoDArrayFloat32 = NDArray< >;
    using TreeDArrayFloat32 = NDArray< >;
    using ZeroDArrayInt32 = NDArray< >;
    using OneDArrayInt32 = NDArray< >;
    using TwoDArrayInt32 = NDArray< >;
    using TreeDArrayInt32 = NDArray< >;
    py::class_<ZeroDArrayFloat32, std::shared_ptr<ZeroDArrayFloat32>, ZeroDArrayBaseFloat32> clsNDArray(m, "ZeroDArrayFloat32");
    */
}
