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

#include <pybind11/numpy.h>
#include "pyngraph/util.hpp"

namespace py = pybind11;

class Util {
    void* numpy_to_c(py::array a) {
        py::buffer_info info = a.request();
        return info.ptr;
    }
}

void regclass_pyngraph_Util(py::module m) {
    py::class_<Util, std::shared_ptr<Util>> util(m, "Util");
    module mod = m.def_submodule("Util", "pyngraph.Util")
    util.def("numpy_to_c", &Util::numpy_to_c);
}
