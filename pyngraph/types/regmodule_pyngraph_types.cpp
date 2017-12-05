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
#include "pyngraph/types/regmodule_pyngraph_types.hpp"

namespace py = pybind11;

void regmodule_pyngraph_types(py::module m){
    regclass_pyngraph_ValueType(m);
    regclass_pyngraph_TensorViewType(m);
    regclass_pyngraph_Type(m);
    regclass_pyngraph_Bool(m);
    regclass_pyngraph_Float32(m);
    regclass_pyngraph_Float64(m);
    regclass_pyngraph_Int8(m);
    regclass_pyngraph_Int16(m);
    regclass_pyngraph_Int32(m);
    regclass_pyngraph_Int64(m);
    regclass_pyngraph_UInt8(m);
    regclass_pyngraph_UInt16(m);
    regclass_pyngraph_UInt32(m);
    regclass_pyngraph_UInt64(m);
}
