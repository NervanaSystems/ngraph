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

#include "pyngraph/axis_set.hpp"
#include "pyngraph/axis_vector.hpp"
#include "pyngraph/coordinate.hpp"
#include "pyngraph/coordinate_diff.hpp"
#include "pyngraph/function.hpp"
#include "pyngraph/node.hpp"
#include "pyngraph/node_vector.hpp"
#include "pyngraph/onnx_import/onnx_import.hpp"
#include "pyngraph/ops/op.hpp"
#include "pyngraph/ops/regmodule_pyngraph_op.hpp"
#include "pyngraph/ops/util/regmodule_pyngraph_op_util.hpp"
#include "pyngraph/passes/regmodule_pyngraph_passes.hpp"
#include "pyngraph/runtime/regmodule_pyngraph_runtime.hpp"
#include "pyngraph/serializer.hpp"
#include "pyngraph/shape.hpp"
#include "pyngraph/strides.hpp"
#include "pyngraph/types/regmodule_pyngraph_types.hpp"
#include "pyngraph/util.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_pyngraph, m)
{
    m.doc() = "Package ngraph.impl that wraps nGraph's namespace ngraph";
    regclass_pyngraph_Node(m);
    regclass_pyngraph_NodeVector(m);
    regclass_pyngraph_Shape(m);
    regclass_pyngraph_Strides(m);
    regclass_pyngraph_CoordinateDiff(m);
    regclass_pyngraph_AxisSet(m);
    regclass_pyngraph_AxisVector(m);
    regclass_pyngraph_Coordinate(m);
    regmodule_pyngraph_types(m);
    regclass_pyngraph_Function(m);
    regclass_pyngraph_Serializer(m);
    py::module m_onnx_import = m.def_submodule("onnx_import",
                                               "Package ngraph.impl.onnx_import "
                                               "that wraps ngraph::onnx_import");
    regmodule_pyngraph_onnx_import(m_onnx_import);
    py::module m_op = m.def_submodule("op", "Package ngraph.impl.op that wraps ngraph::op");
    regclass_pyngraph_op_Op(m_op);
    regmodule_pyngraph_op_util(m_op);
    regmodule_pyngraph_op(m_op);
    regmodule_pyngraph_runtime(m);
    regmodule_pyngraph_passes(m);
    regmodule_pyngraph_util(m);
}
