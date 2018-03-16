#include <pybind11/pybind11.h>
#include "pyngraph/axis_set.hpp"
#include "pyngraph/axis_vector.hpp"
#include "pyngraph/coordinate.hpp"
#include "pyngraph/coordinate_diff.hpp"
#include "pyngraph/function.hpp"
#include "pyngraph/node.hpp"
#include "pyngraph/node_vector.hpp"
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
    m.doc() = "Package ngraph that wraps nGraph's namespace ngraph";
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
    py::module m_op = m.def_submodule("op", "Package ngraph.op that wraps ngraph::op");
    regclass_pyngraph_op_Op(m_op);
    regmodule_pyngraph_op_util(m_op);
    regmodule_pyngraph_op(m_op);
    regmodule_pyngraph_runtime(m);
    regmodule_pyngraph_passes(m);
    regmodule_pyngraph_util(m);
}
