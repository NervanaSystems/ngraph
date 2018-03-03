#include <pybind11/pybind11.h>
#include "pyngraph/node.hpp"
#include "pyngraph/node_vector.hpp"
#include "pyngraph/shape.hpp" 
#include "pyngraph/util.hpp"
#include "pyngraph/function.hpp"
#include "pyngraph/serializer.hpp"
#include "pyngraph/ops/regmodule_pyngraph_op.hpp"
#include "pyngraph/ops/util/regmodule_pyngraph_op_util.hpp"
#include "pyngraph/runtime/regmodule_pyngraph_runtime.hpp"
#include "pyngraph/types/regmodule_pyngraph_types.hpp"
#include "pyngraph/ops/op.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_pyngraph, m){
    m.doc() = "pyngraph plugin";
    regclass_pyngraph_Node(m);
    regclass_pyngraph_NodeVector(m);
    regclass_pyngraph_Shape(m);
    regmodule_pyngraph_types(m);
    regclass_pyngraph_Function(m);
    regclass_pyngraph_Serializer(m);
    py::module m_op = m.def_submodule("op", "module pyngraph.op");
    regclass_pyngraph_op_Op(m_op);
    regmodule_pyngraph_op_util(m);
    regmodule_pyngraph_op(m_op);
    regmodule_pyngraph_runtime(m);
    regmodule_pyngraph_util(m);
}
