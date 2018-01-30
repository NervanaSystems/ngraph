#include <pybind11/pybind11.h>
#include "pyngraph/node.hpp"
#include "pyngraph/util.hpp"
#include "pyngraph/function.hpp"
#include "pyngraph/ops/regmodule_pyngraph_op.hpp"
#include "pyngraph/runtime/regmodule_pyngraph_runtime.hpp"
#include "pyngraph/types/regmodule_pyngraph_types.hpp"


namespace py = pybind11;

PYBIND11_MODULE(_pyngraph, m){
    m.doc() = "pyngraph plugin";
    regclass_pyngraph_Node(m);
    regmodule_pyngraph_types(m);
    regclass_pyngraph_Function(m);
    regmodule_pyngraph_op(m);
    regmodule_pyngraph_runtime(m);
    regmodule_pyngraph_util(m);
}
