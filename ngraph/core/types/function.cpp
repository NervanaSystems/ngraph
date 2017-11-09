#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include "ngraph/function.hpp"

namespace py = pybind11;
namespace ngraph {

PYBIND11_PLUGIN(clsFunction) {

    py::module mod("clsFunction");

    py::module::import("clsParameter");
    py::module::import("clsTensorViewType");
    py::class_<Function, std::shared_ptr<Function>> clsFunction(mod, "clsFunction");

    clsFunction.def(py::init<const std::shared_ptr<op::Parameter>&, const std::shared_ptr<const TensorViewType>&,
                             const std::vector<std::shared_ptr<op::Parameter>>&, const std::string&>());
    clsFunction.def("get_result_type", &Function::get_result_type);

    return mod.ptr();

}

}  // ngraph
