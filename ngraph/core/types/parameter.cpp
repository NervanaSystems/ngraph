#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include "ngraph/ops/parameter.hpp"

namespace py = pybind11;
namespace ngraph {
namespace op {

PYBIND11_PLUGIN(clsParameter) {

    py::module mod("clsParameter");
//    py::class_<Node, std::shared_ptr<Node>> clsNode(mod, "clsNode");

    py::module::import("TraitedType");
    py::class_<Parameter, std::shared_ptr<Parameter>> clsParameter(mod, "clsParameter");

    clsParameter.def(py::init<const ngraph::element::Type&, const ngraph::Shape& >());
    clsParameter.def("description", &Parameter::description);

    return mod.ptr();

}

}}  // ngraph
