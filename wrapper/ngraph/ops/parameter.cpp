#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <string>
#include "ngraph/node.hpp"
#include "ngraph/ops/parameter.hpp"
#include "ngraph/ops/add.hpp"
#include "ngraph/ops/multiply.hpp"

namespace py = pybind11;
namespace ngraph {

PYBIND11_PLUGIN(clsParameter) {

    py::module mod_1("clsNode");
    py::module mod("clsParameter");

    py::module::import("wrapper.ngraph.types.clsTraitedType");
    py::class_<Node, std::shared_ptr<Node>> clsNode(mod_1, "clsNode");
    py::class_<op::Parameter, std::shared_ptr<op::Parameter>, Node> clsParameter(mod, "clsParameter");

    clsParameter.def(py::init<const ngraph::element::Type&, const ngraph::Shape& >());
    clsParameter.def("description", &op::Parameter::description);
    clsNode.def("__add__", [](const std::shared_ptr<ngraph::Node>& a, const std::shared_ptr<ngraph::Node> b) {
                return a + b;
               }, py::is_operator()); 
    clsNode.def("__mul__", [](const std::shared_ptr<ngraph::Node>& a, const std::shared_ptr<ngraph::Node> b) {
                return a * b;
               }, py::is_operator());

    return mod.ptr();

}

}  // ngraph
