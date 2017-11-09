#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include "ngraph/ops/parameter.hpp"
#include "ngraph/node.hpp"
#include "ngraph/ops/add.hpp"

namespace py = pybind11;
namespace ngraph {
namespace op {

PYBIND11_PLUGIN(clsParameter) {

    py::module mod("clsParameter");

    py::module::import("clsTraitedType");
    py::class_<Parameter, std::shared_ptr<Parameter>> clsParameter(mod, "clsParameter");

    clsParameter.def(py::init<const ngraph::element::Type&, const ngraph::Shape& >());
    clsParameter.def("description", &Parameter::description);
    //clsParameter.def(py::self + py::self);

    clsParameter.def("__add__", [](const Parameter &a, const Parameter &b) {
      return
             std::shared_ptr<ngraph::Node>(*const_cast<Node*>(reinterpret_cast<const Node*>(&a))) +
             std::shared_ptr<ngraph::Node>(*const_cast<Node*>(reinterpret_cast<const Node*>(&b)));
    }, py::is_operator());

    return mod.ptr();

}

}}  // ngraph
