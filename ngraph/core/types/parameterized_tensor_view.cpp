#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include "ngraph/runtime/parameterized_tensor_view.hpp"

namespace py = pybind11;
namespace ngraph {
namespace runtime {
namespace {

template <typename ET>
static void declareParameterizedTensorView(py::module & mod, std::string const & suffix) {
    using Class = ParameterizedTensorView<ET>;
    using PyClass = py::class_<Class, std::shared_ptr<Class>, TensorView>;

    PyClass cls(mod, ("ParameterizedTensorView" + suffix).c_str());
}

}

PYBIND11_PLUGIN(clsParameterizedTensorView) {

    py::module mod1("clsTensorView");
    py::class_<TensorView, std::shared_ptr<TensorView>> clsTensorView(mod1, "clsTensorView");
    py::module mod("clsParameterizedTensorView");
    py::module::import("clsTraitedType");

    declareParameterizedTensorView<ngraph::element::TraitedType<float>>(mod, "F");
    //declareParameterizedTensorView<double>(mod, "D");
    //declareParameterizedTensorView<int>(mod, "I");

    return mod.ptr();
}

}}  // ngraph
