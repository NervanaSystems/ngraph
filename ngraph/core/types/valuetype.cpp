#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include "ngraph/types/type.hpp"

namespace py = pybind11;
namespace ngraph {

PYBIND11_PLUGIN(clsTensorViewType) {

    py::module mod("clsTensorViewType");
    py::module mod_1("clsValueType");
    py::class_<ValueType, std::shared_ptr<ValueType>> clsValueType(mod_1, "clsValueType"); 
    py::class_<TensorViewType, std::shared_ptr<TensorViewType>, ValueType> clsTensorViewType(mod, "clsTensorViewType");

    clsTensorViewType.def(py::init<const element::Type&, const Shape&>());
    clsTensorViewType.def("get_shape", &TensorViewType::get_shape);

    return mod.ptr();

}

}  // ngraph
