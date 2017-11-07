#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include "ngraph/types/element_type.hpp"

namespace py = pybind11;
namespace ngraph {
namespace element {
namespace {

template <typename T>
static void declareTraitedType(py::module & mod, std::string const & suffix) {
    using Class = TraitedType<T>;
    using PyClass = py::class_<Class, std::shared_ptr<Class>, Type>;

    PyClass cls(mod, ("TraitedType" + suffix).c_str());

    cls.def(py::init<>());
    cls.def_static("element_type", &Class::element_type,
                   py::return_value_policy::reference);
    cls.def_static("read", (T (*) (const std::string&)) &Class::read);
    cls.def_static("read", (std::vector<T> (*) (const std::vector<std::string>&)) &Class::read);
    cls.def("make_primary_tensor_view", &Class::make_primary_tensor_view);
}

}

PYBIND11_PLUGIN(TraitedType) {

    py::module mod1("clsType");
    py::class_<Type, std::shared_ptr<Type>> clsType(mod1, "clsType");
    py::module mod("TraitedType");

    declareTraitedType<float>(mod, "F");
    declareTraitedType<double>(mod, "D");
    declareTraitedType<int>(mod, "I");

    return mod.ptr();
}

}}  // ngraph
