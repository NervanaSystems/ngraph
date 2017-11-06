#include <pybind11/pybind11.h>
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

//    cls.def(py::init<>());
    cls.def("elementtype", &Class::element_type);
}

}

PYBIND11_PLUGIN(traitedType) {

    py::module mod1("clsType");
    py::class_<Type, std::shared_ptr<Type>> clsType(mod1, "clsType");
    py::module mod("traitedType");

    declareTraitedType<float>(mod, "F");
    //declareTraitedType<double>(mod, "D");

    return mod.ptr();
}

}}  // ngraph
