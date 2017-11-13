#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include "ngraph/runtime/external_function.hpp"

namespace py = pybind11;
namespace ngraph {
namespace runtime {

PYBIND11_PLUGIN(clsExternalFunction) {

    py::module mod("clsExternalFunction");

    py::class_<ExternalFunction, std::shared_ptr<ExternalFunction>> clsExternalFunction(mod, "clsExternalFunction");
    

    return mod.ptr();

}

}}  // ngraph
