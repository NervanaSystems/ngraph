#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/ngvm/ngvm_backend.hpp"

namespace py = pybind11;
namespace ngraph {
namespace runtime {

PYBIND11_PLUGIN(clsBackend) {

    py::module mod("clsNGVMBackend");
    py::module mod_1("clsBackend");

    py::class_<Backend, std::shared_ptr<Backend>> clsBackend(mod_1, "clsBackend");
    py::class_<ngvm::NGVMBackend, std::shared_ptr<ngvm::NGVMBackend>, Backend> clsNGVMBackend(mod, "clsNGVMBackend");

    return mod.ptr();

}

}}  // ngraph
