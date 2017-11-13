#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include "ngraph/runtime/manager.hpp"
#include "ngraph/runtime/ngvm/ngvm_manager.hpp"
#include "ngraph/function.hpp"
#include "ngraph/runtime/external_function.hpp"
#include "ngraph/runtime/backend.hpp"

namespace py = pybind11;
namespace ngraph {
namespace runtime {

PYBIND11_PLUGIN(clsManager) {

    py::module::import("clsFunction");
    py::module::import("clsExternalFunction");
    py::module::import("clsBackend");
    py::module mod("clsManager");
    py::module mod_1("clsNGVMManager");

    py::class_<Manager, std::shared_ptr<Manager>> clsManager(mod, "clsManager");
    py::class_<ngvm::NGVMManager, std::shared_ptr<ngvm::NGVMManager>, Manager> clsNGVMManager(mod_1, "clsNGVMManager");
    clsManager.def_static("get", &Manager::get);
    clsManager.def("compile", &Manager::compile);
    clsManager.def("allocate_backend", &Manager::allocate_backend);

    return mod.ptr();

}

}}  // ngraph
