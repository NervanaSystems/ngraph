#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include "ngraph/runtime/manager.hpp"
#include "ngraph/runtime/ngvm/ngvm_manager.hpp"

namespace py = pybind11;
namespace ngraph {
namespace runtime {

PYBIND11_PLUGIN(clsManager) {

    py::module mod("clsNGVMManager");
    py::module mod_1("clsManager");

    py::class_<Manager, std::shared_ptr<Manager>> clsManager(mod_1, "clsManager");
    py::class_<ngvm::NGVMManager, std::shared_ptr<ngvm::NGVMManager>, Manager> clsNGVMManager(mod, "clsNGVMManager"); 
    clsManager.def("get", &Manager::get);

    return mod.ptr();

}

}}  // ngraph
