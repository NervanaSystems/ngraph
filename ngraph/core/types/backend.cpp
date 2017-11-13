#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/call_frame.hpp"
#include "ngraph/runtime/external_function.hpp"
#include "ngraph/runtime/ngvm/ngvm_backend.hpp"

namespace py = pybind11;
namespace ngraph {
namespace runtime {

PYBIND11_PLUGIN(clsBackend) {

    py::module mod("clsBackend");
    py::module::import("clsCallFrame");

    py::class_<Backend, std::shared_ptr<Backend>> clsBackend(mod, "clsBackend");
    clsBackend.def("make_call_frame", &Backend::make_call_frame);
    return mod.ptr();

}

}}  // ngraph
