#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include "ngraph/runtime/call_frame.hpp"

namespace py = pybind11;
namespace ngraph {
namespace runtime {

PYBIND11_PLUGIN(clsCallFrame) {

    py::module mod("clsCallFrame");

    py::class_<CallFrame, std::shared_ptr<CallFrame>> clsCallFrame(mod, "clsCallFrame");
    return mod.ptr();

}

}}  // ngraph

