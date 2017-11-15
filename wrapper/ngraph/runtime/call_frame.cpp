#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include "ngraph/runtime/call_frame.hpp"
#include "ngraph/runtime/ngvm/call_frame.hpp" 

namespace py = pybind11;
namespace ngraph {
namespace runtime {

PYBIND11_PLUGIN(clsCallFrame) {

    py::module mod("clsCallFrame");
    py::module mod_1("clsNGVMCallFrame");

    py::class_<CallFrame, std::shared_ptr<CallFrame>> clsCallFrame(mod, "clsCallFrame");
    py::class_<ngvm::CallFrame, std::shared_ptr<ngvm::CallFrame>, CallFrame> clsNGVMCallFrame(mod_1, "clsNGVMCallFrame");
    clsCallFrame.def("call", [](const std::vector<std::shared_ptr<ngraph::runtime::Value>>& inputs,
                              const std::vector<std::shared_ptr<ngraph::runtime::Value>>& outputs) {
                      (inputs, outputs);
                     }, py::is_operator());
    return mod.ptr();

}

}}  // ngraph

