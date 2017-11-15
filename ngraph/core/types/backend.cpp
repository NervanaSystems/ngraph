#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/call_frame.hpp"
#include "ngraph/runtime/external_function.hpp"
#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/ngvm/ngvm_backend.hpp"
#include "ngraph/runtime/parameterized_tensor_view.hpp"

namespace py = pybind11;
namespace ngraph {
namespace runtime {

PYBIND11_PLUGIN(clsBackend) {

    py::module mod("clsBackend");
    py::module::import("clsCallFrame");
    py::module::import("clsParameterizedTensorView");
    //py::module::import("clsNDArrayBase");
    using ET = ngraph::element::TraitedType<float>;

    py::class_<Backend, std::shared_ptr<Backend>> clsBackend(mod, "clsBackend");
    clsBackend.def("make_call_frame", &Backend::make_call_frame);
    clsBackend.def("make_primary_tensor_view",
                   &Backend::make_primary_tensor_view);
    clsBackend.def("make_parameterized_tensor_view", (std::shared_ptr<ParameterizedTensorView<ET>> (Backend::*) (const ngraph::Shape& )) &Backend::make_parameterized_tensor_view);
    clsBackend.def("make_parameterized_tensor_view", (std::shared_ptr<ParameterizedTensorView<ET>> (Backend::*) (const NDArrayBase<ET::type>& )) &Backend::make_parameterized_tensor_view);
    return mod.ptr();

}

}}  // ngraph
