#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include "ngraph/runtime/ndarray.hpp"

namespace py = pybind11;
namespace ngraph {
namespace runtime {

template <typename T>
static void declareNDArrayBase(py::module & mod, std::string const & suffix) {
    using NDArrayBaseClass = NDArrayBase<T>;

    py::class_<NDArrayBaseClass, std::shared_ptr<NDArrayBaseClass>> clsNDArrayBase(mod,
                                                             ("clsNDArrayBase" + suffix).c_str());

    clsNDArrayBase.def("get_shape", &NDArrayBaseClass::get_shape);
    clsNDArrayBase.def("begin", &NDArrayBaseClass::begin);
    clsNDArrayBase.def("end", &NDArrayBaseClass::end);
    //clsNDArrayBase.def("get_vector", &NDArrayBaseClass::get_vector);
}

template <typename T, size_t N>
static void declareNDArray(py::module & mod, std::string const & suffix) {
    using NDArrayBaseClass = NDArrayBase<T>;
    using NDArrayClass = NDArray<T, N>;

    py::class_<NDArrayClass, std::shared_ptr<NDArrayClass>, NDArrayBaseClass> clsNDArray(mod,
                                                      ("clsNDArray" + suffix + std::to_string(N)).c_str());
    //clsNDArray.def(py::init<std::vector
}

PYBIND11_PLUGIN(clsNDArray) {

    py::module clsNDArrayBase("clsNDArrayBase");
    declareNDArrayBase<float>(clsNDArrayBase, "F");
    declareNDArrayBase<double>(clsNDArrayBase, "D");
    declareNDArrayBase<int>(clsNDArrayBase, "I");

    py::module clsNDArray("clsNDArray");
    declareNDArray<float, 1>(clsNDArrayBase, "F");
    declareNDArray<double, 1>(clsNDArrayBase, "D");
    declareNDArray<int, 1>(clsNDArrayBase, "I");
}
}}  // ngraph
