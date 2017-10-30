#include <pybind11/pybind11.h>
#include "ngraph/types/element_type.hpp"

namespace py = pybind11;

PYBIND11_MODULE(ops, m) {
	m.doc() = R"pbdoc(testing element type)"
	py::class_<ngraph::element::Type>(m, "Type")
		.def(py::init<size_t bitwidth, bool is_float, bool is_signed, const std::string& cname>())
		.def("c_type_string", &Type::c_type_string);
}