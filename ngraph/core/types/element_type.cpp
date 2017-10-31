#include <pybind11/pybind11.h>
#include "ngraph/types/element_type.hpp"

namespace py = pybind11;

PYBIND11_MODULE(element_type, m) {
	m.doc() = "testing element type";
	py::class_<ngraph::element::Type>(m, "Type")
		.def("hash", &ngraph::element::Type::hash);
}
