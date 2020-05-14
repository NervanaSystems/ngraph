    py::enum_<ngraph::op::PadType>(groupconvolution, "PadType", py::arithmetic())
        .value("EXPLICIT", ngraph::op::PadType::EXPLICIT)
        .value("SAME_LOWER", ngraph::op::PadType::SAME_LOWER)
        .value("SAME_UPPER", ngraph::op::PadType::SAME_UPPER)
        .value("VALID", ngraph::op::PadType::VALID)
        .value("AUTO", ngraph::op::PadType::AUTO)
        .value("NOTSET", ngraph::op::PadType::NOTSET)
        .export_values();