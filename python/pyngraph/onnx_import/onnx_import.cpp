//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <istream>
#include <memory>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ngraph/frontend/onnx_import/onnx.hpp"
#include "ngraph/function.hpp"
#include "pyngraph/onnx_import/onnx_import.hpp"

namespace py = pybind11;

static std::vector<std::shared_ptr<ngraph::Function>>
    load_onnx_model(const std::string& model_proto)
{
    std::istringstream iss(model_proto, std::ios_base::binary | std::ios_base::in);
    return ngraph::onnx_import::load_onnx_model(iss);
}

static std::shared_ptr<ngraph::Function> import_onnx_function(const std::string& model_proto)
{
    std::istringstream iss(model_proto, std::ios_base::binary | std::ios_base::in);
    return ngraph::onnx_import::import_onnx_function(iss);
}

void regmodule_pyngraph_onnx_import(py::module mod)
{
    mod.def("load_onnx_model", &load_onnx_model);
    mod.def("import_onnx_function", &import_onnx_function);
    mod.def("load_onnx_model_file",
            static_cast<std::vector<std::shared_ptr<ngraph::Function>> (*)(const std::string&)>(
                &ngraph::onnx_import::load_onnx_model),
            py::arg());
    mod.def("import_onnx_function_file",
            static_cast<std::shared_ptr<ngraph::Function> (*)(const std::string&)>(
                &ngraph::onnx_import::import_onnx_function),
            py::arg());
}
