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

#pragma once

#include <iostream>
#include <string>

#include "ngraph/function.hpp"

#include "core/operator_set.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        // Registers ONNX custom operator
        void register_operator(const std::string& name,
                               std::int64_t version,
                               const std::string& domain,
                               Operator fn);

        // Convert on ONNX model to a vector of nGraph Functions (input stream)
        std::vector<std::shared_ptr<Function>> load_onnx_model(std::istream&);

        // Convert an ONNX model to a vector of nGraph Functions
        std::vector<std::shared_ptr<Function>> load_onnx_model(const std::string&);

        // Convert the first output of an ONNX model to an nGraph Function (input stream)
        std::shared_ptr<Function> import_onnx_function(std::istream&);

        // Convert the first output of an ONNX model to an nGraph Function
        std::shared_ptr<Function> import_onnx_function(const std::string&);

    } // namespace onnx_import

} // namespace ngraph
