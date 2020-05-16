//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#include <cstdint>
#include <iostream>
#include <memory>
#include <set>
#include <string>

#include "ngraph/function.hpp"
#include "utils/onnx_importer_visibility.hpp"

/// \brief              Top level nGraph namespace.
namespace ngraph
{
    /// \brief          ONNX importer features namespace.
    ///                 Functions in this namespace make it possible to use ONNX models.
    namespace onnx_import
    {
        /// \brief      Returns a set of names of supported operators
        ///             for the given opset version and domain.
        ///
        /// \param[in]  version   An opset version to get the supported operators for.
        /// \param[in]  domain    A domain to get the supported operators for.
        ///
        /// \return     The set containing names of supported operators.
        ONNX_IMPORTER_API
        std::set<std::string> get_supported_operators(std::int64_t version,
                                                      const std::string& domain);

        /// \brief      Determines whether ONNX operator is supported.
        ///
        /// \param[in]  op_name   The ONNX operator name.
        /// \param[in]  version   The ONNX operator set version.
        /// \param[in]  domain    The domain the ONNX operator is registered to.
        ///                       If not set, the default domain "ai.onnx" is used.
        ///
        /// \return     true if operator is supported, false otherwise.
        ONNX_IMPORTER_API
        bool is_operator_supported(const std::string& op_name,
                                   std::int64_t version,
                                   const std::string& domain = "ai.onnx");

        /// \brief      Imports and converts an serialized ONNX model from the input stream
        ///             to an nGraph Function representation.
        ///
        /// \note       If stream parsing fails or the ONNX model contains unsupported ops,
        ///             the function throws an ngraph_error exception.
        ///
        /// \param[in]  stream    The input stream (e.g. file stream, memory stream, etc).
        ///
        /// \return     An nGraph function that represents a single output from the created graph.
        ONNX_IMPORTER_API
        std::shared_ptr<Function> import_onnx_model(std::istream& stream);

        /// \brief     Imports and converts an ONNX model from the input file
        ///            to an nGraph Function representation.
        ///
        /// \note      If file parsing fails or the ONNX model contains unsupported ops,
        ///            the function throws an ngraph_error exception.
        ///
        /// \param[in] file_path  The path to a file containing the ONNX model
        ///                       (relative or absolute).
        ///
        /// \return    An nGraph function that represents a single output from the created graph.
        ONNX_IMPORTER_API
        std::shared_ptr<Function> import_onnx_model(const std::string& file_path);

    } // namespace onnx_import

} // namespace ngraph
