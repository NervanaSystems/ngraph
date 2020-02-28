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
#include "ngraph/ngraph_visibility.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        /// \brief      Return the set of names of supported operators.
        ///
        /// \param[in]  version  Determines set version of operators which are returned
        /// \param[in]  domain   Determines domain of operators which are returned
        ///
        /// \return     The set containing names of supported operators
        NGRAPH_API
        std::set<std::string> get_supported_operators(std::int64_t version,
                                                      const std::string& domain);

        /// \brief      Determines whether ONNX operator is supported.
        ///
        /// \param[in]  op_name  The ONNX operator name
        /// \param[in]  version  The ONNX operator set version
        /// \param[in]  domain   The domain the ONNX operator is registered to
        ///                      If not set default domain "ai.onnx" is used
        ///
        /// \return     True if operator is supported, False otherwise
        NGRAPH_API
        bool is_operator_supported(const std::string& op_name,
                                   std::int64_t version,
                                   const std::string& domain = "ai.onnx");

        /// \brief      Import and convert an ONNX model from stream
        ///             to an nGraph Function representation.
        ///             The serialized ONNX model is read from input stream.
        ///
        /// \note       If parsing stream fails or ONNX model contains not supported ops
        ///             ngraph_error exception can be thrown.
        ///
        /// \param[in]  stream      The input stream (e.g. file stream, memory stream, etc)
        ///
        /// \return     The function returns an nGraph Function which represents single output
        ///             from the created graph
        NGRAPH_API
        std::shared_ptr<Function> import_onnx_model(std::istream& stream);

        /// \brief     Import and convert an ONNX model from file
        ///            to an nGraph Function representation.
        ///            The ONNX model is read from ONNX file.
        ///
        /// \note      If parsing file fails or ONNX model contains not supported ops
        ///            ngraph_error exception can be thrown.
        ///
        /// \param[in] file_path  The path to file containing ONNX model (relative or absolute)
        ///
        /// \return    The function returns an nGraph Function which represents single output
        ///            from the created graph
        NGRAPH_API
        std::shared_ptr<Function> import_onnx_model(const std::string& file_path);

    } // namespace onnx_import

} // namespace ngraph
