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
#include <set>
#include <string>

#include "core/operator_set.hpp"
#include "core/weight.hpp"
#include "ngraph/function.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        /// \brief Registers ONNX custom operator
        /// Performs the registration of external ONNX operator. This means the code
        /// of the operator is not part of ONNX importer. The operator shall be registered
        /// before calling `load_onnx_model()` or `import_onnx_function()` functions.
        /// \param name    name of the operator,
        /// \param version  version of the operator (opset),
        /// \param domain  domain the operator belongs to,
        /// \param fn       function providing the implementation of the operator.
        NGRAPH_API
        void register_operator(const std::string& name,
                               std::int64_t version,
                               const std::string& domain,
                               Operator fn);

        /// \brief      Return the set of names of supported operators.
        ///
        /// \param[in]  version  The requested version of ONNX operators set.
        /// \param[in]  domain   The requested domain the operators where registered for.
        ///
        /// \return     The set containing names of supported operators.
        ///
        NGRAPH_API
        std::set<std::string> get_supported_operators(std::int64_t version,
                                                      const std::string& domain);

        /// \brief      Determines whether ONNX operator is supported.
        ///
        /// \param[in]  op_name  The ONNX operator name.
        /// \param[in]  version  The ONNX operator set version.
        /// \param[in]  domain   The domain the ONNX operator is registered to.
        ///
        /// \return     True if operator is supported, False otherwise.
        ///
        NGRAPH_API
        bool is_operator_supported(const std::string& op_name,
                                   std::int64_t version,
                                   const std::string& domain = "ai.onnx");

        /// \brief Convert an ONNX model to nGraph function
        /// The function translated serialized ONNX model to nGraph function. The serialized
        /// ONNX model is read from input stream.
        /// \param sin       input stream (e.g. file stream, memory stream, etc),
        /// \param weights  weights associated with the model. If weights are embedded into
        ///                   the model this parameter shall be empty. Having weights in a model
        ///                   and providing through this parameters is invalid (the weights from
        ///                   the model  will take precedence).
        /// \return The function returns a nGraph function representing single output from graph.
        NGRAPH_API
        std::shared_ptr<Function> import_onnx_model(std::istream& sin, const Weights& weights = {});

        /// \brief Convert an ONNX model to nGraph functions
        /// The function translated serialized ONNX model to nGraph functions. The ONNX model
        /// is read from ONNX file.
        /// \param filename  file name (relative or absolute path name),
        /// \param weights  weights associated with the model. If weights are embedded into
        ///                   the model this parameter shall be empty. Having weights in a model
        ///                   and providing through this parameters is invalid (the weights from
        ///                   the model  will take precedence).
        /// \return The function returns a nGraph function representing single output from graph.
        NGRAPH_API
        std::shared_ptr<Function> import_onnx_model(const std::string& filename,
                                                    const Weights& weights = {});

    } // namespace onnx_import

} // namespace ngraph
