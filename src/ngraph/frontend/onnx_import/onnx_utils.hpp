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
#include <string>

#include "core/operator_set.hpp"
#include "ngraph/ngraph_visibility.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        /// \brief      Registers ONNX custom operator.
        ///             The function performs the registration of external ONNX operator.
        ///             which is not part of ONNX importer.
        ///
        /// \note       The operator shall be registered before calling
        ///             "import_onnx_model" functions.
        ///
        /// \param      name     The name of the operator.
        /// \param      version  The version of the operator (opset).
        /// \param      domain   The domain the operator belongs to
        /// \param      fn       The function providing the implementation of the operator.
        NGRAPH_API
        void register_operator(const std::string& name,
                               std::int64_t version,
                               const std::string& domain,
                               Operator fn);

    } // namespace onnx_import

} // namespace ngraph
