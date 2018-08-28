/*******************************************************************************
* Copyright 2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#pragma once

#include "ngraph/except.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace error
        {
            struct NotSupported : ngraph_error
            {
                explicit NotSupported(const std::string& op_name,
                                      const std::string& name,
                                      const std::string& message)
                    : ngraph_error{op_name + " node (" + name + "): " + message}
                {
                }
            };

            namespace parameter
            {
                struct Value : ngraph_error
                {
                    Value(const std::string& op_name,
                          const std::string& name,
                          const std::string& message)
                        : ngraph_error{op_name + " node (" + name + "): " + message}
                    {
                    }
                };
            } // namespace paramter

        } // namespace  error

    } // namespace  onnx_import

} // namespace  ngraph
