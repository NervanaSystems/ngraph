//*****************************************************************************
// Copyright 2017-2019 Intel Corporation
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

#include <memory>
#include <string>

#include "ngraph/except.hpp"
#include "ngraph/node.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace rnn
        {
            namespace error
            {
                struct UnknownActivationFunction : ngraph_error
                {
                    UnknownActivationFunction(const std::string& func_name)
                        : ngraph_error{"Unknown activation function: " + func_name}
                    {
                    }
                };
            }

            namespace detail
            {
                std::shared_ptr<ngraph::Node> sigmoid(const std::shared_ptr<ngraph::Node>& arg);
                std::shared_ptr<ngraph::Node> tanh(const std::shared_ptr<ngraph::Node>& arg);
                std::shared_ptr<ngraph::Node> relu(const std::shared_ptr<ngraph::Node>& arg);
            }

            using ActivationFunction =
                std::function<std::shared_ptr<ngraph::Node>(const std::shared_ptr<ngraph::Node>&)>;

            /// \brief      Gets the activation function by name.
            ///
            /// \param[in]  func_name  The function name
            ///
            /// \throws     UnknownActivationFunction When provided func_name is unknown.
            ///
            /// \return     The activation function object.
            ///
            ActivationFunction get_activation_func_by_name(const std::string& func_name);

        } //namespace rnn

    } // namespace onnx_import

} // namespace ngraph
