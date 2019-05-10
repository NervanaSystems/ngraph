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

#ifdef _WIN32
#pragma warning(push)

#pragma warning(disable : 4100)
#endif

// Prevents the compiler from complaining about or optimizing away variables
// that appear unused on Linux
#if (defined(__GNUC__) && !defined(__clang__))
#undef ONNX_ATTRIBUTE_UNUSED
#define ONNX_ATTRIBUTE_UNUSED __attribute__((__unused__))
#else
#define ONNX_ATTRIBUTE_UNUSED
#endif

#define UNUSED_PARAMETER ONNX_ATTRIBUTE_UNUSED = 0

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
                std::shared_ptr<ngraph::Node> sigmoid(const std::shared_ptr<ngraph::Node>& arg,
                                                      float alpha UNUSED_PARAMETER,
                                                      float beta UNUSED_PARAMETER);
                std::shared_ptr<ngraph::Node> tanh(const std::shared_ptr<ngraph::Node>& arg,
                                                   float alpha UNUSED_PARAMETER,
                                                   float beta UNUSED_PARAMETER);
                std::shared_ptr<ngraph::Node> relu(const std::shared_ptr<ngraph::Node>& arg,
                                                   float alpha UNUSED_PARAMETER,
                                                   float beta UNUSED_PARAMETER);
                std::shared_ptr<ngraph::Node>
                    hardsigmoid(const std::shared_ptr<ngraph::Node>& arg, float alpha, float beta);
            }

            using ActivationFunctionType = std::shared_ptr<ngraph::Node> (*)(
                const std::shared_ptr<ngraph::Node>&, float, float);

            class ActivationFunction
            {
            public:
                ActivationFunction(ActivationFunctionType f, float alpha, float beta);
                ActivationFunction(ActivationFunctionType f, float alpha);
                ActivationFunction(ActivationFunctionType f);

                std::shared_ptr<ngraph::Node>
                    operator()(const std::shared_ptr<ngraph::Node>& arg) const;

                void set_alpha(float alpha) { m_alpha = alpha; }
                void set_beta(float beta) { m_beta = beta; }
            private:
                ActivationFunctionType m_function;
                float m_alpha;
                float m_beta;
            };

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

#ifdef _WIN32
#pragma warning(pop)
#endif

#ifdef UNUSED_PARAMETER
#undef UNUSED_PARAMETER
#endif
#ifdef ONNX_ATTRIBUTE_UNUSED
#undef ONNX_ATTRIBUTE_UNUSED
#endif
