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

#include <functional>
#include <iterator>
#include <unordered_map>

#include "activation_functions.hpp"
#include "ngraph/op/relu.hpp"
#include "ngraph/op/sigmoid.hpp"
#include "ngraph/op/tanh.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace rnn
        {
            namespace detail
            {
                std::shared_ptr<ngraph::Node> sigmoid(const std::shared_ptr<ngraph::Node>& arg)
                {
                    return std::make_shared<ngraph::op::Sigmoid>(arg);
                }

                std::shared_ptr<ngraph::Node> tanh(const std::shared_ptr<ngraph::Node>& arg)
                {
                    return std::make_shared<ngraph::op::Tanh>(arg);
                }

                std::shared_ptr<ngraph::Node> relu(const std::shared_ptr<ngraph::Node>& arg)
                {
                    return std::make_shared<ngraph::op::Relu>(arg);
                }
            } // namespace detail

            ActivationFunction get_activation_func_by_name(const std::string& func_name)
            {
                using ActivationFunctionMap = std::unordered_map<std::string, ActivationFunction>;

                static ActivationFunctionMap func_map{
                    {"sigmoid", std::bind(detail::sigmoid, std::placeholders::_1)},
                    {"tanh", std::bind(detail::tanh, std::placeholders::_1)},
                    {"relu", std::bind(detail::relu, std::placeholders::_1)}};

                auto func_it = func_map.find(func_name);
                if (func_it == std::end(func_map))
                {
                    throw error::UnknownActivationFunction(func_name);
                }
                return func_it->second;
            }

        } //namespace rnn

    } // namespace onnx_import

} // namespace ngraph
