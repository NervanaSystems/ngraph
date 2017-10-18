// ----------------------------------------------------------------------------
// Copyright 2017 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------

#pragma once

#include <memory>
#include <unordered_map>

#include "ngraph/runtime/parameterized_tensor_view.hpp"

namespace ngraph
{
    class Node;
    class Function;

    namespace runtime
    {
        class Backend;
        class Manager;
    }

    namespace autodiff
    {
        /// @brief Returns a FunctionSpec for the backprop derivative of its argument.
        /// @param f is f(X_i...)
        /// @returns f'(X_i..., c) where f'(x_i, ..., c)_j is backprop for X_j
        std::shared_ptr<Function> backprop_function(const std::shared_ptr<Function>& f);

        template <typename ET>
        std::vector<std::shared_ptr<runtime::ParameterizedTensorView<ET>>> backprop_derivative(
            const std::shared_ptr<runtime::Manager>& manager,
            const std::shared_ptr<runtime::Backend>& backend,
            const std::shared_ptr<Function>& f,
            const std::vector<std::shared_ptr<runtime::ParameterizedTensorView<ET>>>& args);

        extern template std::vector<
            std::shared_ptr<runtime::ParameterizedTensorView<ngraph::element::Float32>>>
            backprop_derivative<ngraph::element::Float32>(
                const std::shared_ptr<runtime::Manager>& manager,
                const std::shared_ptr<runtime::Backend>& backend,
                const std::shared_ptr<Function>& f,
                const std::vector<
                    std::shared_ptr<runtime::ParameterizedTensorView<element::Float32>>>& args);

        extern template std::vector<
            std::shared_ptr<runtime::ParameterizedTensorView<ngraph::element::Float64>>>
            backprop_derivative<ngraph::element::Float64>(
                const std::shared_ptr<runtime::Manager>& manager,
                const std::shared_ptr<runtime::Backend>& backend,
                const std::shared_ptr<Function>& f,
                const std::vector<
                    std::shared_ptr<runtime::ParameterizedTensorView<element::Float64>>>& args);
    }
}
