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
#include <vector>

#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/manager.hpp"
#include "ngraph/runtime/parameterized_tensor_view.hpp"
#include "ngraph/runtime/tuple.hpp"
#include "ngraph/runtime/value.hpp"
#include "ngraph/types/element_type.hpp"

namespace ngraph
{
    namespace autodiff
    {
        /// @brief numeric approximation of the derivative
        /// @param f A function
        /// @param args Values for the arguments (the independent variables)
        /// @param delta increment for the variables
        /// @returns vector of dy/dvar, where each dy/dvar's shape is concat(y.shape(), var.shape())
        template <typename ET>
        std::vector<std::shared_ptr<runtime::ParameterizedTensorView<ET>>> numeric_derivative(
            const std::shared_ptr<runtime::Manager>& manager,
            const std::shared_ptr<runtime::Backend>& backend,
            const std::shared_ptr<Function>& f,
            const std::vector<std::shared_ptr<runtime::ParameterizedTensorView<ET>>>& args,
            typename ET::type delta);

        extern template std::vector<
            std::shared_ptr<runtime::ParameterizedTensorView<element::Float32>>>
            numeric_derivative(
                const std::shared_ptr<runtime::Manager>& manager,
                const std::shared_ptr<runtime::Backend>& backend,
                const std::shared_ptr<Function>& f,
                const std::vector<
                    std::shared_ptr<runtime::ParameterizedTensorView<element::Float32>>>& args,
                element::Float32::type delta);

        extern template std::vector<
            std::shared_ptr<runtime::ParameterizedTensorView<element::Float64>>>
            numeric_derivative(
                const std::shared_ptr<runtime::Manager>& manager,
                const std::shared_ptr<runtime::Backend>& backend,
                const std::shared_ptr<Function>& f,
                const std::vector<
                    std::shared_ptr<runtime::ParameterizedTensorView<element::Float64>>>& args,
                element::Float64::type delta);
    }
}
