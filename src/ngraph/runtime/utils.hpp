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
    namespace runtime
    {
        /// @brief Framework constructor of a tensor of a specific element type and shape.
        template <typename ET>
        std::shared_ptr<ngraph::runtime::ParameterizedTensorView<ET>>
            make_tensor(const Shape& shape)
        {
            return std::make_shared<runtime::ParameterizedTensorView<ET>>(shape);
        }

        /// @brief Framework constructor of a tuple from a sequence of values.
        std::shared_ptr<ngraph::runtime::Tuple>
            make_tuple(const std::vector<std::shared_ptr<ngraph::runtime::Value>>& elements);

        /// @brief Same as numpy.allclose
        /// @param a First tensor to compare
        /// @param b Second tensor to compare
        /// @param rtol Relative tolerance
        /// @param atol Absolute tolerance
        /// Returns true if shapes match and for all elements, |a_i-b_i| <= atol + rtol*|b_i|.
        template <typename ET>
        bool all_close(const std::shared_ptr<ngraph::runtime::ParameterizedTensorView<ET>>& a,
                       const std::shared_ptr<ngraph::runtime::ParameterizedTensorView<ET>>& b,
                       typename ET::type rtol = 1e-5f,
                       typename ET::type atol = 1e-8f);

        extern template bool ngraph::runtime::all_close<ngraph::element::Float32>(
            const std::shared_ptr<
                ngraph::runtime::ParameterizedTensorView<ngraph::element::Float32>>& a,
            const std::shared_ptr<
                ngraph::runtime::ParameterizedTensorView<ngraph::element::Float32>>& b,
            ngraph::element::Float32::type rtol,
            ngraph::element::Float32::type atol);

        extern template bool ngraph::runtime::all_close<ngraph::element::Float64>(
            const std::shared_ptr<
                ngraph::runtime::ParameterizedTensorView<ngraph::element::Float64>>& a,
            const std::shared_ptr<
                ngraph::runtime::ParameterizedTensorView<ngraph::element::Float64>>& b,
            ngraph::element::Float64::type rtol,
            ngraph::element::Float64::type atol);

        /// @brief Same as numpy.allclose
        /// @param a First tensor to compare
        /// @param b Second tensor to compare
        /// @param rtol Relative tolerance
        /// @param atol Absolute tolerance
        /// @returns true if shapes match and for all elements, |a_i-b_i| <= atol + rtol*|b_i|.
        template <typename T>
        bool all_close(const std::vector<T>& a,
                       const std::vector<T>& b,
                       T rtol = 1e-5f,
                       T atol = 1e-8f);

        extern template bool ngraph::runtime::all_close<float>(const std::vector<float>& a,
                                                               const std::vector<float>& b,
                                                               float rtol,
                                                               float atol);

        extern template bool ngraph::runtime::all_close<double>(const std::vector<double>& a,
                                                                const std::vector<double>& b,
                                                                double rtol,
                                                                double atol);

        /// @brief Contains the information in a Function, but can be used to construct derived functions such as derivatives.
        class FunctionSpec
        {
        public:
            FunctionSpec(const std::shared_ptr<Node>& result,
                         const std::shared_ptr<const ValueType>& result_type,
                         const std::vector<std::shared_ptr<op::Parameter>>& parameters)
                : m_result(result)
                , m_result_type(result_type)
                , m_parameters(parameters)
            {
            }

            FunctionSpec(const std::shared_ptr<Node>& result,
                         const std::vector<std::shared_ptr<op::Parameter>>& parameters)
                : m_result(result)
                , m_result_type(result->get_value_type())
                , m_parameters(parameters)
            {
            }

            const std::shared_ptr<const ValueType> get_result_type() const { return m_result_type; }
            std::shared_ptr<Node> get_result() { return m_result; }
            const std::vector<std::shared_ptr<op::Parameter>> get_parameters() const
            {
                return m_parameters;
            }

            operator std::shared_ptr<Function>() const;

        protected:
            std::shared_ptr<Node> m_result;
            std::shared_ptr<const ValueType> m_result_type;
            std::vector<std::shared_ptr<op::Parameter>> m_parameters;
        };

        /// @brief Returns a FunctionSpec for the backprop derivative of its argument.
        /// @param f is f(X_i...)
        /// @returns f'(c, X_i...) -> tuple of tensors in same order as in X_i
        std::shared_ptr<ngraph::runtime::FunctionSpec>
            derivative(const std::shared_ptr<ngraph::runtime::FunctionSpec>& f);

        /// @brief numeric approximation of the derivative
        /// @param f A function
        /// @param args Values for the arguments (the independent variables)
        /// @param delta increment for the variables
        /// @returns vector of dy/dvar, where each dy/dvar's shape is concat(y.shape(), var.shape())
        template <typename ET>
        std::vector<std::shared_ptr<ngraph::runtime::ParameterizedTensorView<ET>>>
            numeric_derivative(
                const std::shared_ptr<runtime::Manager>& manager,
                const std::shared_ptr<runtime::Backend>& backend,
                const std::shared_ptr<FunctionSpec>& f,
                const std::vector<std::shared_ptr<ngraph::runtime::ParameterizedTensorView<ET>>>&
                    args,
                typename ET::type delta);

        extern template std::vector<
            std::shared_ptr<ngraph::runtime::ParameterizedTensorView<element::Float32>>>
            ngraph::runtime::numeric_derivative(
                const std::shared_ptr<runtime::Manager>& manager,
                const std::shared_ptr<runtime::Backend>& backend,
                const std::shared_ptr<FunctionSpec>& f,
                const std::vector<
                    std::shared_ptr<ngraph::runtime::ParameterizedTensorView<element::Float32>>>&
                    args,
                element::Float32::type delta);

        extern template std::vector<
            std::shared_ptr<ngraph::runtime::ParameterizedTensorView<element::Float64>>>
            ngraph::runtime::numeric_derivative(
                const std::shared_ptr<runtime::Manager>& manager,
                const std::shared_ptr<runtime::Backend>& backend,
                const std::shared_ptr<FunctionSpec>& f,
                const std::vector<
                    std::shared_ptr<ngraph::runtime::ParameterizedTensorView<element::Float64>>>&
                    args,
                element::Float64::type delta);

        template <typename ET>
        std::vector<std::shared_ptr<ngraph::runtime::ParameterizedTensorView<ET>>>
            backwards_derivative(
                const std::shared_ptr<runtime::Manager>& manager,
                const std::shared_ptr<runtime::Backend>& backend,
                const std::shared_ptr<FunctionSpec>& f,
                const std::vector<std::shared_ptr<ngraph::runtime::ParameterizedTensorView<ET>>>&
                    args);

        extern template std::vector<
            std::shared_ptr<ngraph::runtime::ParameterizedTensorView<ngraph::element::Float32>>>
            ngraph::runtime::backwards_derivative<ngraph::element::Float32>(
                const std::shared_ptr<runtime::Manager>& manager,
                const std::shared_ptr<runtime::Backend>& backend,
                const std::shared_ptr<FunctionSpec>& f,
                const std::vector<
                    std::shared_ptr<ngraph::runtime::ParameterizedTensorView<element::Float32>>>&
                    args);

        extern template std::vector<
            std::shared_ptr<ngraph::runtime::ParameterizedTensorView<ngraph::element::Float64>>>
            ngraph::runtime::backwards_derivative<ngraph::element::Float64>(
                const std::shared_ptr<runtime::Manager>& manager,
                const std::shared_ptr<runtime::Backend>& backend,
                const std::shared_ptr<FunctionSpec>& f,
                const std::vector<
                    std::shared_ptr<ngraph::runtime::ParameterizedTensorView<element::Float64>>>&
                    args);
    }
}
