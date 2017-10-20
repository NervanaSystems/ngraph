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

#include "ngraph/runtime/parameterized_tensor_view.hpp"
#include "ngraph/types/element_type.hpp"

namespace ngraph
{
    namespace test
    {
        /// @brief Same as numpy.allclose
        /// @param as First tensors to compare
        /// @param bs Second tensors to compare
        /// @param rtol Relative tolerance
        /// @param atol Absolute tolerance
        /// Returns true if shapes match and for all elements, |a_i-b_i| <= atol + rtol*|b_i|.
        template <typename ET>
        bool all_close(
            const std::vector<std::shared_ptr<ngraph::runtime::ParameterizedTensorView<ET>>>& as,
            const std::vector<std::shared_ptr<ngraph::runtime::ParameterizedTensorView<ET>>>& bs,
            typename ET::type rtol,
            typename ET::type atol);

        extern template bool all_close<element::Float32>(
            const std::vector<std::shared_ptr<
                runtime::ParameterizedTensorView<element::Float32>>>& as,
            const std::vector<std::shared_ptr<
                runtime::ParameterizedTensorView<element::Float32>>>& bs,
            element::Float32::type rtol,
            element::Float32::type atol);

        extern template bool all_close<element::Float64>(
            const std::vector<std::shared_ptr<
                runtime::ParameterizedTensorView<element::Float64>>>& as,
            const std::vector<std::shared_ptr<
                runtime::ParameterizedTensorView<element::Float64>>>& bs,
            element::Float64::type rtol,
            element::Float64::type atol);

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

        extern template bool all_close<ngraph::element::Float32>(
            const std::shared_ptr<
                ngraph::runtime::ParameterizedTensorView<ngraph::element::Float32>>& a,
            const std::shared_ptr<
                ngraph::runtime::ParameterizedTensorView<ngraph::element::Float32>>& b,
            ngraph::element::Float32::type rtol,
            ngraph::element::Float32::type atol);

        extern template bool all_close<ngraph::element::Float64>(
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

        extern template bool all_close<float>(const std::vector<float>& a,
                                              const std::vector<float>& b,
                                              float rtol,
                                              float atol);

        extern template bool all_close<double>(const std::vector<double>& a,
                                               const std::vector<double>& b,
                                               double rtol,
                                               double atol);
    }
}
