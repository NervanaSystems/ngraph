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

#include <algorithm>   // std::generate
#include <cmath>       // std::floor, std::min
#include <cstddef>     // std::size_t
#include <iterator>    // std::begin, std::end
#include <memory>      // std::shared_ptr, std::make_shared
#include <type_traits> // std::enable_if
#include <vector>

#include "ngraph/op/constant.hpp"
#include "ngraph/op/util/broadcasting.hpp"
#include "ngraph/shape.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace common
        {
            /// \brief      Return a monotonic sequence.
            ///
            /// \note       Limitations: this function may not work for very large integer values
            ///             (near numerical limits).
            ///
            /// \param[in]  start_value  The start value of the sequence.
            /// \param[in]  end_value    The end value of the sequence.
            /// \param[in]  step         The step value for the sequence.
            ///
            /// \tparam     T            The data value type.
            ///
            /// \return     The vector with monotonic sequence
            template <typename T>
            std::vector<T> get_monotonic_range(T end_value, T start_value = T{0}, T step = T{1})
            {
                auto value_count =
                    static_cast<std::size_t>(std::floor((end_value - start_value) / step));

                std::vector<T> range(value_count);

                // Calculate initial value (one step below starting value)
                size_t n = start_value - step;
                // Generate a vector of values by adding step to previous value
                std::generate(
                    std::begin(range), std::end(range), [&n, &step]() -> T { return n += step; });

                return range;
            }

            /// \brief      Handle negative axis value.
            ///
            /// \param[in]  axis        The requested axis value.
            /// \param[in]  tensor_dim  The corresponding tensor dimensionality.
            ///
            /// \tparam     T           Provided axis value type.
            ///
            /// \return     If negative axis, then return sum of tensor dimension and axis.
            ///
            template <typename T,
                      typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
            std::int64_t convert_negative_axis(T axis, std::size_t tensor_dim)
            {
                if (axis >= 0)
                {
                    return std::min(axis, static_cast<T>(tensor_dim));
                }
                else
                {
                    return static_cast<std::int64_t>(tensor_dim) + axis;
                }
            }

            /// \brief Creates a square identity matrix.
            ///
            /// \param[in] n Order of the resulting matrix.
            ///
            /// \return A Constant node representing identity matrix with shape (n, n).
            template <typename T = double>
            std::shared_ptr<ngraph::op::Constant> square_identity(const size_t n,
                                                                  const element::Type& type)
            {
                std::vector<T> identity_matrix(n * n, T{0});

                for (size_t row = 0; row < n; ++row)
                {
                    const size_t diagonal_element = (n * row) + row;
                    identity_matrix.at(diagonal_element) = T{1};
                }

                return std::make_shared<ngraph::op::Constant>(type, Shape{{n, n}}, identity_matrix);
            }
        } // namespace  common
    }     // namespace onnx_import
} // namespace ngraph
