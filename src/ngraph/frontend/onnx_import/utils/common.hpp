//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
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

#include <cmath>       // std::floor
#include <cstddef>     // std::size_t
#include <iterator>    // std::begin, std::end
#include <memory>      // std::shared_ptr, std::make_shared
#include <type_traits> // std::enable_if, std::is_floating_point, std::is_integral
#include <vector>

#include "ngraph/op/constant.hpp"
#include "ngraph/shape.hpp"
#include "utils/broadcasting.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace common
        {
            namespace detail
            {
                namespace
                {
                    /// \brief      Fill specified range with monotonic sequence.
                    ///
                    /// \param[in]  first            The iterator to the beginning of the range.
                    /// \param[in]  last             The iterator to the past the end of the range.
                    /// \param[in]  init_value       The initial value for sequence.
                    /// \param[in]  step             The step value for sequence.
                    ///
                    /// \tparam     ForwardIterator  The forward iterator class type.
                    /// \tparam     T                The sequence value type.
                    ///
                    template <typename ForwardIterator, typename T>
                    void fill_monotonic_range(ForwardIterator first,
                                              ForwardIterator last,
                                              T init_value,
                                              T step)
                    {
                        for (; first != last; ++first, init_value += step)
                        {
                            *first = init_value;
                        }
                    }

                } // namespace anonymous
            }     // namespace  detail

            /// \brief      Return the monotonic sequence.
            ///
            /// \note       Specialization for integral types.
            ///
            /// \param[in]  start_value  The start value of the sequence.
            /// \param[in]  end_value    The end value of the sequence.
            /// \param[in]  step         The step value for the sequence.
            ///
            /// \tparam     T            The data value type.
            ///
            /// \return     The vector with monotonic sequence.
            template <typename T,
                      typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
            std::vector<T> get_monotonic_range(T end_value, T start_value = T{0}, T step = T{1})
            {
                std::size_t value_count = (end_value - start_value) / step;
                std::vector<T> range(value_count);
                detail::fill_monotonic_range(std::begin(range), std::end(range), start_value, step);
                return range;
            }

            /// \brief      Return the monotonic sequence.
            ///
            /// \note       Specialization for floating point types.
            ///
            /// \param[in]  start_value  The start value of the sequence.
            /// \param[in]  end_value    The end value of the sequence.
            /// \param[in]  step         The step value for the sequence.
            ///
            /// \tparam     T            The data value type.
            ///
            /// \return     The vector with monotonic sequence
            template <typename T,
                      typename std::enable_if<std::is_floating_point<T>::value, int>::type = 0>
            std::vector<T> get_monotonic_range(T end_value, T start_value = T{0.f}, T step = T{1.f})
            {
                std::size_t value_count =
                    reinterpret_cast<std::size_t>(std::floor((end_value - start_value) / step));
                std::vector<T> range(value_count);
                detail::fill_monotonic_range(std::begin(range), std::end(range), start_value, step);
                return range;
            }

            /// \brief      Makes a Constant Ngraph node.
            ///
            /// \param[in]  type   The node element type.
            /// \param[in]  shape  The tensor data shape.
            /// \param[in]  data   The data to initialize node with.
            ///
            /// \tparam     T      Input data value type.
            ///
            /// \return     The Ngraph node representing Constant data.
            ///
            template <typename T>
            std::shared_ptr<ngraph::Node> make_constant_node(const ngraph::element::Type& type,
                                                             const ngraph::Shape& shape,
                                                             const std::vector<T>& data)
            {
                std::shared_ptr<ngraph::Node> node;
                // Make constant node filled with single value.
                if (data.size() == 1)
                {
                    node = std::make_shared<ngraph::op::Constant>(type, ngraph::Shape{}, data);
                    node = make_broadcast_node(node, shape);
                }
                else
                {
                    node = std::make_shared<ngraph::op::Constant>(type, shape, data);
                }

                return node;
            }

        } // namespace  common
    }     // namespace onnx_import
} // namespace ngraph
