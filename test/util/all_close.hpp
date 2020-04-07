//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#include <cmath>
#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "ngraph/type/element_type.hpp"
#include "test_tools.hpp"

namespace ngraph
{
    namespace test
    {
        /// \brief Same as numpy.allclose
        /// \param a First tensor to compare
        /// \param b Second tensor to compare
        /// \param rtol Relative tolerance
        /// \param atol Absolute tolerance
        /// \returns true if shapes match and for all elements, |a_i-b_i| <= atol + rtol*|b_i|.
        template <typename T>
        typename std::enable_if<std::is_floating_point<T>::value, ::testing::AssertionResult>::type
            all_close(const std::vector<T>& a,
                      const std::vector<T>& b,
                      T rtol = static_cast<T>(1e-5),
                      T atol = static_cast<T>(1e-8))
        {
            bool rc = true;
            ::testing::AssertionResult ar_fail = ::testing::AssertionFailure();
            if (a.size() != b.size())
            {
                throw std::invalid_argument("all_close: Argument vectors' sizes do not match");
            }
            size_t count = 0;
            for (size_t i = 0; i < a.size(); ++i)
            {
                if (std::abs(a[i] - b[i]) > atol + rtol * std::abs(b[i]) || !std::isfinite(a[i]) ||
                    !std::isfinite(b[i]))
                {
                    if (count < 5)
                    {
                        ar_fail << std::setprecision(std::numeric_limits<long double>::digits10 + 1)
                                << a[i] << " is not close to " << b[i] << " at index " << i
                                << std::endl;
                    }
                    count++;
                    rc = false;
                }
            }
            ar_fail << "diff count: " << count << " out of " << a.size() << std::endl;
            return rc ? ::testing::AssertionSuccess() : ar_fail;
        }

        /// \brief Same as numpy.allclose
        /// \param a First tensor to compare
        /// \param b Second tensor to compare
        /// \param rtol Relative tolerance
        /// \param atol Absolute tolerance
        /// \returns true if shapes match and for all elements, |a_i-b_i| <= atol + rtol*|b_i|.
        template <typename T>
        typename std::enable_if<std::is_integral<T>::value, ::testing::AssertionResult>::type
            all_close(const std::vector<T>& a,
                      const std::vector<T>& b,
                      T rtol = static_cast<T>(1e-5),
                      T atol = static_cast<T>(1e-8))
        {
            bool rc = true;
            ::testing::AssertionResult ar_fail = ::testing::AssertionFailure();
            if (a.size() != b.size())
            {
                throw std::invalid_argument("all_close: Argument vectors' sizes do not match");
            }
            for (size_t i = 0; i < a.size(); ++i)
            {
                T abs_diff = (a[i] > b[i]) ? (a[i] - b[i]) : (b[i] - a[i]);
                if (abs_diff > atol + rtol * b[i])
                {
                    // use unary + operator to force integral values to be displayed as numbers
                    ar_fail << +a[i] << " is not close to " << +b[i] << " at index " << i
                            << std::endl;
                    rc = false;
                }
            }
            return rc ? ::testing::AssertionSuccess() : ar_fail;
        }

        /// \brief Same as numpy.allclose
        /// \param a First tensor to compare
        /// \param b Second tensor to compare
        /// \param rtol Relative tolerance
        /// \param atol Absolute tolerance
        /// Returns true if shapes match and for all elements, |a_i-b_i| <= atol + rtol*|b_i|.
        template <typename T>
        ::testing::AssertionResult all_close(const std::shared_ptr<ngraph::runtime::Tensor>& a,
                                             const std::shared_ptr<ngraph::runtime::Tensor>& b,
                                             T rtol = 1e-5f,
                                             T atol = 1e-8f)
        {
            // Check that the layouts are compatible
            if (*a->get_tensor_layout() != *b->get_tensor_layout())
            {
                return ::testing::AssertionFailure()
                       << "Cannot compare tensors with different layouts";
            }

            if (a->get_shape() != b->get_shape())
            {
                return ::testing::AssertionFailure()
                       << "Cannot compare tensors with different shapes";
            }

            return all_close(read_vector<T>(a), read_vector<T>(b), rtol, atol);
        }

        /// \brief Same as numpy.allclose
        /// \param as First tensors to compare
        /// \param bs Second tensors to compare
        /// \param rtol Relative tolerance
        /// \param atol Absolute tolerance
        /// Returns true if shapes match and for all elements, |a_i-b_i| <= atol + rtol*|b_i|.
        template <typename T>
        ::testing::AssertionResult
            all_close(const std::vector<std::shared_ptr<ngraph::runtime::Tensor>>& as,
                      const std::vector<std::shared_ptr<ngraph::runtime::Tensor>>& bs,
                      T rtol,
                      T atol)
        {
            if (as.size() != bs.size())
            {
                return ::testing::AssertionFailure()
                       << "Cannot compare tensors with different sizes";
            }
            for (size_t i = 0; i < as.size(); ++i)
            {
                auto ar = all_close(as[i], bs[i], rtol, atol);
                if (!ar)
                {
                    return ar;
                }
            }
            return ::testing::AssertionSuccess();
        }
    }
}
