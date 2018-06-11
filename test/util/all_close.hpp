/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#pragma once

#include <cmath>
#include <memory>
#include <vector>

#include "ngraph/type/element_type.hpp"
#include "test_tools.hpp"

namespace ngraph
{
    namespace test
    {
        /// @brief Same as numpy.allclose
        /// @param a First tensor to compare
        /// @param b Second tensor to compare
        /// @param rtol Relative tolerance
        /// @param atol Absolute tolerance
        /// @returns true if shapes match and for all elements, |a_i-b_i| <= atol + rtol*|b_i|.
        template <typename T>
        bool all_close(const std::vector<T>& a,
                       const std::vector<T>& b,
                       T rtol = static_cast<T>(1e-5),
                       T atol = static_cast<T>(1e-8))
        {
            bool rc = true;
            assert(a.size() == b.size());
            for (size_t i = 0; i < a.size(); ++i)
            {
                if (std::abs(a[i] - b[i]) > atol + rtol * std::abs(b[i]))
                {
                    NGRAPH_INFO << a[i] << " is not close to " << b[i] << " at index " << i;
                    rc = false;
                }
            }
            return rc;
        }

        /// @brief Same as numpy.allclose
        /// @param a First tensor to compare
        /// @param b Second tensor to compare
        /// @param rtol Relative tolerance
        /// @param atol Absolute tolerance
        /// Returns true if shapes match and for all elements, |a_i-b_i| <= atol + rtol*|b_i|.
        template <typename T>
        bool all_close(const std::shared_ptr<ngraph::runtime::TensorView>& a,
                       const std::shared_ptr<ngraph::runtime::TensorView>& b,
                       T rtol = 1e-5f,
                       T atol = 1e-8f)
        {
            // Check that the layouts are compatible
            if (*a->get_tensor_view_layout() != *b->get_tensor_view_layout())
            {
                throw ngraph_error("Cannot compare tensors with different layouts");
            }

            if (a->get_shape() != b->get_shape())
            {
                return false;
            }

            return all_close(read_vector<T>(a), read_vector<T>(b), rtol, atol);
        }

        /// @brief Same as numpy.allclose
        /// @param as First tensors to compare
        /// @param bs Second tensors to compare
        /// @param rtol Relative tolerance
        /// @param atol Absolute tolerance
        /// Returns true if shapes match and for all elements, |a_i-b_i| <= atol + rtol*|b_i|.
        template <typename T>
        bool all_close(const std::vector<std::shared_ptr<ngraph::runtime::TensorView>>& as,
                       const std::vector<std::shared_ptr<ngraph::runtime::TensorView>>& bs,
                       T rtol,
                       T atol)
        {
            if (as.size() != bs.size())
            {
                return false;
            }
            for (size_t i = 0; i < as.size(); ++i)
            {
                if (!all_close(as[i], bs[i], rtol, atol))
                {
                    return false;
                }
            }
            return true;
        }
    }
}
