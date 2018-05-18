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

#include <memory>
#include <vector>

#include "test_tools.hpp"

namespace ngraph
{
    namespace test
    {
        /// @brief Check if the two floating point number are near
        /// @param a First number to compare
        /// @param b Second number to compare
        /// @param mantissa The mantissa for the underlying number before casting to float
        /// @param bit
        /// @returns true if the two floating point number are near
        ///
        /// s e e e e e e e e m m m m m m m m m m m m m m m m m m m m m m m
        /// |------------bfloat-----------|
        /// |----------------------------float----------------------------|
        ///
        /// bfloat (s1, e8, m7) has 7 + 1 = 8 bits of mantissa or bit_precision
        /// float (s1, e8, m23) has 23 + 1 = 24 bits of mantissa or bit_precision
        /// Picking 8 as the default mantissa for float_near works for bfloat and float
        bool float_near(float a, float b, uint32_t bit_precision = 8, uint32_t bit_tolerance = 2);
    }
}
