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

#include <bitset>
#include <cmath>
#include <limits>
#include <sstream>

#include "gtest/gtest.h"

#include "ngraph/ngraph.hpp"
#include "util/all_close_f.hpp"

using namespace std;
using namespace ngraph;

union FloatUnion {
    float f;
    uint32_t i;
};

string float_to_bits(float f)
{
    FloatUnion fu{f};
    stringstream ss;
    ss << bitset<32>(fu.i);
    return ss.str();
}

float bits_to_float(const string& s)
{
    if (s.size() != 32)
    {
        throw ngraph_error("Input s length must be 32");
    }
    bitset<32> bs(s);
    FloatUnion fu;
    fu.i = static_cast<uint32_t>(bs.to_ulong());
    return fu.f;
}

// Test the exact bounds near +0.f
//
// With mantissa_bits = 8, tolerance_bits = 2
//
//                           Targeted bit
//                           |
//                           v
// s e e e e e e e e m m m m m m m m m m m m m m m m m m m m m m m
//               =>|      8      |
//                           | 2 |<=
//
// [Upper bound]
//                           Add 1 at this bit
//                           |
//                           v
// 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
// +                         1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
// ---------------------------------------------------------------
// 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
//
// [Lower bound]
//                           Minus 1 at this bit
//                           |
//                           v
// 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
// -                         1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
// ---------------------------------------------------------------
// 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
//
// Convert to 2's compliment
// 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
//
// Mask the sign bit
// 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
TEST(all_close_f, mantissa_8_near_0)
{
    // 0.f, the ground-truth value
    float expected = bits_to_float("00000000000000000000000000000000");
    float computed;

    // ~3.67342E-40, the exact upper bound
    computed = bits_to_float("00000000000001000000000000000000");
    EXPECT_TRUE(test::close_f(expected, computed, 8, 2));

    // ~3.67343E-40, the next representable number bigger than upper bound
    computed = bits_to_float("00000000000001000000000000000001");
    EXPECT_TRUE(!test::close_f(expected, computed, 8, 2));

    // ~-3.67342E-40, the exact lower bound
    computed = bits_to_float("10000000000001000000000000000000");
    EXPECT_TRUE(test::close_f(expected, computed, 8, 2));

    // ~-3.67343E-40, the next representable number smaller than lower bound
    computed = bits_to_float("10000000000001000000000000000001");
    EXPECT_TRUE(!test::close_f(expected, computed, 8, 2));
}

// Test the exact bounds near -0.f
//
// With mantissa_bits = 8, tolerance_bits = 2
//
//                           Targeted bit
//                           |
//                           v
// s e e e e e e e e m m m m m m m m m m m m m m m m m m m m m m m
//               =>|      8      |
//                           | 2 |<=
//
// [Upper bound]
//                           Minus 1 at this bit
//                           |
//                           v
// 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
// +                         1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
// ---------------------------------------------------------------
// 0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
//
// Convert to 2's compliment
// 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
//
// Mask off sign bit
// 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
//
// [Lower bound]
//                           Add 1 at this bit
//                           |
//                           v
// 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
// -                         1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
// ---------------------------------------------------------------
// 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
TEST(all_close_f, mantissa_8_near_n0)
{
    // 0.f, the ground-truth value
    float expected = bits_to_float("10000000000000000000000000000000");
    float computed;

    // ~3.67342E-40, the exact upper bound
    computed = bits_to_float("00000000000001000000000000000000");
    EXPECT_TRUE(test::close_f(expected, computed, 8, 2));

    // ~3.67343E-40, the next representable number bigger than upper bound
    computed = bits_to_float("00000000000001000000000000000001");
    EXPECT_TRUE(!test::close_f(expected, computed, 8, 2));

    // ~-3.67342E-40, the exact lower bound
    computed = bits_to_float("10000000000001000000000000000000");
    EXPECT_TRUE(test::close_f(expected, computed, 8, 2));

    // ~-3.67343E-40, the next representable number smaller than lower bound
    computed = bits_to_float("10000000000001000000000000000001");
    EXPECT_TRUE(!test::close_f(expected, computed, 8, 2));
}

// Test the exact bounds near 1.f
//
// With mantissa_bits = 8, tolerance_bits = 2
//
//                           Targeted bit
//                           |
//                           v
// s e e e e e e e e m m m m m m m m m m m m m m m m m m m m m m m
//               =>|      8      |
//                           | 2 |<=
//
// [Upper bound]
//                           Add 1 at this bit to get upper bound
//                           |
//                           v
// 0 0 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
// +                         1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
// ---------------------------------------------------------------
// 0 0 1 1 1 1 1 1 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
//
// [Lower bound]
//                           Minus 1 at this bit to get lower bound
//                           |
//                           v
// 0 0 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
// -                         1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
// ---------------------------------------------------------------
// 0 0 1 1 1 1 1 1 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
TEST(all_close_f, mantissa_8_near_1)
{
    // 1.f, the ground-truth value
    float expected = bits_to_float("00111111100000000000000000000000");
    float computed;

    // 1.03125f, the exact upper bound
    computed = bits_to_float("00111111100001000000000000000000");
    EXPECT_TRUE(test::close_f(expected, computed, 8, 2));

    // 1.031250119f, the next representable number bigger than upper bound
    computed = bits_to_float("00111111100001000000000000000001");
    EXPECT_TRUE(!test::close_f(expected, computed, 8, 2));

    // 0.984375f, the exact lower bound
    computed = bits_to_float("00111111011111000000000000000000");
    EXPECT_TRUE(test::close_f(expected, computed, 8, 2));

    // 0.9843749404f, the next representable number smaller than lower bound
    computed = bits_to_float("00111111011110111111111111111111");
    EXPECT_TRUE(!test::close_f(expected, computed, 8, 2));
}

// Test the exact bounds near -1.f
//
// With mantissa_bits = 8, tolerance_bits = 2
//
//                           Targeted bit
//                           |
//                           v
// s e e e e e e e e m m m m m m m m m m m m m m m m m m m m m m m
//               =>|      8      |
//                           | 2 |<=
//
// [Upper bound]
//                           Minus 1 at this bit
//                           |
//                           v
// 1 0 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
// +                         1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
// ---------------------------------------------------------------
// 1 0 1 1 1 1 1 1 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
//
// [Lower bound]
//                           Add 1 at this bit
//                           |
//                           v
// 1 0 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
// -                         1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
// ---------------------------------------------------------------
// 1 0 1 1 1 1 1 1 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
TEST(all_close_f, mantissa_8_near_n1)
{
    // -1.f, the ground-truth value
    float expected = bits_to_float("10111111100000000000000000000000");
    float computed;

    // -0.984375f, the exact upper bound
    computed = bits_to_float("10111111011111000000000000000000");
    EXPECT_TRUE(test::close_f(expected, computed, 8, 2));

    // -0.984374940395355224609375f, the next representable number bigger than upper bound
    computed = bits_to_float("10111111011110111111111111111111");
    EXPECT_TRUE(!test::close_f(expected, computed, 8, 2));

    // -1.03125f, the exact lower bound
    computed = bits_to_float("10111111100001000000000000000000");
    EXPECT_TRUE(test::close_f(expected, computed, 8, 2));

    // -1.03125011920928955078125f, the next representable number smaller than lower bound
    computed = bits_to_float("10111111100001000000000000000001");
    EXPECT_TRUE(!test::close_f(expected, computed, 8, 2));
}

// For intuitive understanding of tightness of bounds in decimal
// Test bounds near 0, 1, 10, 100, 1000 with mantissa_bits = 8, tolerance_bits = 2
TEST(all_close_f, mantissa_8_near_0_1_10_100_1000)
{
    float expected;
    float upper_bound;
    float bigger_than_upper_bound;
    float lower_bound;
    float smaller_than_lower_bound;

    expected = 0.f; // 00000000000000000000000000000000
    upper_bound = 3.67342e-40f; // 00000000000001000000000000000000, approximated
    bigger_than_upper_bound = 3.67343e-40f; // 00000000000001000000000000000001, approximated
    lower_bound = -3.67342e-40f; // 10000000000001000000000000000000, approximated
    smaller_than_lower_bound = 3.67343e-40f; // 10000000000001000000000000000001, approximated
    EXPECT_TRUE(test::close_f(expected, upper_bound, 8, 2));
    EXPECT_TRUE(!test::close_f(expected, bigger_than_upper_bound, 8, 2));
    EXPECT_TRUE(test::close_f(expected, lower_bound, 8, 2));
    EXPECT_TRUE(!test::close_f(expected, smaller_than_lower_bound, 8, 2));

    expected = 1.f; // 00111111100000000000000000000000
    upper_bound = 1.03125f; // 00111111100001000000000000000000
    bigger_than_upper_bound = 1.031250119f; // 00111111100001000000000000000001
    lower_bound = 0.984375f; // 00111111011111000000000000000000
    smaller_than_lower_bound = 0.9843749404f; // 00111111011110111111111111111111
    EXPECT_TRUE(test::close_f(expected, upper_bound, 8, 2));
    EXPECT_TRUE(!test::close_f(expected, bigger_than_upper_bound, 8, 2));
    EXPECT_TRUE(test::close_f(expected, lower_bound, 8, 2));
    EXPECT_TRUE(!test::close_f(expected, smaller_than_lower_bound, 8, 2));

    // expected = 0.f; // 00000000000000000000000000000000
    // upper_bound = 3.67342e-40f; // 00000000000001000000000000000000
    // bigger_than_upper_bound = 3.67343e-40f; // 00000000000001000000000000000001
    // lower_bound = -3.67342e-40f; // 10000000000001000000000000000000
    // smaller_than_lower_bound = 3.67343e-40f; // 10000000000001000000000000000001
    // EXPECT_TRUE(test::close_f(expected, upper_bound, 8, 2));
    // EXPECT_TRUE(!test::close_f(expected, bigger_than_upper_bound, 8, 2));
    // EXPECT_TRUE(test::close_f(expected, lower_bound, 8, 2));
    // EXPECT_TRUE(!test::close_f(expected, smaller_than_lower_bound, 8, 2));
}
