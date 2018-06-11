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
        throw ngraph_error("Input length must be 32");
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
    EXPECT_FALSE(test::close_f(expected, computed, 8, 2));

    // ~-3.67342E-40, the exact lower bound
    computed = bits_to_float("10000000000001000000000000000000");
    EXPECT_TRUE(test::close_f(expected, computed, 8, 2));

    // ~-3.67343E-40, the next representable number smaller than lower bound
    computed = bits_to_float("10000000000001000000000000000001");
    EXPECT_FALSE(test::close_f(expected, computed, 8, 2));
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
// -                         1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
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
// +                         1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
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
    EXPECT_FALSE(test::close_f(expected, computed, 8, 2));

    // ~-3.67342E-40, the exact lower bound
    computed = bits_to_float("10000000000001000000000000000000");
    EXPECT_TRUE(test::close_f(expected, computed, 8, 2));

    // ~-3.67343E-40, the next representable number smaller than lower bound
    computed = bits_to_float("10000000000001000000000000000001");
    EXPECT_FALSE(test::close_f(expected, computed, 8, 2));
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
    EXPECT_FALSE(test::close_f(expected, computed, 8, 2));

    // 0.984375f, the exact lower bound
    computed = bits_to_float("00111111011111000000000000000000");
    EXPECT_TRUE(test::close_f(expected, computed, 8, 2));

    // 0.9843749404f, the next representable number smaller than lower bound
    computed = bits_to_float("00111111011110111111111111111111");
    EXPECT_FALSE(test::close_f(expected, computed, 8, 2));
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
// -                         1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
// ---------------------------------------------------------------
// 1 0 1 1 1 1 1 1 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
//
// [Lower bound]
//                           Add 1 at this bit
//                           |
//                           v
// 1 0 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
// +                         1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
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
    EXPECT_FALSE(test::close_f(expected, computed, 8, 2));

    // -1.03125f, the exact lower bound
    computed = bits_to_float("10111111100001000000000000000000");
    EXPECT_TRUE(test::close_f(expected, computed, 8, 2));

    // -1.03125011920928955078125f, the next representable number smaller than lower bound
    computed = bits_to_float("10111111100001000000000000000001");
    EXPECT_FALSE(test::close_f(expected, computed, 8, 2));
}

// For intuitive understanding of tightness of bounds in decimal
// Test bounds near 0, 1, 10, 100, 1000 with mantissa_bits = 8, tolerance_bits = 2
//
//                           Targeted bit
//                           |
//                           v
// s e e e e e e e e m m m m m m m m m m m m m m m m m m m m m m m
//               =>|      8      |
//                           | 2 |<=
TEST(all_close_f, mantissa_8_near_0_1_10_100_1000)
{
    float expected;
    float upper_bound;
    float bigger_than_upper_bound;
    float lower_bound;
    float smaller_than_lower_bound;

    // Bounds around 0: 0 +- 3.67e-40
    expected = 0.f;                          // 00000000000000000000000000000000
    upper_bound = 3.67342e-40f;              // 00000000000001000000000000000000, approximated
    bigger_than_upper_bound = 3.67343e-40f;  // 00000000000001000000000000000001, approximated
    lower_bound = -3.67342e-40f;             // 10000000000001000000000000000000, approximated
    smaller_than_lower_bound = 3.67343e-40f; // 10000000000001000000000000000001, approximated
    EXPECT_TRUE(test::close_f(expected, upper_bound, 8, 2));
    EXPECT_FALSE(test::close_f(expected, bigger_than_upper_bound, 8, 2));
    EXPECT_TRUE(test::close_f(expected, lower_bound, 8, 2));
    EXPECT_FALSE(test::close_f(expected, smaller_than_lower_bound, 8, 2));

    // Bounds around 1: 1 +- 0.03
    expected = 1.f;                           // 00111111100000000000000000000000
    upper_bound = 1.03125f;                   // 00111111100001000000000000000000
    bigger_than_upper_bound = 1.031250119f;   // 00111111100001000000000000000001
    lower_bound = 0.984375f;                  // 00111111011111000000000000000000
    smaller_than_lower_bound = 0.9843749404f; // 00111111011110111111111111111111
    EXPECT_TRUE(test::close_f(expected, upper_bound, 8, 2));
    EXPECT_FALSE(test::close_f(expected, bigger_than_upper_bound, 8, 2));
    EXPECT_TRUE(test::close_f(expected, lower_bound, 8, 2));
    EXPECT_FALSE(test::close_f(expected, smaller_than_lower_bound, 8, 2));

    // Bounds around 10: 10 +- 0.25
    expected = 10.f;                                    // 01000001001000000000000000000000
    upper_bound = 10.25f;                               // 01000001001001000000000000000000
    bigger_than_upper_bound = 10.25000095367431640625f; // 01000001001001000000000000000001
    lower_bound = 9.75f;                                // 01000001000111000000000000000000
    smaller_than_lower_bound = 9.74999904632568359375f; // 01000001000110111111111111111111
    EXPECT_TRUE(test::close_f(expected, upper_bound, 8, 2));
    EXPECT_FALSE(test::close_f(expected, bigger_than_upper_bound, 8, 2));
    EXPECT_TRUE(test::close_f(expected, lower_bound, 8, 2));
    EXPECT_FALSE(test::close_f(expected, smaller_than_lower_bound, 8, 2));

    // Bounds around 100: 100 +- 2
    expected = 100.f;                                 // 01000010110010000000000000000000
    upper_bound = 102.f;                              // 01000010110011000000000000000000
    bigger_than_upper_bound = 102.00000762939453125f; // 01000010110011000000000000000001
    lower_bound = 98.0f;                              // 01000010110001000000000000000000
    smaller_than_lower_bound = 97.99999237060546875f; // 01000010110000111111111111111111
    EXPECT_TRUE(test::close_f(expected, upper_bound, 8, 2));
    EXPECT_FALSE(test::close_f(expected, bigger_than_upper_bound, 8, 2));
    EXPECT_TRUE(test::close_f(expected, lower_bound, 8, 2));
    EXPECT_FALSE(test::close_f(expected, smaller_than_lower_bound, 8, 2));

    // Bounds around 1000: 1000 +- 16
    expected = 1000.f;                              // 01000100011110100000000000000000
    upper_bound = 1016.f;                           // 01000100011111100000000000000000
    bigger_than_upper_bound = 1016.00006103515625f; // 01000100011111100000000000000001
    lower_bound = 984.0f;                           // 01000100011101100000000000000000
    smaller_than_lower_bound = 983.99993896484375f; // 01000100011101011111111111111111
    EXPECT_TRUE(test::close_f(expected, upper_bound, 8, 2));
    EXPECT_FALSE(test::close_f(expected, bigger_than_upper_bound, 8, 2));
    EXPECT_TRUE(test::close_f(expected, lower_bound, 8, 2));
    EXPECT_FALSE(test::close_f(expected, smaller_than_lower_bound, 8, 2));
}

// For intuitive understanding of tightness of bounds in decimal
// Test bounds near 0, 1, 10, 100, 1000 with mantissa_bits = 24, tolerance_bits = 2
//
//                                                           Targeted bit
//                                                           |
//                                                           v
// s e e e e e e e e m m m m m m m m m m m m m m m m m m m m m m m
//               =>|                     24                      |
//                                                           | 2 |<=
TEST(all_close_f, mantissa_24_near_0_1_10_100_1000)
{
    float expected;
    float upper_bound;
    float bigger_than_upper_bound;
    float lower_bound;
    float smaller_than_lower_bound;

    // Bounds around 0: 0 +- 5.6e-45
    expected = 0.f;
    upper_bound = bits_to_float("00000000000000000000000000000100");
    bigger_than_upper_bound = bits_to_float("00000000000000000000000000000101");
    lower_bound = bits_to_float("10000000000000000000000000000100");
    smaller_than_lower_bound = bits_to_float("10000000000000000000000000000101");
    EXPECT_TRUE(test::close_f(expected, upper_bound, 24, 2));
    EXPECT_FALSE(test::close_f(expected, bigger_than_upper_bound, 24, 2));
    EXPECT_TRUE(test::close_f(expected, lower_bound, 24, 2));
    EXPECT_FALSE(test::close_f(expected, smaller_than_lower_bound, 24, 2));

    // Bounds around 1: 1 +- 4.77e-7
    expected = 1.f;
    upper_bound = bits_to_float("00111111100000000000000000000100");
    bigger_than_upper_bound = bits_to_float("00111111100000000000000000000101");
    lower_bound = bits_to_float("00111111011111111111111111111100");
    smaller_than_lower_bound = bits_to_float("00111111011111111111111111111011");
    EXPECT_TRUE(test::close_f(expected, upper_bound, 24, 2));
    EXPECT_FALSE(test::close_f(expected, bigger_than_upper_bound, 24, 2));
    EXPECT_TRUE(test::close_f(expected, lower_bound, 24, 2));
    EXPECT_FALSE(test::close_f(expected, smaller_than_lower_bound, 24, 2));

    // Bounds around 10: 10 +- 3.81e-6
    expected = 10.f;
    upper_bound = bits_to_float("01000001001000000000000000000100");
    bigger_than_upper_bound = bits_to_float("01000001001000000000000000000101");
    lower_bound = bits_to_float("01000001000111111111111111111100");
    smaller_than_lower_bound = bits_to_float("01000001000111111111111111111011");
    EXPECT_TRUE(test::close_f(expected, upper_bound, 24, 2));
    EXPECT_FALSE(test::close_f(expected, bigger_than_upper_bound, 24, 2));
    EXPECT_TRUE(test::close_f(expected, lower_bound, 24, 2));
    EXPECT_FALSE(test::close_f(expected, smaller_than_lower_bound, 24, 2));

    // Bounds around 100: 100 +- 3.05e-5
    expected = 100.f;
    upper_bound = bits_to_float("01000010110010000000000000000100");
    bigger_than_upper_bound = bits_to_float("01000010110010000000000000000101");
    lower_bound = bits_to_float("01000010110001111111111111111100");
    smaller_than_lower_bound = bits_to_float("01000010110001111111111111111011");
    EXPECT_TRUE(test::close_f(expected, upper_bound, 24, 2));
    EXPECT_FALSE(test::close_f(expected, bigger_than_upper_bound, 24, 2));
    EXPECT_TRUE(test::close_f(expected, lower_bound, 24, 2));
    EXPECT_FALSE(test::close_f(expected, smaller_than_lower_bound, 24, 2));

    // Bounds around 1000: 1000 +- 2.44e-4
    expected = 1000.f;
    upper_bound = bits_to_float("01000100011110100000000000000100");
    bigger_than_upper_bound = bits_to_float("01000100011110100000000000000101");
    lower_bound = bits_to_float("01000100011110011111111111111100");
    smaller_than_lower_bound = bits_to_float("01000100011110011111111111111011");
    EXPECT_TRUE(test::close_f(expected, upper_bound, 24, 2));
    EXPECT_FALSE(test::close_f(expected, bigger_than_upper_bound, 24, 2));
    EXPECT_TRUE(test::close_f(expected, lower_bound, 24, 2));
    EXPECT_FALSE(test::close_f(expected, smaller_than_lower_bound, 24, 2));
}

TEST(all_close_f, inf_nan)
{
    float zero = 0.f;
    float infinity = numeric_limits<float>::infinity();
    float neg_infinity = -numeric_limits<float>::infinity();
    float quiet_nan = numeric_limits<float>::quiet_NaN();
    float signaling_nan = numeric_limits<float>::signaling_NaN();

    EXPECT_FALSE(test::close_f(zero, infinity));
    EXPECT_FALSE(test::close_f(zero, neg_infinity));
    EXPECT_FALSE(test::close_f(zero, quiet_nan));
    EXPECT_FALSE(test::close_f(zero, signaling_nan));

    EXPECT_FALSE(test::close_f(infinity, infinity));
    EXPECT_FALSE(test::close_f(neg_infinity, neg_infinity));
    EXPECT_FALSE(test::close_f(quiet_nan, quiet_nan));
    EXPECT_FALSE(test::close_f(signaling_nan, signaling_nan));
}
