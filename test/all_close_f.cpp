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

// TEST(all_close_f, mantissa_24)
// {
//     float val0 = bits_to_float("00000000000000000000000000000000");
//     float val_0 = bits_to_float("00000000000000000000000000000001");
//     float val_1 = bits_to_float("00000000000000000000000000000010");
//     float val_2 = bits_to_float("00000000000000000000000000000011");
//     float val_3 = bits_to_float("00000000000000000000000000000100");
//     float val_5 = bits_to_float("00000000000000000000000000000101");

//     float val_0 = bits_to_float("10000000000000000000000000000000");
//     float val_0 = bits_to_float("10000000000000000000000000000001");
//     float val_1 = bits_to_float("10000000000000000000000000000010");
//     float val_2 = bits_to_float("10000000000000000000000000000011");
//     float val_3 = bits_to_float("10000000000000000000000000000100");
//     float val_5 = bits_to_float("10000000000000000000000000000101");

//     NGRAPH_INFO << val0;
//     NGRAPH_INFO << val_0;
//     NGRAPH_INFO << val_1;
//     NGRAPH_INFO << val_2;
//     NGRAPH_INFO << val_3;
//     NGRAPH_INFO << val_5;

//     NGRAPH_INFO << val_0;
//     NGRAPH_INFO << val_0;
//     NGRAPH_INFO << val_1;
//     NGRAPH_INFO << val_2;
//     NGRAPH_INFO << val_3;
//     NGRAPH_INFO << val_5;

//     NGRAPH_INFO << test::close_f(0.f, val0, 24);
//     NGRAPH_INFO << test::close_f(0.f, val_0, 24);
//     NGRAPH_INFO << test::close_f(0.f, val_1, 24);
//     NGRAPH_INFO << test::close_f(0.f, val_2, 24);
//     NGRAPH_INFO << test::close_f(0.f, val_3, 24);
//     NGRAPH_INFO << test::close_f(0.f, val_5, 24);

//     NGRAPH_INFO << test::close_f(0.f, val_0, 24);
//     NGRAPH_INFO << test::close_f(0.f, val_0, 24);
//     NGRAPH_INFO << test::close_f(0.f, val_1, 24);
//     NGRAPH_INFO << test::close_f(0.f, val_2, 24);
//     NGRAPH_INFO << test::close_f(0.f, val_3, 24);
//     NGRAPH_INFO << test::close_f(0.f, val_5, 24);
// }

TEST(all_close_f, mantissa_8_near_n1)
{
    float val0 = bits_to_float("00000000000000000000000000000000");
    float val_0 = bits_to_float("00000000000001000000000000000000");
    float val_1 = bits_to_float("00000000000001000000000000000001");
    float val_2 = bits_to_float("00000000000010000000000000000000");

    NGRAPH_INFO << test::close_f(val0, val0);
    NGRAPH_INFO << test::close_f(val0, val_0);
    NGRAPH_INFO << test::close_f(val0, val_1);
    NGRAPH_INFO << test::close_f(val0, val_2);
}

TEST(all_close_f, mantissa_8_near_1)
{
    // This test gives an exact bound when a number is considered to be near 1.f
    //
    // With mantissa_bits = 8, tolerance_bits = 2
    //
    //                           Add 1 at this bit to get upper bound
    //                           Minus 1 at this bit to get lower bound
    //                           |
    //                           v
    // 0 0 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    // s e e e e e e e e m m m m m m m m m m m m m m m m m m m m m m m
    //               =>|      8      |
    //                           | 2 |<=
    //
    // [Upper bound]
    // 0 0 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    // +                         1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    // ---------------------------------------------------------------
    // 0 0 1 1 1 1 1 1 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    //
    // [Lower bound]
    // 0 0 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    // -                         1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    // ---------------------------------------------------------------
    // 0 0 1 1 1 1 1 1 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0

    // 1.f, the ground-truth value
    float expected_val = bits_to_float("00111111100000000000000000000000");
    float computed_val;

    // 1.03125f, upper bound
    computed_val = bits_to_float("00111111100001000000000000000000");
    EXPECT_TRUE(test::close_f(expected_val, computed_val, 8, 2));

    // 1.031250119f, bigger than upper bound
    computed_val = bits_to_float("00111111100001000000000000000001");
    EXPECT_TRUE(!test::close_f(expected_val, computed_val, 8, 2));

    // 0.984375f, lower bound
    computed_val = bits_to_float("00111111011111000000000000000000");
    EXPECT_TRUE(test::close_f(expected_val, computed_val, 8, 2));

    // 0.9843749404f, smaller than lower bound
    computed_val = bits_to_float("00111111011110111111111111111111");
    EXPECT_TRUE(!test::close_f(expected_val, computed_val, 8, 2));
}
