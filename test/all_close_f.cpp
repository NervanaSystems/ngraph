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

string float_to_bit_string(float f)
{
    FloatUnion fu{f};
    stringstream ss;
    ss << bitset<32>(fu.i);
    return ss.str();
}

float bit_string_to_float(const string& s)
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

TEST(all_close_f, bit_string_conversion)
{
    EXPECT_EQ(float_to_bit_string(8), "01000001000000000000000000000000");
    EXPECT_EQ(bit_string_to_float("01000001000000000000000000000000"), 8);
    EXPECT_EQ(float_to_bit_string(-8), "11000001000000000000000000000000");
    EXPECT_EQ(bit_string_to_float("11000001000000000000000000000000"), -8);
}

TEST(all_close_f, example_compare)
{
    // float lhs = 1.5;
    float rhs = 1.75;
    NGRAPH_INFO << float_to_bit_string(8);
    NGRAPH_INFO << float_to_bit_string(rhs);

    float param = 8.;
    float mantissa;
    int exponent;

    mantissa = frexp(param, &exponent);
    NGRAPH_INFO << "mantissa " << mantissa;
    NGRAPH_INFO << "exponent " << exponent;
}

TEST(all_close_f, float_close_basic)
{
    NGRAPH_INFO << test::close_f(1.5f, 1.5f);
    NGRAPH_INFO << test::close_f(1.5f, 1.52f);
    NGRAPH_INFO << test::close_f(1.5f, 1.53f);
    NGRAPH_INFO << test::close_f(1.5f, 1.54f);
    NGRAPH_INFO << test::close_f(1.5f, 1.55f);
    NGRAPH_INFO << test::close_f(1.5f, 1.56f);
    NGRAPH_INFO << test::close_f(1.5f, 1.57f);
}

TEST(all_close_f, float_close_zero)
{
    NGRAPH_INFO << test::close_f(0.0f, 1e-8f);
    NGRAPH_INFO << test::close_f(1e-8f, 0.0f);
    NGRAPH_INFO << test::close_f(1e-7f, 0.0f);
    NGRAPH_INFO << test::close_f(1e-6f, 0.0f);
    NGRAPH_INFO << test::close_f(1e-5f, 0.0f);
    NGRAPH_INFO << test::close_f(1e-4f, 0.0f);
    NGRAPH_INFO << test::close_f(1e-3f, 0.0f);
    NGRAPH_INFO << test::close_f(1e-2f, 0.0f);
    NGRAPH_INFO << test::close_f(1e-1f, 0.0f);
}

TEST(all_close_f, float_close_zero_b)
{
    NGRAPH_INFO << float_to_bit_string(0.0f);
    NGRAPH_INFO << float_to_bit_string(0.015625f);
    NGRAPH_INFO << test::close_f(0.0f, 0.015625f);
    NGRAPH_INFO << test::close_f(0.015625f, 0.0f);
    NGRAPH_INFO << test::close_f(0.00006103515f, 0.0f);
}

TEST(all_close_f, gtest_float_equal)
{
    float val0 = bit_string_to_float("00000000000000000000000000000000");
    float val1 = bit_string_to_float("00000000000000000000000000000001");
    float val2 = bit_string_to_float("00000000000000000000000000000010");
    float val3 = bit_string_to_float("00000000000000000000000000000011");
    float val4 = bit_string_to_float("00000000000000000000000000000100");
    float val5 = bit_string_to_float("00000000000000000000000000000101");

    float val_0 = bit_string_to_float("10000000000000000000000000000000");
    float val_1 = bit_string_to_float("10000000000000000000000000000001");
    float val_2 = bit_string_to_float("10000000000000000000000000000010");
    float val_3 = bit_string_to_float("10000000000000000000000000000011");
    float val_4 = bit_string_to_float("10000000000000000000000000000100");
    float val_5 = bit_string_to_float("10000000000000000000000000000101");

    NGRAPH_INFO << val0;
    NGRAPH_INFO << val1;
    NGRAPH_INFO << val2;
    NGRAPH_INFO << val3;
    NGRAPH_INFO << val4;
    NGRAPH_INFO << val5;

    NGRAPH_INFO << val_0;
    NGRAPH_INFO << val_1;
    NGRAPH_INFO << val_2;
    NGRAPH_INFO << val_3;
    NGRAPH_INFO << val_4;
    NGRAPH_INFO << val_5;

    EXPECT_FLOAT_EQ(0.f, val0);
    EXPECT_FLOAT_EQ(0.f, val1);
    EXPECT_FLOAT_EQ(0.f, val2);
    EXPECT_FLOAT_EQ(0.f, val3);
    EXPECT_FLOAT_EQ(0.f, val4);
    EXPECT_FLOAT_EQ(0.f, val5);

    EXPECT_FLOAT_EQ(0.f, val_0);
    EXPECT_FLOAT_EQ(0.f, val_1);
    EXPECT_FLOAT_EQ(0.f, val_2);
    EXPECT_FLOAT_EQ(0.f, val_3);
    EXPECT_FLOAT_EQ(0.f, val_4);
    EXPECT_FLOAT_EQ(0.f, val_5);
}

TEST(close_g, mantissa_24)
{
    float val0 = bit_string_to_float("00000000000000000000000000000000");
    float val1 = bit_string_to_float("00000000000000000000000000000001");
    float val2 = bit_string_to_float("00000000000000000000000000000010");
    float val3 = bit_string_to_float("00000000000000000000000000000011");
    float val4 = bit_string_to_float("00000000000000000000000000000100");
    float val5 = bit_string_to_float("00000000000000000000000000000101");

    float val_0 = bit_string_to_float("10000000000000000000000000000000");
    float val_1 = bit_string_to_float("10000000000000000000000000000001");
    float val_2 = bit_string_to_float("10000000000000000000000000000010");
    float val_3 = bit_string_to_float("10000000000000000000000000000011");
    float val_4 = bit_string_to_float("10000000000000000000000000000100");
    float val_5 = bit_string_to_float("10000000000000000000000000000101");

    NGRAPH_INFO << val0;
    NGRAPH_INFO << val1;
    NGRAPH_INFO << val2;
    NGRAPH_INFO << val3;
    NGRAPH_INFO << val4;
    NGRAPH_INFO << val5;

    NGRAPH_INFO << val_0;
    NGRAPH_INFO << val_1;
    NGRAPH_INFO << val_2;
    NGRAPH_INFO << val_3;
    NGRAPH_INFO << val_4;
    NGRAPH_INFO << val_5;

    NGRAPH_INFO << test::close_g(0.f, val0, 24);
    NGRAPH_INFO << test::close_g(0.f, val1, 24);
    NGRAPH_INFO << test::close_g(0.f, val2, 24);
    NGRAPH_INFO << test::close_g(0.f, val3, 24);
    NGRAPH_INFO << test::close_g(0.f, val4, 24);
    NGRAPH_INFO << test::close_g(0.f, val5, 24);

    NGRAPH_INFO << test::close_g(0.f, val_0, 24);
    NGRAPH_INFO << test::close_g(0.f, val_1, 24);
    NGRAPH_INFO << test::close_g(0.f, val_2, 24);
    NGRAPH_INFO << test::close_g(0.f, val_3, 24);
    NGRAPH_INFO << test::close_g(0.f, val_4, 24);
    NGRAPH_INFO << test::close_g(0.f, val_5, 24);
}

TEST(close_g, mantissa_8_close_zero)
{
    float val0 = bit_string_to_float("00000000000000000000000000000000");
    float val1 = bit_string_to_float("00000000000001000000000000000000");
    float val2 = bit_string_to_float("00000000000001000000000000000001");
    float val3 = bit_string_to_float("00000000000010000000000000000000");

    NGRAPH_INFO << test::close_g(val0, val0);
    NGRAPH_INFO << test::close_g(val0, val1);
    NGRAPH_INFO << test::close_g(val0, val2);
    NGRAPH_INFO << test::close_g(val0, val3);
}

TEST(close_g, mantissa_8_close_one)
{
    // 1.f
    float val0 = bit_string_to_float("00111111100000000000000000000000");
    // Numbers close to 1.f+
    float val1 = bit_string_to_float("00111111100001000000000000000000"); // close
    float val2 = bit_string_to_float("00111111100001000000000000000001"); // not close
    float val3 = bit_string_to_float("00111111100010000000000000000000"); // not close
    // Numbers close to 1.f-
    float val4 = bit_string_to_float("00111111011111000000000000000000"); // close
    float val5 = bit_string_to_float("00111111011110111111111111111111"); // not close
    float val6 = bit_string_to_float("00111111011110000000000000000000"); // not close

    NGRAPH_INFO << "val " << setprecision(10) << val0;
    NGRAPH_INFO << test::close_g(val0, val0);
    NGRAPH_INFO << "val " << setprecision(10) << val1;
    NGRAPH_INFO << test::close_g(val0, val1);
    NGRAPH_INFO << "val " << setprecision(10) << val2;
    NGRAPH_INFO << test::close_g(val0, val2);
    NGRAPH_INFO << "val " << setprecision(10) << val3;
    NGRAPH_INFO << test::close_g(val0, val3);
    NGRAPH_INFO << "val " << setprecision(10) << val4;
    NGRAPH_INFO << test::close_g(val0, val4);
    NGRAPH_INFO << "val " << setprecision(10) << val5;
    NGRAPH_INFO << test::close_g(val0, val5);
    NGRAPH_INFO << "val " << setprecision(10) << val6;
    NGRAPH_INFO << test::close_g(val0, val6);
}
