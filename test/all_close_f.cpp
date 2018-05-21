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
    NGRAPH_INFO << test::close_f(0.0f, 0.1f);
    NGRAPH_INFO << test::close_f(0.0f, 0.01f);
    NGRAPH_INFO << test::close_f(0.0f, 0.001f);
    NGRAPH_INFO << test::close_f(0.0f, 0.0001f);
    NGRAPH_INFO << test::close_f(0.0f, 0.00001f);
    NGRAPH_INFO << test::close_f(0.0f, 0.000001f);
    NGRAPH_INFO << test::close_f(0.0f, 0.0000001f);
}
