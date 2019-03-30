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

#include "gtest/gtest.h"

#include "ngraph/type/bfloat16.hpp"
#include "util/float_util.hpp"

using namespace std;
using namespace ngraph;

TEST(bfloat16, from_float)
{
    // // 1.f, the ground-truth value
    // float expected = bits_to_float("0  01111111  000 0000 0000 0000 0000 0000");
    // float computed;

    // // 1.03125f, the exact upper bound
    // computed = bits_to_float("0  01111111  000 0100 0000 0000 0000 0000");
    // EXPECT_TRUE(test::close_f(expected, computed, tolerance_bits));
    // EXPECT_TRUE(
    //     test::all_close_f(vector<float>({expected}), vector<float>({computed}), tolerance_bits));

    // // 1.031250119f, the next representable number bigger than upper bound
    // computed = bits_to_float("0  01111111  000 0100 0000 0000 0000 0001");
    // EXPECT_FALSE(test::close_f(expected, computed, tolerance_bits));
    // EXPECT_FALSE(
    //     test::all_close_f(vector<float>({expected}), vector<float>({computed}), tolerance_bits));

    // // 0.984375f, the exact lower bound
    // computed = bits_to_float("0  01111110  111 1100 0000 0000 0000 0000");
    // EXPECT_TRUE(test::close_f(expected, computed, tolerance_bits));
    // EXPECT_TRUE(
    //     test::all_close_f(vector<float>({expected}), vector<float>({computed}), tolerance_bits));

    // // 0.9843749404f, the next representable number smaller than lower bound
    // computed = bits_to_float("0  01111110  111 1011 1111 1111 1111 1111");
    // EXPECT_FALSE(test::close_f(expected, computed, tolerance_bits));
    // EXPECT_FALSE(
    //     test::all_close_f(vector<float>({expected}), vector<float>({computed}), tolerance_bits));
}

TEST(bfloat16, from_float_rounding)
{
}

TEST(bfloat16, to_float)
{
}
