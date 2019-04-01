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

#include <random>

#include "gtest/gtest.h"

#include "ngraph/runtime/aligned_buffer.hpp"
#include "ngraph/type/bfloat16.hpp"
#include "util/float_util.hpp"

using namespace std;
using namespace ngraph;

//***********************
// NOTE
//***********************
// This test uses exact comparisons of floating point values. It is testing for bit-exact
// creation and truncation/rounding of bfloat16 values.
TEST(bfloat16, conversions)
{
    bfloat16 bf;
    string source_string;
    string bf_string;

    // 1.f, the ground-truth value
    source_string = "0  01111111  000 0000";
    bf = test::bits_to_bfloat16(source_string);
    EXPECT_EQ(bf, bfloat16(1.0));
    bf_string = test::bfloat16_to_bits(bf);
    EXPECT_STREQ(source_string.c_str(), bf_string.c_str());

    // 1.03125f, the exact upper bound
    source_string = "0  01111111  000 0100";
    bf = test::bits_to_bfloat16(source_string);
    EXPECT_EQ(bf, bfloat16(1.03125));
    bf_string = test::bfloat16_to_bits(bf);
    EXPECT_STREQ(source_string.c_str(), bf_string.c_str());
}

TEST(bfloat16, round_to_nearest)
{
    // 1.03515625f, the next representable number which should round up
    string fstring = "0  01111111  000 0100 1000 0000 0000 0000";
    float fvalue = test::bits_to_float(fstring);
    uint16_t bf_trunc = bfloat16::truncate(fvalue);
    uint16_t bf_round = bfloat16::round_to_nearest(fvalue);
    EXPECT_EQ(bf_trunc, bfloat16(1.03125).to_bits());
    EXPECT_EQ(bf_round, bfloat16(1.0390625).to_bits());

    // 1.99609375f, the next representable number which should round up
    fstring = "0  01111111  111 1111 1000 0000 0000 0000";
    fvalue = test::bits_to_float(fstring);
    bf_trunc = bfloat16::truncate(fvalue);
    bf_round = bfloat16::round_to_nearest(fvalue);
    EXPECT_EQ(bf_trunc, bfloat16(1.9921875).to_bits());
    EXPECT_EQ(bf_round, bfloat16(2.0).to_bits());

    // 1.9921875f, the next representable number which should not round up
    fstring = "0  01111111  111 1111 0000 0000 0000 0000";
    fvalue = test::bits_to_float(fstring);
    bf_trunc = bfloat16::truncate(fvalue);
    bf_round = bfloat16::round_to_nearest(fvalue);
    EXPECT_EQ(bf_trunc, bfloat16(1.9921875).to_bits());
    EXPECT_EQ(bf_round, bfloat16(1.9921875).to_bits());
}

TEST(bfloat16, round_to_nearest_even)
{
    string fstring = "0  01111111  000 0100 1000 0000 0000 0000";
    string expected = "0  01111111  000 0100";
    float fvalue = test::bits_to_float(fstring);
    uint16_t bf_round = bfloat16::round_to_nearest_even(fvalue);
    EXPECT_EQ(bf_round, test::bits_to_bfloat16(expected).to_bits());

    fstring = "0  01111111  000 0101 1000 0000 0000 0000";
    expected = "0  01111111  000 0110";
    fvalue = test::bits_to_float(fstring);
    bf_round = bfloat16::round_to_nearest_even(fvalue);
    EXPECT_EQ(bf_round, test::bits_to_bfloat16(expected).to_bits());

    fstring = "0  01111111  000 0101 0000 0000 0000 0000";
    expected = "0  01111111  000 0101";
    fvalue = test::bits_to_float(fstring);
    bf_round = bfloat16::round_to_nearest_even(fvalue);
    EXPECT_EQ(bf_round, test::bits_to_bfloat16(expected).to_bits());

    fstring = "0  01111111  111 1111 1000 0000 0000 0000";
    expected = "0  10000000  000 0000";
    fvalue = test::bits_to_float(fstring);
    bf_round = bfloat16::round_to_nearest_even(fvalue);
    EXPECT_EQ(bf_round, test::bits_to_bfloat16(expected).to_bits());

    fstring = "0  01111111  111 1111 0000 0000 0000 0000";
    expected = "0  01111111  111 1111";
    fvalue = test::bits_to_float(fstring);
    bf_round = bfloat16::round_to_nearest_even(fvalue);
    EXPECT_EQ(bf_round, test::bits_to_bfloat16(expected).to_bits());
}

TEST(bfloat16, to_float)
{
    bfloat16 bf;
    string source_string;

    // 1.f, the ground-truth value
    source_string = "0  01111111  000 0000";
    bf = test::bits_to_bfloat16(source_string);
    float f = static_cast<float>(bf);
    EXPECT_EQ(f, 1.0f);

    // 1.03125f, the exact upper bound
    source_string = "0  01111111  000 0100";
    bf = test::bits_to_bfloat16(source_string);
    f = static_cast<float>(bf);
    EXPECT_EQ(f, 1.03125f);
}

TEST(benchmark, bfloat16)
{
    size_t buffer_size = 128 * 3 * 224 * 224;
    ngraph::runtime::AlignedBuffer data(buffer_size * sizeof(float), 4096);
    float* f = static_cast<float*>(data.get_ptr());
    // vector<float> data(buffer_size);
    mt19937 rng(2112);
    uniform_real_distribution<float> distribution(-300, 300);
    for (size_t i = 0; i < buffer_size; ++i)
    {
        f[i] = distribution(rng);
    }
    NGRAPH_INFO << "buffer size " << data.size() << " floats or " << data.size() * sizeof(float)
                << " bytes";

    {
        ngraph::runtime::AlignedBuffer bf_data(buffer_size * sizeof(bfloat16), 4096);
        bfloat16* p = static_cast<bfloat16*>(bf_data.get_ptr());
        stopwatch timer;
        timer.start();
        for (size_t i = 0; i < buffer_size; ++i)
        {
            p[i] = bfloat16(f[i]);
        }
        timer.stop();
        NGRAPH_INFO << "float to bfloat16 ctor                  " << timer.get_milliseconds()
                    << "ms";
    }

    {
        ngraph::runtime::AlignedBuffer bf_data(buffer_size * sizeof(bfloat16), 4096);
        bfloat16* p = static_cast<bfloat16*>(bf_data.get_ptr());
        stopwatch timer;
        timer.start();
        for (size_t i = 0; i < buffer_size; ++i)
        {
            p[i] = bfloat16::truncate(f[i]);
        }
        timer.stop();
        NGRAPH_INFO << "float to bfloat16 truncate              " << timer.get_milliseconds()
                    << "ms";
    }

    {
        ngraph::runtime::AlignedBuffer bf_data(buffer_size * sizeof(bfloat16), 4096);
        bfloat16* p = static_cast<bfloat16*>(bf_data.get_ptr());
        stopwatch timer;
        timer.start();
        for (size_t i = 0; i < buffer_size; ++i)
        {
            p[i] = bfloat16::round_to_nearest(f[i]);
        }
        timer.stop();
        NGRAPH_INFO << "float to bfloat16 round to nearest      " << timer.get_milliseconds()
                    << "ms";
    }

    {
        ngraph::runtime::AlignedBuffer bf_data(buffer_size * sizeof(bfloat16), 4096);
        bfloat16* p = static_cast<bfloat16*>(bf_data.get_ptr());
        stopwatch timer;
        timer.start();
        for (size_t i = 0; i < buffer_size; ++i)
        {
            p[i] = bfloat16::round_to_nearest_even(f[i]);
        }
        timer.stop();
        NGRAPH_INFO << "float to bfloat16 round to nearest even " << timer.get_milliseconds()
                    << "ms";
    }
}
