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

#include <cmath>

#include "util/all_close_f.hpp"

using namespace std;
using namespace ngraph;

// Returns |a - b| < 2 ^ (a.mantissa - mantissa_bits + 1 + tolerance_bits)
bool test::close_f(float a, float b, int mantissa_bits, int tolerance_bits)
{
    static const float min_value = pow(2.f, -100.f);
    if (!isfinite(a) || !isfinite(b))
    {
        return false;
    }
    a = abs(a) > min_value ? a : min_value;
    b = abs(b) > min_value ? b : min_value;

    int a_e;
    frexp(a, &a_e);

    return abs(a - b) < pow(2.f, static_cast<float>(a_e - mantissa_bits + 1 + tolerance_bits));
}

bool test::all_close_f(const vector<float>& a,
                       const vector<float>& b,
                       int mantissa_bits,
                       int tolerance_bits)
{
    bool rc = true;
    if (a.size() != b.size())
    {
        throw ngraph_error("a.size() != b.size() for all_close comparison.");
    }
    for (size_t i = 0; i < a.size(); ++i)
    {
        bool is_close_f = close_f(a[i], b[i], mantissa_bits, tolerance_bits);
        if (!is_close_f)
        {
            NGRAPH_INFO << a[i] << " !≈ " << b[i];
            rc = false;
        }
        else
        {
            NGRAPH_INFO << a[i] << " ≈ " << b[i];
        }
    }
    return rc;
}

union FloatUnion {
    float f;
    uint32_t i;
};

bool test::close_g(float a, float b, int mantissa_bits, int tolerance_bits)
{
    FloatUnion a_fu{a};
    FloatUnion b_fu{b};
    uint32_t a_uint = a_fu.i;
    uint32_t b_uint = b_fu.i;

    // If negative: convert to two's complement
    // If positive: mask with sign bit
    uint32_t sign_mask = static_cast<uint32_t>(1U) << 31;
    a_uint = (sign_mask & a_uint) ? (~a_uint + 1) : (sign_mask | a_uint);
    b_uint = (sign_mask & b_uint) ? (~b_uint + 1) : (sign_mask | b_uint);

    uint32_t distance = (a_uint >= b_uint) ? (a_uint - b_uint) : (b_uint - a_uint);

    // e.g. for float with 24 bit mantissa and 2 bit accuracy
    // tolerance_bit_shift = 32 -           (1 +  8 + (24 -     1         ) - 2)
    //                       float_length    sign exp  mantissa implicit 1    tolerance_bits
    uint32_t tolerance_bit_shift = 32 - (1 + 8 + (mantissa_bits - 1) - tolerance_bits);
    uint32_t tolerance = static_cast<uint32_t>(1U) << tolerance_bit_shift;

    return distance <= tolerance;
}

bool test::all_close_g(const vector<float>& a,
                       const vector<float>& b,
                       int mantissa_bits,
                       int tolerance_bits)
{
    bool rc = true;
    if (a.size() != b.size())
    {
        throw ngraph_error("a.size() != b.size() for all_close comparison.");
    }
    for (size_t i = 0; i < a.size(); ++i)
    {
        bool is_close_f = close_g(a[i], b[i], mantissa_bits, tolerance_bits);
        if (!is_close_f)
        {
            NGRAPH_INFO << a[i] << " !≈ " << b[i];
            rc = false;
        }
        else
        {
            NGRAPH_INFO << a[i] << " ≈ " << b[i];
        }
    }
    return rc;
}
