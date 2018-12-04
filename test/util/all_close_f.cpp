//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
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

#include <climits>
#include <cmath>

#include "util/all_close_f.hpp"

using namespace std;
using namespace ngraph;

union FloatUnion {
    float f;
    uint32_t i;
};

union DoubleUnion {
    double d;
    uint64_t i;
};

uint32_t test::float_distance(float a, float b)
{
    if (!isfinite(a) || !isfinite(b))
    {
        return UINT_MAX;
    }

    FloatUnion a_fu{a};
    FloatUnion b_fu{b};
    uint32_t a_uint = a_fu.i;
    uint32_t b_uint = b_fu.i;

    // A trick to handle both positive and negative numbers, see https://goo.gl/YbdnFQ
    // - If negative: convert to two's complement
    // - If positive: mask with sign bit
    uint32_t sign_mask = static_cast<uint32_t>(1U) << 31;
    a_uint = (sign_mask & a_uint) ? (~a_uint + 1) : (sign_mask | a_uint);
    b_uint = (sign_mask & b_uint) ? (~b_uint + 1) : (sign_mask | b_uint);

    uint32_t distance = (a_uint >= b_uint) ? (a_uint - b_uint) : (b_uint - a_uint);
    return distance;
}

uint64_t test::float_distance(double a, double b)
{
    if (!isfinite(a) || !isfinite(b))
    {
        return UINT_MAX;
    }

    DoubleUnion a_du{a};
    DoubleUnion b_du{b};
    uint64_t a_uint = a_du.i;
    uint64_t b_uint = b_du.i;

    // A trick to handle both positive and negative numbers, see https://goo.gl/YbdnFQ
    // - If negative: convert to two's complement
    // - If positive: mask with sign bit
    uint64_t sign_mask = static_cast<uint64_t>(1U) << 63;
    a_uint = (sign_mask & a_uint) ? (~a_uint + 1) : (sign_mask | a_uint);
    b_uint = (sign_mask & b_uint) ? (~b_uint + 1) : (sign_mask | b_uint);

    uint64_t distance = (a_uint >= b_uint) ? (a_uint - b_uint) : (b_uint - a_uint);
    return distance;
}

bool test::close_f(float a, float b, int mantissa_bits, int tolerance_bits)
{
    // isfinite(a) => !isinf(a) && !isnan(a)
    if (!isfinite(a) || !isfinite(b))
    {
        return false;
    }

    uint32_t distance = float_distance(a, b);

    // e.g. for float with 24 bit mantissa, 2 bit accuracy, and hard-coded 8 bit exponent_bits
    // tolerance_bit_shift = 32 -           (1 +  8 + (24 -     1         ) - 2             )
    //                       float_length    sign exp  mantissa implicit 1    tolerance_bits
    uint32_t tolerance_bit_shift = 32 - (1 + 8 + (mantissa_bits - 1) - tolerance_bits);
    uint32_t tolerance = static_cast<uint32_t>(1U) << tolerance_bit_shift;

    return distance <= tolerance;
}

bool test::close_f(double a, double b, int mantissa_bits, int tolerance_bits)
{
    // isfinite(a) => !isinf(a) && !isnan(a)
    if (!isfinite(a) || !isfinite(b))
    {
        return false;
    }

    uint64_t distance = float_distance(a, b);

    // e.g. for double with 52 bit mantissa, 2 bit accuracy, and hard-coded 11 bit exponent_bits
    // tolerance_bit_shift = 64 -           (1 +  11 + (52 -     1         ) - 2             )
    //                       double_length   sign exp   mantissa implicit 1    tolerance_bits
    uint64_t tolerance_bit_shift = 64 - (1 + 11 + (mantissa_bits - 1) - tolerance_bits);
    uint64_t tolerance = static_cast<uint64_t>(1U) << tolerance_bit_shift;

    return distance <= tolerance;
}

vector<uint32_t> test::float_distances(const vector<float>& a, const vector<float>& b)
{
    if (a.size() != b.size())
    {
        throw ngraph_error("a.size() != b.size() for float_distances comparison.");
    }
    vector<uint32_t> distances(a.size());
    for (size_t i = 0; i < a.size(); ++i)
    {
        distances[i] = float_distance(a[i], b[i]);
    }

    return distances;
}

uint32_t test::matching_mantissa_bits(uint32_t distance)
{
    uint32_t tolerance_needed = distance;

    if (tolerance_needed < 0x80000000)
    {
        // Set up the dominos - turn on all the bits below maximal bit
        tolerance_needed |= tolerance_needed >> 1;
        tolerance_needed |= tolerance_needed >> 2;
        tolerance_needed |= tolerance_needed >> 4;
        tolerance_needed |= tolerance_needed >> 8;
        tolerance_needed |= tolerance_needed >> 16;

        // Tumble the dominos so we end up with next highest bit
        ++tolerance_needed;

        // all_close_f is <= test for tolerance
        if ((tolerance_needed >> 1) == distance)
        {
            tolerance_needed = distance;
        }
    }

    uint32_t tolerance_bit_shift = 0;
    while (tolerance_needed >>= 1)
    {
        ++tolerance_bit_shift;
    }

    // all_close_f calculation of tolerance_bit_shift:
    // e.g. for float with 24 bit mantissa, 2 bit accuracy, and hard-coded 8 bit exponent_bits
    //  tolerance_bit_shift   =     32 -          (1 +  8 + (24 -                    1         ) - 2             )
    //                              float_length   sign exp  matching_matissa_bits   implicit 1    tolerance_bits
    //
    // Assuming 0 tolerance_bits and solving for matching_matissa_bits yields:
    //  tolerance_bit_shift   =     32 -          (1 +  8 + (matching_matissa_bits - 1         ) - 0             )
    //  tolerance_bit_shift   =     32 -          (1 +  8 + (matching_matissa_bits - 1         )                 )
    //  matching_matissa_bits =     32 -          (1 +  8 + (tolerance_bit_shift   - 1         )                 )
    uint32_t matching_matissa_bits =
        tolerance_bit_shift < 24 ? (32 - (1 + 8 + (tolerance_bit_shift - 1))) : 0;
    return matching_matissa_bits;
}

bool test::all_close_f(const vector<float>& a,
                       const vector<float>& b,
                       int mantissa_bits,
                       int tolerance_bits)
{
    bool rc = true;
    if (a.size() != b.size())
    {
        throw ngraph_error("a.size() != b.size() for all_close_f comparison.");
    }
    vector<uint32_t> distances = float_distances(a, b);

    // e.g. for float with 24 bit mantissa, 2 bit accuracy, and hard-coded 8 bit exponent_bits
    // tolerance_bit_shift = 32 -           (1 +  8 + (24 -     1         ) - 2             )
    //                       float_length    sign exp  mantissa implicit 1    tolerance_bits
    uint32_t tolerance_bit_shift = 32 - (1 + 8 + (mantissa_bits - 1) - tolerance_bits);
    uint32_t tolerance = static_cast<uint32_t>(1U) << tolerance_bit_shift;
    uint32_t max_distance = 0;
    uint32_t min_distance = UINT_MAX;
    size_t max_distance_index = 0;
    size_t min_distance_index = 0;
    size_t diff_count = 0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        if (distances[i] > max_distance)
        {
            max_distance = distances[i];
            max_distance_index = i;
        }
        if (distances[i] < min_distance)
        {
            min_distance = distances[i];
            min_distance_index = i;
        }
        bool is_close_f = distances[i] <= tolerance;
        if (!is_close_f)
        {
            if (diff_count < 5)
            {
                NGRAPH_INFO << a[i] << " is not close to " << b[i] << " at index " << i;
            }

            rc = false;
            diff_count++;
        }
    }
    if (!rc)
    {
        NGRAPH_INFO << "diff count: " << diff_count << " out of " << a.size();
    }
    // Find median value via partial sorting
    size_t middle = distances.size() / 2;
    std::nth_element(distances.begin(), distances.begin() + middle, distances.end());
    uint32_t median_distance = distances[middle];
    if (distances.size() % 2 == 0)
    {
        // Find middle-1 value
        uint64_t median_sum = static_cast<uint64_t>(median_distance) +
                              *max_element(distances.begin(), distances.begin() + middle);
        median_distance = median_sum / 2;
    }

    NGRAPH_INFO << "passing criteria: " << (mantissa_bits - tolerance_bits) << " mantissa bits ("
                << mantissa_bits << " mantissa bits w/ " << tolerance_bits << " tolerance bits)";
    NGRAPH_INFO << "tightest match:   " << matching_mantissa_bits(min_distance)
                << " mantissa bits (" << a[min_distance_index] << " vs " << b[min_distance_index]
                << " at [" << min_distance_index << "])";
    NGRAPH_INFO << "loosest match:    " << matching_mantissa_bits(max_distance)
                << " mantissa bits (" << a[max_distance_index] << " vs " << b[max_distance_index]
                << " at [" << max_distance_index << "])";
    NGRAPH_INFO << "median match:     " << matching_mantissa_bits(median_distance)
                << " mantissa bits";

    return rc;
}

bool test::all_close_f(const std::shared_ptr<runtime::Tensor>& a,
                       const std::shared_ptr<runtime::Tensor>& b,
                       int mantissa_bits,
                       int tolerance_bits)
{
    // Check that the layouts are compatible
    if (*a->get_tensor_layout() != *b->get_tensor_layout())
    {
        throw ngraph_error("Cannot compare tensors with different layouts");
    }
    if (a->get_shape() != b->get_shape())
    {
        return false;
    }

    return test::all_close_f(
        read_float_vector(a), read_float_vector(b), mantissa_bits, tolerance_bits);
}

bool test::all_close_f(const std::vector<std::shared_ptr<runtime::Tensor>>& as,
                       const std::vector<std::shared_ptr<runtime::Tensor>>& bs,
                       int mantissa_bits,
                       int tolerance_bits)
{
    if (as.size() != bs.size())
    {
        return false;
    }
    for (size_t i = 0; i < as.size(); ++i)
    {
        if (!test::all_close_f(as[i], bs[i], mantissa_bits, tolerance_bits))
        {
            return false;
        }
    }
    return true;
}
