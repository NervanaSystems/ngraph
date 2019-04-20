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

#include "test_vector_generator.hpp"
#include "ngraph/log.hpp"
#include "util/float_util.hpp"
#include "util/test_tools.hpp"

using namespace ngraph;
using namespace std;

template <>
std::vector<float> test::make_floating_point_data(float min, float max)
{
    std::vector<float> data;
    if (0 >= min && 0 <= max)
    {
        data.push_back(0);
    }
    for (int32_t e = 0; e < 255; e++)
    {
        uint32_t x = e;
        x <<= 23;
        test::FloatUnion u;
        u.i = x | 0x499999;
        // NGRAPH_INFO << test::float_to_bits(u.f) << ", " << u.f;
        if (u.f >= min && u.f <= max)
        {
            data.push_back(u.f);
        }
        if (-u.f >= min && -u.f <= max)
        {
            data.push_back(-u.f);
        }
    }
    sort(data.begin(), data.end());
    return data;
}

template <>
std::vector<bfloat16> test::make_floating_point_data(bfloat16 min, bfloat16 max)
{
    vector<bfloat16> bfloat16_data;
    for (float f : make_floating_point_data<float>(min, max))
    {
        bfloat16_data.emplace_back(bfloat16::from_bits(bfloat16::truncate(f)));
        // NGRAPH_INFO << test::bfloat16_to_bits(bfloat16_data.back()) << ", " << bfloat16_data.back();
    }
    return bfloat16_data;
}
