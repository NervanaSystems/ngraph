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

#include <sstream>
#include <bitset>

#include "gtest/gtest.h"

#include "ngraph/ngraph.hpp"

using namespace std;
using namespace ngraph;

union FloatUnion
{
    float f;
    int32_t i;
};

string to_bit_string(float f)
{
    FloatUnion fu{f}; // FloatInt fi = FloatInt(f);
    stringstream ss;
    ss << std::bitset<32>(fu.i);
    return ss.str();
}

uint32_t uint32_with_accuracy_bit(uint32_t accuracy_bit)
{

}

TEST(all_close, example_compare)
{
    float lhs = 1.5;
    float rhs = 1.75;
    NGRAPH_INFO << to_bit_string(lhs);
    NGRAPH_INFO << to_bit_string(rhs);

    FloatUnion lhs_u{lhs};
    FloatUnion rhs_u{rhs};
    // FloatUnion lhs_u_plus
}
