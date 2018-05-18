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

bool test::float_close(float a, float b, int mantissa, int bit_tolerance)
{
    int a_e;
    frexp(a, &a_e);
    float bound = pow(2.f, static_cast<float>(a_e - mantissa + 1 + bit_tolerance));
    float err = abs(a - b);
    return err < bound;
}

bool test::all_close_f(const vector<float>& a,
                       const vector<float>& b,
                       int mantissa,
                       int bit_tolerance)
{
    bool rc = true;
    if (a.size() != b.size())
    {
        throw ngraph_error("a.size() != b.size() for all_close comparison.");
    }
    for (size_t i = 0; i < a.size(); ++i)
    {
    }
    return rc;
}
