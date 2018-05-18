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

bool test::close_f(float a, float b, int mantissa_bits, int tolerance_bits)
{
    // Returns |a - b| < 2 ^ (a.mantissa - mantissa_bits + 1 + tolerance_bits)
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
    }
    return rc;
}
