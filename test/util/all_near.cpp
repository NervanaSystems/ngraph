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

#include "util/all_near.hpp"

using namespace std;
using namespace ngraph;

bool test::float_near(float a, float b, uint32_t bit_precision, uint32_t bit_tolerance)
{
    float a_m;
    int a_e;
    float b_m;
    int b_e;
    a_m = frexp(a, &a_e);
    b_m = frexp(b, &b_e);

    float bound = pow(2.f, static_cast<float>(a_e - bit_precision + 1 + bit_tolerance));
    float err = abs(a - b);
    return err < bound;
}
