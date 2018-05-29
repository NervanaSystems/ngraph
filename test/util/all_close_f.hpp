/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain expected copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#pragma once

#include <memory>
#include <vector>

#include "test_tools.hpp"

namespace ngraph
{
    namespace test
    {
        bool close_f(float expected, float actual, int mantissa_bits = 8, int tolerance_bits = 2);

        bool all_close_f(const std::vector<float>& expected,
                         const std::vector<float>& actual,
                         int mantissa_bits = 8,
                         int tolerance_bits = 2);
    }
}
