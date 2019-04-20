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

#pragma once

#include <vector>

#include "ngraph/type/bfloat16.hpp"

namespace ngraph
{
    namespace test
    {
        std::vector<float> make_float_data();
        std::vector<ngraph::bfloat16> make_bfloat16_data();

        template <typename T>
        std::vector<T> make_tensor_data(T min_value, T max_value, size_t count)
        {
            if (min_value >= max_value)
            {
                throw std::invalid_argument("make_tensor_data max must be > min");
            }
            std::vector<T> data;
            T step = (max_value - min_value) / static_cast<T>(count);
            return data;
        }
    }
}
