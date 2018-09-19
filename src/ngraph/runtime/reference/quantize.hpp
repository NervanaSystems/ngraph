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

#pragma once

#include <cmath>
#include <iostream>
#include <vector>
#include <iostream>

#include "ngraph/axis_vector.hpp"
#include "ngraph/coordinate_transform.hpp"
#include "ngraph/runtime/reference/add.hpp"
#include "ngraph/runtime/reference/broadcast.hpp"
#include "ngraph/runtime/reference/divide.hpp"
#include "ngraph/runtime/reference/multiply.hpp"
#include "ngraph/runtime/reference/sum.hpp"
#include "ngraph/util.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            template <typename TI, typename TO>
            void quantize(const TI* input, const TI* scale, const TO* offset, TO* output, const Shape& shape, const AxisSet& axes)
            {
                TI i = input[0];
                TI s = scale[0];
                TO o = offset[0];

                std::cout << "input ptr  = |" << (size_t) input << "|" << std::endl;
                std::cout << "input data = |" << i << "|" << std::endl;

                std::cout << "scale ptr  = |" << (size_t) scale << "|" << std::endl;
                std::cout << "scale data = |" << s << "|" << std::endl;

                std::cout << "offset ptr  = |" << (size_t) offset << "|" << std::endl;
                std::cout << "offset data = |" << (uint32_t) o << "|" << std::endl;

                for(uint32_t i = 0; i < shape_size(shape); ++i) {
                    output[i] = input[0];
                }

            }
        }
    }
}
