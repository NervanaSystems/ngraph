// Copyright 2017-2020 Intel Corporation
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

#include "ngraph/check.hpp"
#include "ngraph/coordinate_transform.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            template <typename T>
            void strided_slice(const T* arg,
                               T* out,
                               const Shape& arg_shape,
                               const Coordinate& lower_bound,
                               const Coordinate& upper_bound,
                               const Strides& strides,
                               const Shape& out_shape,
                               const std::vector<int64_t>& begin_mask,
                               const std::vector<int64_t>& end_mask,
                               const std::vector<int64_t>& new_axis_mask = std::vector<int64_t>{},
                               const std::vector<int64_t>& shrink_axis_mask = std::vector<int64_t>{},
                               const std::vector<int64_t>& ellipsis_mask = std::vector<int64_t>{})
            {
            }