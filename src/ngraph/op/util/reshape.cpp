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

#include <numeric>

#include "reshape.hpp"

using namespace ngraph;

AxisVector op::util::get_default_axis_vector(std::size_t data_shape_rank, std::size_t start_value)
{
    AxisVector axes(data_shape_rank);
    std::iota(std::begin(axes), std::end(axes), start_value);
    return axes;
}
