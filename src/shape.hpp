// ----------------------------------------------------------------------------
// Copyright 2017 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------

#pragma once

#include <cstdio>
#include <iostream>
#include <vector>

#include "common.hpp"

namespace ngraph
{
    /// Number of elements in spanned by a shape
    size_t shape_size(const Shape& shape);

    /// Row-major strides for a shape
    Strides row_major_strides(const Shape& shape);
}
