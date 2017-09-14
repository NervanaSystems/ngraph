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

#include <algorithm>
#include <vector>

#include "ngraph/shape.hpp"

using namespace std;
using namespace ngraph;

size_t ngraph::shape_size(const Shape& shape)
{
    size_t size = 1;
    for (auto d : shape)
    {
        size *= d;
    }
    return size;
}

Strides ngraph::row_major_strides(const Shape& shape)
{
    Strides strides;
    size_t  s = 1;
    for (auto d = shape.rbegin(); d != shape.rend(); d++)
    {
        strides.push_back(s);
        s *= *d;
    }
    reverse(strides.begin(), strides.end());
    return strides;
}
