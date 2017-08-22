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

#include <memory>
#include <vector>

#include "element_type.hpp"
#include "strides.hpp"

namespace ngraph
{
    class ndarray;
}

class ngraph::ndarray
{
public:
    ndarray(ElementType           dtype   = element_type_float,
            std::vector<size_t>   shape   = std::vector<size_t>(),
            std::shared_ptr<char> data    = nullptr,
            size_t                offset  = 0,
            const tensor_stride&  strides = tensor_stride());

    ElementType           dtype;
    std::vector<size_t>   shape;
    std::shared_ptr<char> buffer;
    tensor_stride         strides;
    size_t                offset;
};
