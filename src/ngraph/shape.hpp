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

#include <vector>

namespace ngraph
{
    /**
     ** Holds the shape of a tensor view.
     **/
    class Shape
    {
    public:
        /**
         ** \param sizes A sequence of sizes.
         **/
        Shape(const std::initializer_list<size_t>& sizes)
            : m_sizes(sizes)
        {
        }

        /**
         ** Conversion to a vector of sizes.
         **/
        operator const std::vector<size_t>&() const { return m_sizes; }

    protected:
        std::vector<size_t> m_sizes;
    };
}
