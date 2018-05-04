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

#pragma once

#include <cstddef>
#include <vector>

#include "ngraph/attribute.hpp"
#include "ngraph/util.hpp"

namespace ngraph
{
    /// \brief Strides for a tensor.
    class Strides : public std::vector<size_t>, public Attribute
    {
    public:
        Strides(const std::initializer_list<size_t>& axis_strides)
            : std::vector<size_t>(axis_strides)
        {
        }

        Strides(const std::vector<size_t>& axis_strides)
            : std::vector<size_t>(axis_strides)
        {
        }

        Strides(const Strides& axis_strides)
            : std::vector<size_t>(axis_strides)
        {
        }

        explicit Strides(size_t n, size_t initial_value = 0)
            : std::vector<size_t>(n, initial_value)
        {
        }

        template <class InputIterator>
        Strides(InputIterator first, InputIterator last)
            : std::vector<size_t>(first, last)
        {
        }

        Strides() {}
        Strides& operator=(const Strides& v)
        {
            static_cast<std::vector<size_t>*>(this)->operator=(v);
            return *this;
        }
        Strides& operator=(Strides&& v)
        {
            static_cast<std::vector<size_t>*>(this)->operator=(v);
            return *this;
        }

        std::string to_string() const { return ("{" + ngraph::join(*this) + "}"); }
        Attribute* clone() const { return new Strides(*this); }
    };
}
