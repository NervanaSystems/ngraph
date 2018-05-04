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
#include <set>
#include <sstream>
#include <type_traits>

#include "ngraph/attribute.hpp"

namespace ngraph
{
    /// \brief A boxing class to hold a primitive type (e.g., int, bool, double) as an attribute.
    template <typename T>
    class BoxedAttribute : public Attribute
    {
    public:
        BoxedAttribute(typename std::enable_if<std::is_fundamental<T>::value, T>::type value)
            : m_value(value)
        {
        }

        BoxedAttribute& operator=(T value)
        {
            m_value = value;
            return *this;
        }

        BoxedAttribute& operator=(BoxedAttribute<T>&& v)
        {
            m_value = v.m_valus;
            return *this;
        }

        T unbox() { return m_value; }
        std::string to_string() const
        {
            std::stringstream ss;
            ss << m_value;
            return ss.str();
        }

        Attribute* clone() const { return new BoxedAttribute<T>(m_value); }
    private:
        T m_value;
    };
}
