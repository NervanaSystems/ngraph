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

#include <cassert>
#include <cmath>

#include "element_type.hpp"

const ngraph::ElementType element_type_float    = ngraph::ElementType(32, true, true, "float");
const ngraph::ElementType element_type_int8_t   = ngraph::ElementType(8, false, true, "int8_t");
const ngraph::ElementType element_type_int32_t  = ngraph::ElementType(32, false, true, "int32_t");
const ngraph::ElementType element_type_int64_t  = ngraph::ElementType(64, false, true, "int64_t");
const ngraph::ElementType element_type_uint8_t  = ngraph::ElementType(8, false, false, "int8_t");
const ngraph::ElementType element_type_uint32_t = ngraph::ElementType(32, false, false, "int32_t");
const ngraph::ElementType element_type_uint64_t = ngraph::ElementType(64, false, false, "int64_t");

std::map<std::string, ngraph::ElementType> ngraph::ElementType::m_element_list;

ngraph::ElementType::ElementType(size_t bitwidth, bool is_float, bool is_signed, const std::string& cname)
    : m_bitwidth{bitwidth}
    , m_is_float{is_float}
    , m_is_signed{is_signed}
    , m_cname{cname}
{
    assert(m_bitwidth % 8 == 0);
}

const std::string& ngraph::ElementType::c_type_string() const
{
    return m_cname;
}

bool ngraph::ElementType::operator==(const ElementType& other) const
{
    return m_bitwidth == other.m_bitwidth && m_is_float == other.m_is_float &&
           m_is_signed == other.m_is_signed;
}

size_t ngraph::ElementType::size() const
{
    return std::ceil((float)m_bitwidth / 8.0);
}
