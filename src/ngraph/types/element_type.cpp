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
#include <iostream>

#include "ngraph/log.hpp"
#include "ngraph/types/element_type.hpp"

using namespace ngraph;

const element::Type element::boolean(8, false, false, "char");
const element::Type element::f32(32, true, true, "float");
const element::Type element::f64(64, true, true, "double");
const element::Type element::i8(8, false, true, "int8_t");
const element::Type element::i16(16, false, true, "int16_t");
const element::Type element::i32(32, false, true, "int32_t");
const element::Type element::i64(64, false, true, "int64_t");
const element::Type element::u8(8, false, false, "uint8_t");
const element::Type element::u16(16, false, false, "uint16_t");
const element::Type element::u32(32, false, false, "uint32_t");
const element::Type element::u64(64, false, false, "uint64_t");

element::Type::Type(size_t bitwidth, bool is_real, bool is_signed, const std::string& cname)
    : m_bitwidth{bitwidth}
    , m_is_real{is_real}
    , m_is_signed{is_signed}
    , m_cname{cname}
{
}

const std::string& element::Type::c_type_string() const
{
    return m_cname;
}

bool element::Type::operator==(const element::Type& other) const
{
    return m_bitwidth == other.m_bitwidth && m_is_real == other.m_is_real &&
           m_is_signed == other.m_is_signed && m_cname == other.m_cname;
}

bool element::Type::operator<(const Type& other) const
{
    size_t v1 = m_bitwidth << 2;
    v1 |= (m_is_real ? 2 : 0);
    v1 |= (m_is_signed ? 1 : 0);

    size_t v2 = other.m_bitwidth << 2;
    v2 |= (other.m_is_real ? 2 : 0);
    v2 |= (other.m_is_signed ? 1 : 0);

    return v1 < v2;
}

size_t element::Type::size() const
{
    return std::ceil(static_cast<float>(m_bitwidth) / 8.0f);
}

size_t element::Type::hash() const
{
    size_t h1 = std::hash<size_t>{}(m_bitwidth);
    size_t h2 = std::hash<bool>{}(m_is_real);
    size_t h3 = std::hash<bool>{}(m_is_signed);
    return h1 ^ ((h2 ^ (h3 << 1)) << 1);
}

std::ostream& element::operator<<(std::ostream& out, const element::Type& obj)
{
    out << "element::Type(" << obj.m_bitwidth << ", " << obj.m_is_real << ", " << obj.m_is_signed
        << ")";
    return out;
}
