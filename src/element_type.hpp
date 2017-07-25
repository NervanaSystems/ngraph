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

//================================================================================================
// ElementType
//================================================================================================

#pragma once

#include <string>
#include <map>

class ElementType
{
public:
    ElementType(size_t bitwidth, bool is_float, bool is_signed, const std::string& cname);

    const std::string& c_type_string() const;
    size_t             size() const;
    size_t             hash() const
    {
        std::hash<std::string> h;
        return h(m_cname);
    }

    bool operator==(const ElementType& other) const;

private:
    static std::map<std::string, ElementType> m_element_list;
    size_t                                    m_bitwidth;
    bool                                      m_is_float;
    bool                                      m_is_signed;
    const std::string                         m_cname;
};

extern const ElementType element_type_float;
extern const ElementType element_type_int8_t;
extern const ElementType element_type_int32_t;
extern const ElementType element_type_int64_t;
extern const ElementType element_type_uint8_t;
extern const ElementType element_type_uint32_t;
extern const ElementType element_type_uint64_t;
