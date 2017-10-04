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
#include "ngraph/runtime/parameterized_tensor_view.hpp"
#include "ngraph/types/element_type.hpp"

using namespace ngraph;

ngraph::element::Type::Type(size_t bitwidth,
                            bool is_float,
                            bool is_signed,
                            const std::string& cname)
    : m_bitwidth{bitwidth}
    , m_is_float{is_float}
    , m_is_signed{is_signed}
    , m_cname{cname}
{
    assert(m_bitwidth % 8 == 0);
}

const std::string& ngraph::element::Type::c_type_string() const
{
    return m_cname;
}

bool ngraph::element::Type::operator==(const element::Type& other) const
{
    return m_bitwidth == other.m_bitwidth && m_is_float == other.m_is_float &&
           m_is_signed == other.m_is_signed;
}

size_t ngraph::element::Type::size() const
{
    return std::ceil((float)m_bitwidth / 8.0);
}

std::ostream& ngraph::element::operator<<(std::ostream& out, const ngraph::element::Type& obj)
{
    out << obj.m_cname;
    return out;
}

#if 0
template <typename T>
std::shared_ptr<ngraph::runtime::TensorView>
    ngraph::element::TraitedType<T>::make_primary_tensor_view(const ngraph::Shape& shape) const
{
    return std::make_shared<runtime::ParameterizedTensorView<TraitedType<T>>>(shape);
}

template class ngraph::element::TraitedType<char>;
template class ngraph::element::TraitedType<float>;
template class ngraph::element::TraitedType<int8_t>;
template class ngraph::element::TraitedType<int32_t>;
template class ngraph::element::TraitedType<int64_t>;
template class ngraph::element::TraitedType<uint8_t>;
template class ngraph::element::TraitedType<uint32_t>;
template class ngraph::element::TraitedType<uint64_t>;
#endif
