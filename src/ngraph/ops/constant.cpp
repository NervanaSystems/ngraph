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

#include "ngraph/ops/constant.hpp"
#include "ngraph/util.hpp"

using namespace ngraph;

namespace
{
    template <typename ET>
    void check_value_strings(const std::vector<std::string>& value_strings)
    {
        auto result = parse_string<typename ET::type>(value_strings);
    }
}

//
// Utility macro for dispatching an element type-templated function at runtime.
//

// clang-format off
// Sorry, but you really don't want to see what clang-format does to this thing. :)
#define FUNCTION_ON_ELEMENT_TYPE(et, err_msg, f, ...)                                     \
    (                                                                                     \
        ((et) == element::Bool::element_type()) ? (f<element::Bool>(__VA_ARGS__)) :       \
        ((et) == element::Float32::element_type()) ? (f<element::Float32>(__VA_ARGS__)) : \
        ((et) == element::Int8::element_type()) ? (f<element::Int8>(__VA_ARGS__)) :       \
        ((et) == element::Int16::element_type()) ? (f<element::Int16>(__VA_ARGS__)) :     \
        ((et) == element::Int32::element_type()) ? (f<element::Int32>(__VA_ARGS__)) :     \
        ((et) == element::Int64::element_type()) ? (f<element::Int64>(__VA_ARGS__)) :     \
        ((et) == element::UInt8::element_type()) ? (f<element::UInt8>(__VA_ARGS__)) :     \
        ((et) == element::UInt16::element_type()) ? (f<element::UInt16>(__VA_ARGS__)) :   \
        ((et) == element::UInt32::element_type()) ? (f<element::UInt32>(__VA_ARGS__)) :   \
        ((et) == element::UInt64::element_type()) ? (f<element::UInt64>(__VA_ARGS__)) :   \
        (throw ngraph_error(err_msg))                                                     \
    )
// clang-format on

op::Constant::Constant(const element::Type& et,
                       const Shape& shape,
                       const std::vector<std::string>& value_strings)
    : ConstantBase("Constant", std::make_shared<TensorViewType>(et, shape))
    , m_value_strings(value_strings)
{
    check_args();
}

/// \brief Constructs a tensor constant with the same initialization value copied across the tensor.
///
/// \param et The element type of the tensor constant.
/// \param shape The shape of the tensor constant.
/// \param value_string A literal for initializing each tensor constant.
op::Constant::Constant(const element::Type& et, const Shape& shape, const std::string& value_string)
    : ConstantBase("Constant", std::make_shared<TensorViewType>(et, shape))
    , m_value_strings(ngraph::shape_size(shape), value_string)
{
    check_args();
}

void op::Constant::check_args()
{
    // We check the number of value strings and
    // also call check_value_strings just to make sure the result will be parseable at compile
    // time. (It will throw an exception if not.)
    auto tvt = std::dynamic_pointer_cast<const TensorViewType>(m_value_type);
    if (nullptr == tvt)
    {
        throw ngraph_error("Constant does not have tensor view type");
    }
    auto shape = tvt->get_shape();

    if (ngraph::shape_size(shape) != m_value_strings.size())
    {
        throw ngraph_error("Constant does not have the expected number of literals");
    }

    auto& et = tvt->get_element_type();

    FUNCTION_ON_ELEMENT_TYPE(
        et, "Constant has unhandled element type", check_value_strings, m_value_strings);
}
