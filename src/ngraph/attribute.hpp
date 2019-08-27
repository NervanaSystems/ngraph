//*****************************************************************************
// Copyright 2017-2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

#include <cstdint>
#include <iostream>
#include <string>

#include "ngraph/axis_set.hpp"
#include "ngraph/axis_vector.hpp"
#include "ngraph/check.hpp"
#include "ngraph/coordinate.hpp"
#include "ngraph/coordinate_diff.hpp"
#include "ngraph/dimension.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/partial_shape.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/strides.hpp"
#include "ngraph/type/bfloat16.hpp"
#include "ngraph/type/element_type.hpp"
#include "ngraph/type/float16.hpp"

namespace ngraph
{
    template <typename T>
    struct IsAttributeType
    {
        static constexpr bool value = false;
    };

#define LEGALIZE_ATTRIBUTE_TYPE(T)                                                                 \
    template <>                                                                                    \
    struct IsAttributeType<T>                                                                      \
    {                                                                                              \
        static constexpr bool value = true;                                                        \
    };

    LEGALIZE_ATTRIBUTE_TYPE(bool);
    LEGALIZE_ATTRIBUTE_TYPE(int8_t);
    LEGALIZE_ATTRIBUTE_TYPE(int16_t);
    LEGALIZE_ATTRIBUTE_TYPE(int32_t);
    LEGALIZE_ATTRIBUTE_TYPE(int64_t);
    LEGALIZE_ATTRIBUTE_TYPE(uint8_t);
    LEGALIZE_ATTRIBUTE_TYPE(uint16_t);
    LEGALIZE_ATTRIBUTE_TYPE(uint32_t);
    LEGALIZE_ATTRIBUTE_TYPE(uint64_t);
    LEGALIZE_ATTRIBUTE_TYPE(bfloat16);
    LEGALIZE_ATTRIBUTE_TYPE(float16);
    LEGALIZE_ATTRIBUTE_TYPE(float);
    LEGALIZE_ATTRIBUTE_TYPE(double);

    LEGALIZE_ATTRIBUTE_TYPE(std::string);
    LEGALIZE_ATTRIBUTE_TYPE(element::Type);
    LEGALIZE_ATTRIBUTE_TYPE(PartialShape);
    LEGALIZE_ATTRIBUTE_TYPE(Dimension);
    LEGALIZE_ATTRIBUTE_TYPE(Shape);
    LEGALIZE_ATTRIBUTE_TYPE(Strides);
    LEGALIZE_ATTRIBUTE_TYPE(Coordinate);
    LEGALIZE_ATTRIBUTE_TYPE(CoordinateDiff);
    LEGALIZE_ATTRIBUTE_TYPE(AxisSet);
    LEGALIZE_ATTRIBUTE_TYPE(AxisVector);

    LEGALIZE_ATTRIBUTE_TYPE(op::PadMode);
    LEGALIZE_ATTRIBUTE_TYPE(op::PadType);
    LEGALIZE_ATTRIBUTE_TYPE(op::EpsMode);
    LEGALIZE_ATTRIBUTE_TYPE(op::AutoBroadcastSpec);
    LEGALIZE_ATTRIBUTE_TYPE(op::RoundMode);
    LEGALIZE_ATTRIBUTE_TYPE(op::SortType);

#undef LEGALIZE_ATTRIBUTE_TYPE

    template <typename T>
    class Attribute;

    class AttributeBase
    {
    public:
        virtual ~AttributeBase() {}
        template <typename T>
        const T& get() const;

        template <typename T>
        void set(const T& val);

        template <typename T>
        bool has_type() const;

        template <typename T>
        const Attribute<T>& as_type() const;

        template <typename T>
        Attribute<T>& as_type();

        virtual std::ostream& write_to_stream(std::ostream& str) const = 0;

    protected:
        AttributeBase() = default;

    private:
        AttributeBase(const AttributeBase&) = delete;
        AttributeBase(AttributeBase&&) = delete;
        AttributeBase& operator=(const AttributeBase&) = delete;
    };

    std::ostream& operator<<(std::ostream& str, const AttributeBase& attr);

    template <typename T>
    class Attribute : public AttributeBase
    {
    public:
        static_assert(
            IsAttributeType<T>::value,
            "Attribute<T> is only valid for types where IsAttributeType<T>::value is true");
        Attribute()
            : AttributeBase()
        {
        }
        Attribute(const T& val)
            : AttributeBase()
            , m_val(val)
        {
        }
        const T& get() const { return m_val; }
        void set(const T& val) { m_val = val; }
        std::ostream& write_to_stream(std::ostream& str) const { return (str << m_val); }
    private:
        T m_val;
    };
}

template <typename T>
const T& ngraph::AttributeBase::get() const
{
    auto as_t = dynamic_cast<const Attribute<T>*>(this);

    NGRAPH_CHECK(as_t, "AttributeBase::get() called with incorrect type");

    return as_t->get();
}

template <typename T>
void ngraph::AttributeBase::set(const T& val)
{
    auto as_t = dynamic_cast<Attribute<T>*>(this);

    NGRAPH_CHECK(as_t, "AttributeBase::set() called with incorrect type");

    as_t->set(val);
}

template <typename T>
bool ngraph::AttributeBase::has_type() const
{
    return dynamic_cast<const Attribute<T>*>(this) != nullptr;
}

template <typename T>
const ngraph::Attribute<T>& ngraph::AttributeBase::as_type() const
{
    auto as_t = dynamic_cast<const Attribute<T>*>(this);

    NGRAPH_CHECK(as_t, "AttributeBase::as_type() called with incorrect type");

    return *as_t;
}

template <typename T>
ngraph::Attribute<T>& ngraph::AttributeBase::as_type()
{
    auto as_t = dynamic_cast<Attribute<T>*>(this);

    NGRAPH_CHECK(as_t, "AttributeBase::as_type() called with incorrect type");

    return *as_t;
}
