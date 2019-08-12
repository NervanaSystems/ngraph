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

        friend class Attribute<bool>;
        friend class Attribute<int8_t>;
        friend class Attribute<int16_t>;
        friend class Attribute<int32_t>;
        friend class Attribute<int64_t>;
        friend class Attribute<uint8_t>;
        friend class Attribute<uint16_t>;
        friend class Attribute<uint32_t>;
        friend class Attribute<uint64_t>;
        friend class Attribute<bfloat16>;
        friend class Attribute<float16>;
        friend class Attribute<float>;
        friend class Attribute<double>;

        friend class Attribute<std::string>;
        friend class Attribute<element::Type>;
        friend class Attribute<PartialShape>;
        friend class Attribute<Dimension>;
        friend class Attribute<Shape>;
        friend class Attribute<Strides>;
        friend class Attribute<Coordinate>;
        friend class Attribute<CoordinateDiff>;
        friend class Attribute<AxisSet>;
        friend class Attribute<AxisVector>;

        friend class Attribute<op::PadMode>;
        friend class Attribute<op::PadType>;
        friend class Attribute<op::AutoBroadcastSpec>;
        friend class Attribute<op::RoundMode>;
        friend class Attribute<op::SortType>;

    private:
        AttributeBase() = default;
        AttributeBase(const AttributeBase&) = delete;
        AttributeBase(AttributeBase&&) = delete;
        AttributeBase& operator=(const AttributeBase&) = delete;
    };

    template <typename T>
    class Attribute : public AttributeBase
    {
    public:
        Attribute() = default;
        Attribute(const T& val)
            : AttributeBase()
            , m_val(val)
        {
        }
        const T& get() const { return m_val; }
        void set(const T& val) { m_val = val; }
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
