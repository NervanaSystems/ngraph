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

#include "ngraph/element_type.hpp"

namespace ngraph
{
    // Defines methods to all constant scalars
    class ScalarConstantBaseOp : public Node
    {
    protected:
        ScalarConstantBaseOp(const std::shared_ptr<TensorViewType>& type)
            : Node({}, type)
        {
        }

        virtual void propagate_types() override;
    };

    // Implement a constant scalar for each element type.
    // The static make method takes a
    template <typename T>
    class ScalarConstantOp : public ScalarConstantBaseOp
    {
    public:
        // The ngraph element type
        using element_type = T;
        // The C++ type that holds the element type
        using ctype = typename T::ctype;

        ScalarConstantOp(typename T::ctype value)
            : ScalarConstantBaseOp(std::make_shared<TensorViewType>(T::type, ngraph::Shape{}))
            , m_value(value)
        {
        }

        virtual std::string description() const override { return "ConstantScalar"; }

        typename T::ctype value() const { return m_value; }

        // Make a constant from any value that can be converted to the C++ type we use
        // to represent the values.
        template <typename U>
        static std::shared_ptr<ScalarConstantOp<T>> make(U value)
        {
            return std::make_shared<ScalarConstantOp<T>>(
                static_cast<ScalarConstantOp<T>::ctype>(value));
        }

    protected:
        typename T::ctype m_value;
    };

    using FloatScalarConstantOp  = ScalarConstantOp<element::Float>;
    using Int8ScalarConstantOp   = ScalarConstantOp<element::Int8>;
    using Int32ScalarConstantOp  = ScalarConstantOp<element::Int32>;
    using Int64ScalarConstantOp  = ScalarConstantOp<element::Int64>;
    using UInt8ScalarConstantOp  = ScalarConstantOp<element::UInt8>;
    using UInt32ScalarConstantOp = ScalarConstantOp<element::UInt32>;
    using UInt64ScalarConstantOp = ScalarConstantOp<element::UInt64>;
}
