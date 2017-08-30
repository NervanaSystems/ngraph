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
    // Defines methods to all literal scalars
    class ScalarLiteralBaseOp : public Node
    {
    protected:
        ScalarLiteralBaseOp(const std::shared_ptr<TensorViewType>& type)
            : Node({}, type)
        {
        }

        virtual void propagate_types() override;
    };

    // Implement a literal scalar for each element type.
    // The static make method takes a
    template <typename T>
    class ScalarLiteralOp : public ScalarLiteralBaseOp
    {
    public:
        // The ngraph element type
        using element_type = T;
        // The C++ type that holds the element type
        using ctype = typename T::ctype;

        ScalarLiteralOp(typename T::ctype value)
            : ScalarLiteralBaseOp(std::make_shared<TensorViewType>(T::type, ngraph::Shape{}))
            , m_value(value)
        {
        }

        virtual std::string description() const override { return "LiteralScalar"; }

        typename T::ctype value() const { return m_value; }

        // Make a literal from any value that can be converted to the C++ type we use
        // to represent the values.
        template <typename U>
        static std::shared_ptr<ScalarLiteralOp<T>> make(U value)
        {
            return std::make_shared<ScalarLiteralOp<T>>(
                static_cast<ScalarLiteralOp<T>::ctype>(value));
        }

    protected:
        typename T::ctype m_value;
    };

    using FloatScalarOp  = ScalarLiteralOp<element::Float>;
    using Int8ScalarOp   = ScalarLiteralOp<element::Int8>;
    using Int32ScalarOp  = ScalarLiteralOp<element::Int32>;
    using Int64ScalarOp  = ScalarLiteralOp<element::Int64>;
    using UInt8ScalarOp  = ScalarLiteralOp<element::UInt8>;
    using UInt32ScalarOp = ScalarLiteralOp<element::UInt32>;
    using UInt64ScalarOp = ScalarLiteralOp<element::UInt64>;
}
