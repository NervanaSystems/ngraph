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

#include <sstream>

#include "ngraph/element_type.hpp"

namespace ngraph
{
    namespace op
    {
        // Defines methods to all constant scalars
        class ScalarConstantBase : public Node
        {
        protected:
            ScalarConstantBase(const std::shared_ptr<TensorViewType>& type)
                : Node({}, type)
            {
            }

            virtual void propagate_types() override;
        };

        // Implement a constant scalar for each element type.
        // The static make method takes a
        template <typename T>
        class ScalarConstant : public ScalarConstantBase
        {
        public:
            // The ngraph element type
            using element_type = T;
            // The C++ type that holds the element type
            using type = typename T::type;

            ScalarConstant(typename T::type value)
                : ScalarConstantBase(std::make_shared<TensorViewType>(T::element_type(), Shape{}))
                , m_value(value)
            {
            }

            virtual std::string description() const override { return "ScalarConstant"; }
            virtual std::string get_node_id() const override
            {
                std::stringstream ss;
                ss << description() << "_" /* << node_id() */;
                return ss.str();
            }

            typename T::type get_value() const { return m_value; }

        protected:
            typename T::type m_value;
        };

        using Float32ScalarConstant = ScalarConstant<element::Float32>;
        using Int8ScalarConstant    = ScalarConstant<element::Int8>;
        using Int32ScalarConstant   = ScalarConstant<element::Int32>;
        using Int64ScalarConstant   = ScalarConstant<element::Int64>;
        using UInt8ScalarConstant   = ScalarConstant<element::UInt8>;
        using UInt32ScalarConstant  = ScalarConstant<element::UInt32>;
        using UInt64ScalarConstant  = ScalarConstant<element::UInt64>;
    }
}
