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

#include "ngraph/node.hpp"
#include "ngraph/runtime/utils.hpp"
#include "ngraph/types/element_type.hpp"

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
        class ParameterizedScalarConstant : public ScalarConstantBase
        {
        public:
            // The ngraph element type
            using element_type = T;
            // The C++ type that holds the element type
            using type = typename T::type;

            ParameterizedScalarConstant(typename T::type value)
                : ScalarConstantBase(std::make_shared<TensorViewType>(T::element_type(), Shape{}))
                , m_value(value)
            {
            }

            virtual std::string description() const override
            {
                return "ParameterizedScalarConstant";
            }
            virtual std::string get_node_id() const override
            {
                std::stringstream ss;
                ss << description() << "_" /* << node_id() */;
                return ss.str();
            }

            type get_value() const { return m_value; }
        protected:
            typename T::type m_value;
        };

        using Float32ScalarConstant = ParameterizedScalarConstant<element::Float32>;
        using Int8ScalarConstant = ParameterizedScalarConstant<element::Int8>;
        using Int32ScalarConstant = ParameterizedScalarConstant<element::Int32>;
        using Int64ScalarConstant = ParameterizedScalarConstant<element::Int64>;
        using UInt8ScalarConstant = ParameterizedScalarConstant<element::UInt8>;
        using UInt32ScalarConstant = ParameterizedScalarConstant<element::UInt32>;
        using UInt64ScalarConstant = ParameterizedScalarConstant<element::UInt64>;

        class ScalarConstant : public ScalarConstantBase
        {
        public:
            ScalarConstant(const element::Type& et, const std::string& value_string)
                : ScalarConstantBase(std::make_shared<TensorViewType>(et, Shape{}))
                , m_value_string(value_string)
            {
            }

            virtual std::string description() const override { return "ScalarConstant"; }
            virtual std::string get_node_id() const override
            {
                std::stringstream ss;
                ss << description() << "_" /* << node_id() */;
                return ss.str();
            }

            const std::string& get_value_string() const { return m_value_string; }
        protected:
            const std::string m_value_string;
        };

        // Defines methods to all constant tensors
        class TensorConstantBase : public Node
        {
        protected:
            TensorConstantBase(const std::shared_ptr<TensorViewType>& type)
                : Node({}, type)
            {
            }

            virtual void propagate_types() override;
        };

        // Implement a constant tensor for each element type.
        template <typename T>
        class TensorConstant : public TensorConstantBase
        {
        public:
            // The ngraph element type
            using element_type = T;
            // The C++ type that holds the element type
            using type = typename T::type;

            TensorConstant(const Shape& shape)
                : TensorConstantBase(std::make_shared<TensorViewType>(T::element_type(), shape))
                , m_value(ngraph::runtime::make_tensor<T>(shape))
            {
            }

            virtual std::string description() const override { return "TensorConstant"; }
            virtual std::string get_node_id() const override
            {
                std::stringstream ss;
                ss << description() << "_" /* << node_id() */;
                return ss.str();
            }

            typename std::shared_ptr<ngraph::runtime::ParameterizedTensorView<T>> get_value() const
            {
                return m_value;
            }

        protected:
            std::shared_ptr<ngraph::runtime::ParameterizedTensorView<T>> m_value;
        };

        using Float32TensorConstant = TensorConstant<element::Float32>;
        using Int8TensorConstant = TensorConstant<element::Int8>;
        using Int32TensorConstant = TensorConstant<element::Int32>;
        using Int64TensorConstant = TensorConstant<element::Int64>;
        using UInt8TensorConstant = TensorConstant<element::UInt8>;
        using UInt32TensorConstant = TensorConstant<element::UInt32>;
        using UInt64TensorConstant = TensorConstant<element::UInt64>;
    }
}
