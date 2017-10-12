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
        // Defines methods to all constants
        class ConstantBase : public Node
        {
        protected:
            ConstantBase(const std::shared_ptr<TensorViewType>& type)
                : Node({}, type)
            {
            }

            virtual void propagate_types() override;
        };

        // Implement a constant tensor for each element type.
        template <typename T>
        class ParameterizedConstant : public ConstantBase
        {
        public:
            // The ngraph element type
            using element_type = T;
            // The C++ type that holds the element type
            using type = typename T::type;

            ParameterizedConstant(
                const Shape& shape,
                typename std::shared_ptr<ngraph::runtime::ParameterizedTensorView<T>>& value)
                : ConstantBase(std::make_shared<TensorViewType>(T::element_type(), shape))
                , m_value(value)
            {
            }

            virtual std::string description() const override { return "ParameterizedConstant"; }
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

        using Float32Constant = ParameterizedConstant<element::Float32>;
        using Int8Constant = ParameterizedConstant<element::Int8>;
        using Int32Constant = ParameterizedConstant<element::Int32>;
        using Int64Constant = ParameterizedConstant<element::Int64>;
        using UInt8Constant = ParameterizedConstant<element::UInt8>;
        using UInt32Constant = ParameterizedConstant<element::UInt32>;
        using UInt64Constant = ParameterizedConstant<element::UInt64>;

        class Constant : public ConstantBase
        {
        public:
            Constant(const element::Type& et,
                     const Shape& shape,
                     const std::vector<std::string>& value_strings)
                : ConstantBase(std::make_shared<TensorViewType>(et, shape))
                , m_value_strings(value_strings)
            {
            }

            Constant(const element::Type& et, const Shape& shape, const std::string& value_string)
                : ConstantBase(std::make_shared<TensorViewType>(et, shape))
                , m_value_strings(ngraph::shape_size(shape), value_string)
            {
            }

            virtual std::string description() const override { return "Constant"; }
            virtual std::string get_node_id() const override
            {
                std::stringstream ss;
                ss << description() << "_" /* << node_id() */;
                return ss.str();
            }

            const std::vector<std::string>& get_value_strings() const { return m_value_strings; }
        protected:
            const std::vector<std::string> m_value_strings;
        };
    }
}
