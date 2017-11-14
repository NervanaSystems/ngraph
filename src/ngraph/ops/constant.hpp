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
        /// \brief Abstract base class for constants.
        ///
        /// There are two subclasses: ParameterizedConstant and Constant. ParameterizedConstant allows constant values to be supplied via vectors of the corresponding C++ type;
        /// however, the ParameterizedConstant subclass can only be used when type information is available at C++ compile-time. In cases where types are not known until
        /// C++ runtime, the Constant subclass must be used instead.
        class ConstantBase : public Node
        {
        protected:
            /// \brief Constructs a constant base-type node.
            ///
            /// \param type The TensorViewType for the constant.
            ConstantBase(const std::string& node_type, const std::shared_ptr<TensorViewType>& type)
                : Node(node_type, {})
            {
                set_value_type_checked(type);
            }
        };

        /// \brief Class for constants whose element types are known at C++ compile-time.
        ///
        /// \tparam T The ngraph::element::Type of the tensor's elements.
        ///
        /// This class can be used when the type of the tensor constant is known at C++ compile-time. For other cases, Constant must be used.
        ///
        /// ## Parameters
        ///
        /// |         | Description                                                                          |
        /// | ------- | ------------------------------------------------------------------------------------ |
        /// | `shape` | The ngraph::Shape of the tensor constant.                                            |
        /// | `value` | The ngraph::runtime::ParameterizedTensorView containing data fo the tensor constant. |
        ///
        /// ## Output
        ///
        /// | Type                   | Description                                                           |
        /// | ---------------------- | --------------------------------------------------------------------- |
        /// | \f$E[d_1,\dots,d_n]\f$ | A constant tensor with the specified element type, shape, and values. |
        ///
        /// ## Implementation Status
        ///
        /// | Backend | Status             |
        /// | ------- | ------------------ |
        /// | NGVM    | Fully implemented. |
        template <typename T>
        class ParameterizedConstant : public ConstantBase
        {
        public:
            /// \brief The ngraph element type
            using element_type = T;
            /// \brief The C++ type that holds the element type
            using type = typename T::type;

            /// \brief Constructs a parameterized tensor constant.
            ///
            /// \param shape The shape of the tensor constant.
            /// \param value The value of the tensor constant.
            ParameterizedConstant(
                const Shape& shape,
                const typename std::shared_ptr<ngraph::runtime::ParameterizedTensorView<T>>& value)
                : ConstantBase("ParameterizedConstant",
                               std::make_shared<TensorViewType>(T::element_type(), shape))
                , m_value(value)
            {
            }

            virtual std::shared_ptr<Node> copy_with_new_args(
                const std::vector<std::shared_ptr<Node>>& new_args) const override
            {
                if (new_args.size() != 0)
                    throw ngraph_error("Incorrect number of new arguments");
                return std::make_shared<ParameterizedConstant<T>>(get_shape(), m_value);
            }

            /// \return The value of the tensor constant.
            typename std::shared_ptr<ngraph::runtime::ParameterizedTensorView<T>> get_value() const
            {
                return m_value;
            }

        protected:
            const std::shared_ptr<ngraph::runtime::ParameterizedTensorView<T>> m_value;
        };

        /// \brief A 32-bit floating-point tensor constant.
        using Float32Constant = ParameterizedConstant<element::Float32>;
        /// \brief An 8-bit signed integer tensor constant.
        using Int8Constant = ParameterizedConstant<element::Int8>;
        /// \brief A 32-bit signed integer tensor constant.
        using Int32Constant = ParameterizedConstant<element::Int32>;
        /// \brief A 64-bit signed integer tensor constant.
        using Int64Constant = ParameterizedConstant<element::Int64>;
        /// \brief An 8-bit unsigned integer tensor constant.
        using UInt8Constant = ParameterizedConstant<element::UInt8>;
        /// \brief A 16-bit unsigned integer tensor constant.
        using UInt32Constant = ParameterizedConstant<element::UInt32>;
        /// \brief A 64-bit unsigned integer tensor constant.
        using UInt64Constant = ParameterizedConstant<element::UInt64>;

        /// \brief Class for constants whose element types may not be known until graph construction time.
        ///
        /// This class must be used when the type of the tensor constant is unknown at C++ compile-time. For other cases, ParameterizedConstant should be used.
        ///
        /// ## Parameters
        ///
        /// |                 | Description                                                                                                                                                                    |
        /// | --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
        /// | `et`            | The ngraph::element::Type of the tensor constant.                                                                                                                              |
        /// | `shape`         | The ngraph::Shape of the tensor constant.                                                                                                                                      |
        /// | `value_strings` | A list of strings containing literals for initialization of the tensor constant. These strings are parsed with the appropriate instance of ngraph::element::TraitedType::read. |
        ///
        /// ## Output
        ///
        /// | Type                   | Description                                                           |
        /// | ---------------------- | --------------------------------------------------------------------- |
        /// | \f$E[d_1,\dots,d_n]\f$ | A constant tensor with the specified element type, shape, and values. |
        ///
        /// ## Implementation Status
        ///
        /// | Backend | Status             |
        /// | ------- | ------------------ |
        /// | NGVM    | Fully implemented. |
        class Constant : public ConstantBase
        {
        public:
            /// \brief Constructs a tensor constant.
            ///
            /// \param et The element type of the tensor constant.
            /// \param shape The shape of the tensor constant.
            /// \param value_strings A list of literals for initializing the tensor constant. There must be one literal for each element of the tensor; i.e., `value_strings.size()` must equal `ngraph::shape_size(shape)`.
            Constant(const element::Type& et,
                     const Shape& shape,
                     const std::vector<std::string>& value_strings);

            /// \brief Constructs a tensor constant with the same initialization value copied across the tensor.
            ///
            /// \param et The element type of the tensor constant.
            /// \param shape The shape of the tensor constant.
            /// \param value_string A literal for initializing each tensor constant.
            Constant(const element::Type& et, const Shape& shape, const std::string& value_string);

            virtual std::shared_ptr<Node> copy_with_new_args(
                const std::vector<std::shared_ptr<Node>>& new_args) const override
            {
                if (new_args.size() != 0)
                    throw ngraph_error("Incorrect number of new arguments");
                return std::make_shared<Constant>(get_element_type(), get_shape(), m_value_strings);
            }

            /// \return The initialization literals for the tensor constant.
            const std::vector<std::string>& get_value_strings() const { return m_value_strings; }
        protected:
            void check_args();

            const std::vector<std::string> m_value_strings;
        };
    }
}
