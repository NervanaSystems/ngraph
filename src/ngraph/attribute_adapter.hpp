//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#include <type_traits>
#include <vector>

#include "ngraph/enum_names.hpp"
#include "ngraph/type.hpp"

namespace ngraph
{
    template <typename Type>
    class ValueAccessor;

    /// ValueAccessor<void> is for values that do not provide
    template <>
    class NGRAPH_API ValueAccessor<void>
    {
    public:
        virtual const DiscreteTypeInfo& get_type_info() const = 0;
        virtual ~ValueAccessor() {}
    };

    /// ValueAccessor<T> represents values that support get/set through T
    template <typename T>
    class ValueAccessor : public ValueAccessor<void>
    {
    public:
        virtual const DiscreteTypeInfo& get_type_info() const = 0;
        /// Returns the value
        virtual const T& get() = 0;
        /// Sets the value
        virtual void set(const T& value) = 0;

    protected:
        T m_buffer;
        bool m_buffer_valid{false};
    };

    template <typename Type>
    class ValueReference
    {
    public:
        operator Type&() const { return m_value; }
    protected:
        ValueReference(Type& value)
            : m_value(value)
        {
        }
        Type& m_value;
    };

    template <typename Type>
    class AttributeAdapter
    {
    };

    template <typename Type>
    class EnumAttributeAdapterBase : public ValueReference<Type>, public ValueAccessor<std::string>
    {
    public:
        EnumAttributeAdapterBase(Type& value)
            : ValueReference<Type>(value)
        {
        }

        const std::string& get() override { return as_string(ValueReference<Type>::m_value); }
        void set(const std::string& value) override
        {
            ValueReference<Type>::m_value = as_enum<Type>(value);
        }
    };

    template <>
    class NGRAPH_API AttributeAdapter<float> : public ValueReference<float>,
                                               public ValueAccessor<double>
    {
    public:
        AttributeAdapter(float& value)
            : ValueReference<float>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<float>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
        const double& get() override;
        void set(const double& value) override;
    };

    template <>
    class NGRAPH_API AttributeAdapter<double> : public ValueReference<double>,
                                                public ValueAccessor<double>
    {
    public:
        AttributeAdapter(double& value)
            : ValueReference<double>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<double>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
        const double& get() override;
        void set(const double& value) override;
    };

    template <>
    class NGRAPH_API AttributeAdapter<int8_t> : public ValueReference<int8_t>,
                                                public ValueAccessor<int64_t>
    {
    public:
        AttributeAdapter(int8_t& value)
            : ValueReference<int8_t>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<int8_t>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
        const int64_t& get() override;
        void set(const int64_t& value) override;
    };

    template <>
    class NGRAPH_API AttributeAdapter<int16_t> : public ValueReference<int16_t>,
                                                 public ValueAccessor<int64_t>
    {
    public:
        AttributeAdapter(int16_t& value)
            : ValueReference<int16_t>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<int16_t>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
        const int64_t& get() override;
        void set(const int64_t& value) override;
    };

    template <>
    class NGRAPH_API AttributeAdapter<int32_t> : public ValueReference<int32_t>,
                                                 public ValueAccessor<int64_t>
    {
    public:
        AttributeAdapter(int32_t& value)
            : ValueReference<int32_t>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<int32_t>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
        const int64_t& get() override;
        void set(const int64_t& value) override;
    };

    template <>
    class NGRAPH_API AttributeAdapter<int64_t> : public ValueReference<int64_t>,
                                                 public ValueAccessor<int64_t>
    {
    public:
        AttributeAdapter(int64_t& value)
            : ValueReference<int64_t>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<int64_t>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
        const int64_t& get() override;
        void set(const int64_t& value) override;
    };

    template <>
    class NGRAPH_API AttributeAdapter<uint8_t> : public ValueReference<uint8_t>,
                                                 public ValueAccessor<int64_t>
    {
    public:
        AttributeAdapter(uint8_t& value)
            : ValueReference<uint8_t>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<uint8_t>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
        const int64_t& get() override;
        void set(const int64_t& value) override;
    };

    template <>
    class NGRAPH_API AttributeAdapter<uint16_t> : public ValueReference<uint16_t>,
                                                  public ValueAccessor<int64_t>
    {
    public:
        AttributeAdapter(uint16_t& value)
            : ValueReference<uint16_t>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<uint16_t>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
        const int64_t& get() override;
        void set(const int64_t& value) override;
    };

    template <>
    class NGRAPH_API AttributeAdapter<uint32_t> : public ValueReference<uint32_t>,
                                                  public ValueAccessor<int64_t>
    {
    public:
        AttributeAdapter(uint32_t& value)
            : ValueReference<uint32_t>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<uint32_t>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
        const int64_t& get() override;
        void set(const int64_t& value) override;
    };

    template <>
    class NGRAPH_API AttributeAdapter<uint64_t> : public ValueReference<uint64_t>,
                                                  public ValueAccessor<int64_t>
    {
    public:
        AttributeAdapter(uint64_t& value)
            : ValueReference<uint64_t>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<uint64_t>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
        const int64_t& get() override;
        void set(const int64_t& value) override;
    };

    /// Note: These class bodies cannot be defined with templates because of interactions
    /// between dllexport and templates on Windows.
    template <>
    class NGRAPH_API AttributeAdapter<std::vector<int64_t>>
        : public ValueReference<std::vector<int64_t>>, public ValueAccessor<std::vector<int64_t>>
    {
    public:
        AttributeAdapter(std::vector<int64_t>& value)
            : ValueReference<std::vector<int64_t>>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<vector<int64_t>>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
        const std::vector<int64_t>& get() override;
        void set(const std::vector<int64_t>& value) override;
    };

    template <>
    class NGRAPH_API AttributeAdapter<std::vector<uint64_t>>
        : public ValueReference<std::vector<uint64_t>>, public ValueAccessor<std::vector<int64_t>>
    {
    public:
        AttributeAdapter(std::vector<uint64_t>& value)
            : ValueReference<std::vector<uint64_t>>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<vector<uint64_t>>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
        const std::vector<int64_t>& get() override;
        void set(const std::vector<int64_t>& value) override;
    };

    template <typename A, typename B>
    A copy_from(B& b)
    {
        A result(b.size());
        for (int i = 0; i < b.size(); ++i)
        {
            result[i] =
                static_cast<typename std::remove_reference<decltype(result[i])>::type>(b[i]);
        }
        return result;
    }
}
