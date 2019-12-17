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

#include <cstddef>
#include <ostream>

#include "ngraph/attribute_adapter.hpp"
#include "ngraph/ngraph_visibility.hpp"
#include "ngraph/type.hpp"

namespace ngraph
{
    namespace op
    {
        /// \brief Modes for the `Pad` operator.
        enum class PadMode
        {
            CONSTANT = 0,
            EDGE,
            REFLECT,
            SYMMETRIC
        };

        std::ostream& operator<<(std::ostream& s, const PadMode& type);
    }

    template <>
    class NGRAPH_API AttributeAdapter<op::PadMode> : public EnumAttributeAdapterBase<op::PadMode>
    {
    public:
        AttributeAdapter(op::PadMode& value)
            : EnumAttributeAdapterBase<op::PadMode>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<op::PadMode>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
    };

    namespace op
    {
        /// \brief Padding Type used for `Convolution` and `Pooling`
        ///
        /// Follows ONNX padding type definitions
        /// EXPLICIT   - Pad dimensions are explicity specified
        /// SAME_LOWER - Pad dimensions computed to match input shape
        ///              Ceil(num_dims/2) at the beginning and
        ///              Floor(num_dims/2) at the end
        /// SAME_UPPER - Pad dimensions computed to match input shape
        ///              Floor(num_dims/2) at the beginning and
        ///              Ceil(num_dims/2) at the end
        /// VALID      - No padding
        ///
        enum class PadType
        {
            EXPLICIT = 0,
            SAME_LOWER,
            SAME_UPPER,
            VALID,
            AUTO = SAME_UPPER,
            NOTSET = EXPLICIT,
        };

        std::ostream& operator<<(std::ostream& s, const PadType& type);
    }

    template <>
    class NGRAPH_API AttributeAdapter<op::PadType> : public EnumAttributeAdapterBase<op::PadType>
    {
    public:
        AttributeAdapter(op::PadType& value)
            : EnumAttributeAdapterBase<op::PadType>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<op::PadType>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
    };

    namespace op
    {
        /// \brief Rounding Type used for `Pooling` operators.
        enum class RoundingType
        {
            FLOOR = 0,
            CEIL = 1,
        };

        std::ostream& operator<<(std::ostream& s, const RoundingType& type);
    }

    template <>
    class NGRAPH_API AttributeAdapter<op::RoundingType>
        : public EnumAttributeAdapterBase<op::RoundingType>
    {
    public:
        AttributeAdapter(op::RoundingType& value)
            : EnumAttributeAdapterBase<op::RoundingType>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<op::RoundingType>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
    };

    namespace op
    {
        /// \brief Specifies the algorithm to use for implicit broadcasting of a tensor
        ///        to align with another tensor
        ///
        /// NONE  - No implicit broadcasting of tensor
        /// NUMPY - Numpy-style implicit broadcasting
        ///         (https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
        ///         Right-align dimensions of the two tensors, with missing dimensions
        ///         treated as size 1 dimensions. After alignment, for each dimension,
        ///         their sizes should either match or one of them should be of size 1.
        ///         Size 1 dimension will be implicitly broadcast to match the other
        ///         size.
        ///
        ///         E.g.,
        ///              A: Shape(2, 1, 6)
        ///              B: Shape(   3, 1)
        ///         Result: Shape(2, 3, 6)
        ///
        ///              A: Shape(2, 1, 6)
        ///              B: Shape(   3, 1)
        ///         Result: Shape(2, 3, 6)
        /// PDPD  - PaddlePaddle-style implicit broadcasting
        ///         (https://github.com/PaddlePaddle/Paddle/blob/release/1.5/paddle/
        ///                  fluid/operators/elementwise/elementwise_op.h#L126)
        ///         Broadcast B to match the shape of A, where axis is the start
        ///         dimension index to align B with A. If axis is -1 (default), i
        ///         axis = rank(A) - rank(B). The trailing dimensions of size 1 for B
        ///         will be ignored.
        ///
        ///         E.g.,
        ///              A: Shape(2, 3, 4, 5)
        ///              B: Shape(   3, 4   ) with axis =1
        ///         Result: Shape(2, 3, 4, 5)
        ///
        ///              A: Shape(2, 3, 4, 5)
        ///              B: Shape(   3, 1   ) with axis = 1
        ///         Result: Shape(2, 3, 4, 5)
        ///
        /// TODO: Add more implicit broadcast modes used by frameworks
        enum class AutoBroadcastType
        {
            NONE = 0,
            EXPLICIT = NONE,
            NUMPY,
            PDPD
        };

        std::ostream& operator<<(std::ostream& s, const AutoBroadcastType& type);
    }

    template <>
    class NGRAPH_API AttributeAdapter<op::AutoBroadcastType>
        : public EnumAttributeAdapterBase<op::AutoBroadcastType>
    {
    public:
        AttributeAdapter(op::AutoBroadcastType& value)
            : EnumAttributeAdapterBase<op::AutoBroadcastType>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<op::AutoBroadcastType>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
    };

    namespace op
    {
        /// \brief Specifies how eps is combined with L2 value
        enum class EpsMode
        {
            // Add bias to norm
            ADD,
            // Calculate max of norm and bias
            MAX
        };

        std::ostream& operator<<(std::ostream& s, const EpsMode& type);
    }

    template <>
    class NGRAPH_API AttributeAdapter<op::EpsMode> : public EnumAttributeAdapterBase<op::EpsMode>
    {
    public:
        AttributeAdapter(op::EpsMode& value)
            : EnumAttributeAdapterBase<op::EpsMode>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<op::EpsMode>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
    };

    namespace op
    {
        enum class TopKSortType
        {
            // Returned values are not sorted
            NONE,
            // Sort result based on element indices
            SORT_INDICES,
            // Sort result based on element values
            SORT_VALUES,
        };
        std::ostream& operator<<(std::ostream& s, const TopKSortType& type);
    }

    template <>
    class NGRAPH_API AttributeAdapter<op::TopKSortType>
        : public EnumAttributeAdapterBase<op::TopKSortType>
    {
    public:
        AttributeAdapter(op::TopKSortType& value)
            : EnumAttributeAdapterBase<op::TopKSortType>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<op::TopKSortType>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
    };

    namespace op
    {
        /// \brief Implicit broadcast specification
        struct NGRAPH_API AutoBroadcastSpec
        {
            AutoBroadcastSpec()
                : m_type(AutoBroadcastType::NONE)
                , m_axis(0)
            {
            }
            AutoBroadcastSpec(AutoBroadcastType type)
                : m_type(type)
                , m_axis(0)
            {
            }
            AutoBroadcastSpec(const char* type)
                : AutoBroadcastSpec(type_from_string(type))
            {
            }
            AutoBroadcastSpec(AutoBroadcastType type, int64_t axis)
                : m_type(type)
                , m_axis(axis)
            {
            }

            AutoBroadcastType m_type; // Implicit broadcasting algorithm
            int64_t m_axis;           // Axis to start alignment on

            bool operator==(const AutoBroadcastSpec& a) const
            {
                return a.m_type == m_type && a.m_axis == m_axis;
            }

            static const AutoBroadcastSpec NUMPY;
            static const AutoBroadcastSpec NONE;

        private:
            AutoBroadcastType type_from_string(const std::string& type) const;
        };
    }

    template <>
    class AttributeAdapter<op::AutoBroadcastSpec> : public ValueReference<op::AutoBroadcastSpec>,
                                                    public ValueAccessor<void>
    {
    public:
        AttributeAdapter(op::AutoBroadcastSpec& value)
            : ValueReference<op::AutoBroadcastSpec>(value)
        {
        }
        NGRAPH_API
        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<op::AutoBroadcastSpec>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
    };
}
