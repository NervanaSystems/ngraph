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
#include <map>

#include "ngraph/check.hpp"
#include "ngraph/enum_names.hpp"
#include "ngraph/op/util/attr_types.hpp"

using namespace ngraph;

const op::AutoBroadcastSpec op::AutoBroadcastSpec::NUMPY(AutoBroadcastType::NUMPY, 0);
const op::AutoBroadcastSpec op::AutoBroadcastSpec::NONE{AutoBroadcastType::NONE, 0};

namespace ngraph
{
    template <>
    EnumNames<op::PadMode>& EnumNames<op::PadMode>::get()
    {
        static auto enum_names = EnumNames<op::PadMode>("op::PadMode",
                                                        {{"CONSTANT", op::PadMode::CONSTANT},
                                                         {"EDGE", op::PadMode::EDGE},
                                                         {"REFLECT", op::PadMode::REFLECT},
                                                         {"SYMMETRIC", op::PadMode::SYMMETRIC}});
        return enum_names;
    }

    constexpr DiscreteTypeInfo AttributeAdapter<op::PadMode>::type_info;

    std::ostream& op::operator<<(std::ostream& s, const op::PadMode& type)
    {
        return s << as_string(type);
    }

    template <>
    EnumNames<op::PadType>& EnumNames<op::PadType>::get()
    {
        static auto enum_names = EnumNames<op::PadType>("op::PadType",
                                                        {{"EXPLICIT", op::PadType::EXPLICIT},
                                                         {"SAME_LOWER", op::PadType::SAME_LOWER},
                                                         {"SAME_UPPER", op::PadType::SAME_UPPER},
                                                         {"VALID", op::PadType::VALID}});
        return enum_names;
    }

    constexpr DiscreteTypeInfo AttributeAdapter<op::PadType>::type_info;

    std::ostream& op::operator<<(std::ostream& s, const op::PadType& type)
    {
        return s << as_string(type);
    }

    template <>
    EnumNames<op::RoundingType>& EnumNames<op::RoundingType>::get()
    {
        static auto enum_names = EnumNames<op::RoundingType>(
            "op::RoundingType",
            {{"FLOOR", op::RoundingType::FLOOR}, {"CEIL", op::RoundingType::CEIL}});
        return enum_names;
    }

    constexpr DiscreteTypeInfo AttributeAdapter<op::RoundingType>::type_info;

    std::ostream& op::operator<<(std::ostream& s, const op::RoundingType& type)
    {
        return s << as_string(type);
    }

    template <>
    EnumNames<op::AutoBroadcastType>& EnumNames<op::AutoBroadcastType>::get()
    {
        static auto enum_names =
            EnumNames<op::AutoBroadcastType>("op::AutoBroadcastType",
                                             {{"NONE", op::AutoBroadcastType::NONE},
                                              {"NUMPY", op::AutoBroadcastType::NUMPY},
                                              {"PDPD", op::AutoBroadcastType::PDPD}});
        return enum_names;
    }

    constexpr DiscreteTypeInfo AttributeAdapter<op::AutoBroadcastType>::type_info;

    std::ostream& op::operator<<(std::ostream& s, const op::AutoBroadcastType& type)
    {
        return s << as_string(type);
    }

    template <>
    EnumNames<op::EpsMode>& EnumNames<op::EpsMode>::get()
    {
        static auto enum_names = EnumNames<op::EpsMode>(
            "op::EpsMode", {{"ADD", op::EpsMode::ADD}, {"MAX", op::EpsMode::MAX}});
        return enum_names;
    }

    constexpr DiscreteTypeInfo AttributeAdapter<op::EpsMode>::type_info;

    std::ostream& op::operator<<(std::ostream& s, const op::EpsMode& type)
    {
        return s << as_string(type);
    }

    template <>
    EnumNames<op::TopKSortType>& EnumNames<op::TopKSortType>::get()
    {
        static auto enum_names =
            EnumNames<op::TopKSortType>("op::TopKSortType",
                                        {{"NONE", op::TopKSortType::NONE},
                                         {"SORT_INDICES", op::TopKSortType::SORT_INDICES},
                                         {"SORT_VALUES", op::TopKSortType::SORT_VALUES}});
        return enum_names;
    }

    constexpr DiscreteTypeInfo AttributeAdapter<op::TopKSortType>::type_info;

    std::ostream& op::operator<<(std::ostream& s, const op::TopKSortType& type)
    {
        return s << as_string(type);
    }

    op::AutoBroadcastType op::AutoBroadcastSpec::type_from_string(const std::string& type) const
    {
        static const std::map<std::string, AutoBroadcastType> allowed_values = {
            {"NONE", AutoBroadcastType::NONE},
            {"NUMPY", AutoBroadcastType::NUMPY},
            {"PDPD", AutoBroadcastType::PDPD},
            {"EXPLICIT", AutoBroadcastType::EXPLICIT}};

        NGRAPH_CHECK(allowed_values.count(type) > 0, "Invalid 'type' value passed in.");

        return allowed_values.at(type);
    }

    NGRAPH_API constexpr DiscreteTypeInfo AttributeAdapter<op::AutoBroadcastSpec>::type_info;
}
