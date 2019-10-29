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

#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/type.hpp"

using namespace ngraph;

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

    template <>
    const DiscreteTypeInfo EnumAdapter<op::PadMode>::type_info = {"op::PadMode", 0};

    std::ostream& op::operator<<(std::ostream& s, const op::PadMode& type)
    {
        return s << as_type<std::string>(type);
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

    template <>
    const DiscreteTypeInfo EnumAdapter<op::PadType>::type_info = {"op::PadType", 0};

    std::ostream& op::operator<<(std::ostream& s, const op::PadType& type)
    {
        return s << as_type<std::string>(type);
    }

    template <>
    EnumNames<op::RoundingType>& EnumNames<op::RoundingType>::get()
    {
        static auto enum_names = EnumNames<op::RoundingType>(
            "op::RoundingType",
            {{"FLOOR", op::RoundingType::FLOOR}, {"CEIL", op::RoundingType::CEIL}});
        return enum_names;
    }

    template <>
    const DiscreteTypeInfo EnumAdapter<op::RoundingType>::type_info = {"op::RoundingType", 0};

    std::ostream& op::operator<<(std::ostream& s, const op::RoundingType& type)
    {
        return s << as_type<std::string>(type);
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

    template <>
    const DiscreteTypeInfo EnumAdapter<op::AutoBroadcastType>::type_info = {"op::AutoBroadcastType",

                                                                            0};

    std::ostream& op::operator<<(std::ostream& s, const op::AutoBroadcastType& type)
    {
        return s << as_type<std::string>(type);
    }

    template <>
    EnumNames<op::EpsMode>& EnumNames<op::EpsMode>::get()
    {
        static auto enum_names = EnumNames<op::EpsMode>(
            "op::EpsMode", {{"ADD", op::EpsMode::ADD}, {"MAX", op::EpsMode::MAX}});
        return enum_names;
    }

    template <>
    const DiscreteTypeInfo EnumAdapter<op::EpsMode>::type_info = {"op::EpsMode", 0};

    std::ostream& op::operator<<(std::ostream& s, const op::EpsMode& type)
    {
        return s << as_type<std::string>(type);
    }
}
