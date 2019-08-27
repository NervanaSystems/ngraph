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
#include "ngraph/check.hpp"

using namespace std;
using namespace ngraph;

std::ostream& op::operator<<(std::ostream& str, const op::PadMode& pad_mode)
{
    switch (pad_mode)
    {
    case op::PadMode::CONSTANT: return (str << "PadMode::CONSTANT");
    case op::PadMode::EDGE: return (str << "PadMode::EDGE");
    case op::PadMode::REFLECT: return (str << "PadMode::REFLECT");
    case op::PadMode::SYMMETRIC: return (str << "PadMode::SYMMETRIC");
    }

    NGRAPH_UNREACHABLE("Unexpected value for enum op::PadMode");
}

std::ostream& op::operator<<(std::ostream& str, const op::PadType& pad_type)
{
    switch (pad_type)
    {
    case op::PadType::EXPLICIT: return (str << "PadType::EXPLICIT");
    case op::PadType::SAME_LOWER: return (str << "PadType::SAME_LOWER");
    case op::PadType::SAME_UPPER: return (str << "PadType::SAME_UPPER");
    case op::PadType::VALID: return (str << "PadType::VALID");
    }

    NGRAPH_UNREACHABLE("Unexpected value for enum op::PadType");
}

std::ostream& op::operator<<(std::ostream& str, const op::AutoBroadcastType& autobroadcast_type)
{
    switch (autobroadcast_type)
    {
    case op::AutoBroadcastType::NONE: return (str << "AutoBroadcastType::NONE");
    case op::AutoBroadcastType::NUMPY: return (str << "AutoBroadcastType::NUMPY");
    }

    NGRAPH_UNREACHABLE("Unexpected value for enum op::AutoBroadcastType");
}

std::ostream& op::operator<<(std::ostream& str, const op::EpsMode& eps_mode)
{
    switch (eps_mode)
    {
    case op::EpsMode::ADD: return (str << "EpsMode::ADD");
    case op::EpsMode::MAX: return (str << "EpsMode::MAX");
    }

    NGRAPH_UNREACHABLE("Unexpected value for enum op::EpsMode");
}

std::ostream& op::operator<<(std::ostream& str, const op::AutoBroadcastSpec& autobroadcast_spec)
{
    // NOTE: m_axis will only be relevant for autobroadcast types that are
    // planned for future implementation.
    switch (autobroadcast_spec.m_type)
    {
    case op::AutoBroadcastType::NONE:
    case op::AutoBroadcastType::NUMPY:
        return (str << "AutoBroadcastSpec(" << autobroadcast_spec.m_type << ")");
    }

    NGRAPH_UNREACHABLE("Unexpected value for enum op::AutoBroadcastType");
}

std::ostream& op::operator<<(std::ostream& str, const op::RoundMode& round_mode)
{
    switch (round_mode)
    {
    case op::RoundMode::ROUND_NEAREST_TOWARD_INFINITY:
        return (str << "RoundMode::ROUND_NEAREST_TOWARD_INFINITY");
    case op::RoundMode::ROUND_NEAREST_TOWARD_ZERO:
        return (str << "RoundMode::ROUND_NEAREST_TOWARD_ZERO");
    case op::RoundMode::ROUND_NEAREST_UPWARD: return (str << "RoundMode::ROUND_NEAREST_UPWARD");
    case op::RoundMode::ROUND_NEAREST_DOWNWARD: return (str << "RoundMode::ROUND_NEAREST_DOWNWARD");
    case op::RoundMode::ROUND_NEAREST_TOWARD_EVEN:
        return (str << "RoundMode::ROUND_NEAREST_TOWARD_EVEN");
    case op::RoundMode::ROUND_TOWARD_INFINITY: return (str << "RoundMode::ROUND_TOWARD_INFINITY");
    case op::RoundMode::ROUND_TOWARD_ZERO: return (str << "RoundMode::ROUND_TOWARD_ZERO");
    case op::RoundMode::ROUND_UP: return (str << "RoundMode::ROUND_UP");
    case op::RoundMode::ROUND_DOWN: return (str << "RoundMode::ROUND_DOWN");
    }

    NGRAPH_UNREACHABLE("Unexpected value for enum op::RoundMode");
}

std::ostream& op::operator<<(std::ostream& str, const op::SortType& sort_type)
{
    switch (sort_type)
    {
    case op::SortType::NONE: return (str << "SortType::NONE");
    case op::SortType::SORT_INDICES: return (str << "SortType::SORT_INDICES");
    case op::SortType::SORT_VALUES: return (str << "SortType::SORT_VALUES");
    }

    NGRAPH_UNREACHABLE("Unexpected value for enum op::SortType");
}
