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

using namespace ngraph;

const op::AutoBroadcastSpec op::AutoBroadcastSpec::NUMPY(AutoBroadcastType::NUMPY, 0);
const op::AutoBroadcastSpec op::AutoBroadcastSpec::NONE{AutoBroadcastType::NONE, 0};

std::ostream& op::operator<<(std::ostream& s, const op::AutoBroadcastType& type)
{
    switch (type)
    {
    case op::AutoBroadcastType::NONE: s << "NONE"; break;
    case op::AutoBroadcastType::NUMPY: s << "NUMPY"; break;
    case op::AutoBroadcastType::PDPD: s << "PDPD"; break;
    default: s << "Undefined Type";
    }
    return s;
}
