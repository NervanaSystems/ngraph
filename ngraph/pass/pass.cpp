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

#include "ngraph/pass/pass.hpp"
#include "ngraph/pass/manager.hpp"

using namespace std;
using namespace ngraph;

pass::PassBase::PassBase()
    : m_property{all_pass_property_off}
{
}

pass::ManagerState& pass::PassBase::get_state()
{
    return *m_state;
}

void pass::PassBase::set_state(ManagerState& state)
{
    m_state = &state;
}

bool pass::PassBase::get_property(const PassPropertyMask& prop) const
{
    return m_property.is_set(prop);
}

void pass::PassBase::set_property(const PassPropertyMask& prop, bool value)
{
    if (value)
    {
        m_property.set(prop);
    }
    else
    {
        m_property.clear(prop);
    }
}
