//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
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

#include "ngraph/length.hpp"

std::ostream& ngraph::operator<<(std::ostream& str, const Length& length)
{
    if (length.fixed())
    {
        return (str << size_t(length));
    }
    else
    {
        return (str << "?");
    }
}

ngraph::Length ngraph::operator+(const Length& l1, const Length& l2)
{
    return (l1.fixed() && l2.fixed() ? size_t(l1) + size_t(l2) : Length(undet));
}
