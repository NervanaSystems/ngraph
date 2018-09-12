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

#include "ngraph/rank.hpp"

std::ostream& ngraph::operator<<(std::ostream& str, const Rank& rank)
{
    if (rank.fixed())
    {
        return (str << size_t(rank));
    }
    else
    {
        return (str << "?");
    }
}

bool ngraph::operator==(const Rank& r1, const Rank& r2)
{
    return (r1.fixed() && r2.fixed() && size_t(r1) == size_t(r2));
}

bool ngraph::operator!=(const Rank& r1, const Rank& r2)
{
    return (r1.fixed() && r2.fixed() && size_t(r1) != size_t(r2));
}
