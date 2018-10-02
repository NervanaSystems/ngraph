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

#include <sstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "ngraph/uuid.hpp"

using namespace std;
using namespace ngraph;

TEST(uuid, zero)
{
    uuid_type zero = uuid_type::zero();

    stringstream ss;
    ss << zero;
    std::string expected = "00000000-0000-0000-0000-000000000000";

    EXPECT_STREQ(expected.c_str(), ss.str().c_str());
}

TEST(uuid, eq)
{
    uuid_type z1 = uuid_type::zero();
    uuid_type z2 = uuid_type::zero();
    EXPECT_EQ(z1, z2);
}

TEST(uuid, ne)
{
    uuid_type u1;
    uuid_type u2;

    EXPECT_NE(u1, u2);
}
