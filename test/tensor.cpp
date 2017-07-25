// ----------------------------------------------------------------------------
// Copyright 2017 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------

#include <vector>
#include <string>
#include <sstream>
#include <memory>

#include "gtest/gtest.h"

using namespace std;

TEST(tensor, test)
{
    class test
    {
    public:
        test(int i)
            : value{i}
        {
        }

        int value;
    };
    map<int, shared_ptr<test>> test_map;

    test_map[1] = make_shared<test>(2);

    EXPECT_NE(nullptr, test_map[1]);
    EXPECT_EQ(nullptr, test_map[2]);
}
