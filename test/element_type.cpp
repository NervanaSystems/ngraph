/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <map>

#include "gtest/gtest.h"

#include "ngraph/type/element_type.hpp"

using namespace ngraph;

TEST(element_type, from)
{
    EXPECT_EQ(element::from<char>(), element::boolean);
    EXPECT_EQ(element::from<bool>(), element::boolean);
    EXPECT_EQ(element::from<float>(), element::f32);
    EXPECT_EQ(element::from<double>(), element::f64);
    EXPECT_EQ(element::from<int8_t>(), element::i8);
    EXPECT_EQ(element::from<int16_t>(), element::i16);
    EXPECT_EQ(element::from<int32_t>(), element::i32);
    EXPECT_EQ(element::from<int64_t>(), element::i64);
    EXPECT_EQ(element::from<uint8_t>(), element::u8);
    EXPECT_EQ(element::from<uint16_t>(), element::u16);
    EXPECT_EQ(element::from<uint32_t>(), element::u32);
    EXPECT_EQ(element::from<uint64_t>(), element::u64);
}

TEST(element_type, mapable)
{
    std::map<element::Type, std::string> test_map;

    test_map.insert({element::f32, "float"});
}

TEST(element_type, size)
{
    {
        element::Type t1{1, false, false, ""};
        EXPECT_EQ(1, t1.size());
    }
    {
        element::Type t1{2, false, false, ""};
        EXPECT_EQ(1, t1.size());
    }
    {
        element::Type t1{3, false, false, ""};
        EXPECT_EQ(1, t1.size());
    }
    {
        element::Type t1{4, false, false, ""};
        EXPECT_EQ(1, t1.size());
    }
    {
        element::Type t1{5, false, false, ""};
        EXPECT_EQ(1, t1.size());
    }
    {
        element::Type t1{6, false, false, ""};
        EXPECT_EQ(1, t1.size());
    }
    {
        element::Type t1{7, false, false, ""};
        EXPECT_EQ(1, t1.size());
    }
    {
        element::Type t1{8, false, false, ""};
        EXPECT_EQ(1, t1.size());
    }
    {
        element::Type t1{9, false, false, ""};
        EXPECT_EQ(2, t1.size());
    }
}
