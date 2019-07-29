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

TEST(element_type, merge_both_dynamic)
{
    element::Type t;
    ASSERT_TRUE(element::Type::merge(t, element::dynamic, element::dynamic));
    ASSERT_TRUE(t.is_dynamic());
}

TEST(element_type, merge_left_dynamic)
{
    element::Type t;
    ASSERT_TRUE(element::Type::merge(t, element::dynamic, element::u64));
    ASSERT_TRUE(t.is_static());
    ASSERT_EQ(t, element::u64);
}

TEST(element_type, merge_right_dynamic)
{
    element::Type t;
    ASSERT_TRUE(element::Type::merge(t, element::i16, element::dynamic));
    ASSERT_TRUE(t.is_static());
    ASSERT_EQ(t, element::i16);
}

TEST(element_type, merge_both_static_equal)
{
    element::Type t;
    ASSERT_TRUE(element::Type::merge(t, element::f64, element::f64));
    ASSERT_TRUE(t.is_static());
    ASSERT_EQ(t, element::f64);
}

TEST(element_type, merge_both_static_unequal)
{
    element::Type t = element::f32;
    ASSERT_FALSE(element::Type::merge(t, element::i8, element::i16));
    ASSERT_TRUE(t.is_static());
    ASSERT_EQ(t, element::f32);
}

template <typename T>
static std::string get_type_name()
{
    NGRAPH_CHECK(false, "Unhandled element type");
}

template <>
std::string get_type_name<int32_t>()
{
    return "int32_t";
}

template <>
std::string get_type_name<int64_t>()
{
    return "int64_t";
}

TEST(element_type, with_et)
{
    WITH_ET(element::i32, T, { ASSERT_EQ(get_type_name<T>(), "int32_t"); });

    WITH_ET(element::i64, T, { ASSERT_EQ(get_type_name<T>(), "int64_t"); });
    WITH_ET(element::i64, T, ASSERT_EQ(get_type_name<T>(), "int64_t"));

    WITH_ET(element::i32, T, {
        WITH_ET(element::i64, U, {
            WITH_ET(element::i32, V, {
                ASSERT_EQ(get_type_name<T>(), "int32_t");
                ASSERT_EQ(get_type_name<U>(), "int64_t");
                ASSERT_EQ(get_type_name<V>(), "int32_t");
            });
        });
    });
}
