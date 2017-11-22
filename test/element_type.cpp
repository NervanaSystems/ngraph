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

#include "gtest/gtest.h"

#include "ngraph/types/element_type.hpp"

using namespace ngraph;

TEST(element_type, to_type)
{
    EXPECT_EQ(element::to_type<char>(), element::boolean);
    EXPECT_EQ(element::to_type<bool>(), element::boolean);
    EXPECT_EQ(element::to_type<float>(), element::f32);
    EXPECT_EQ(element::to_type<double>(), element::f64);
    EXPECT_EQ(element::to_type<int8_t>(), element::i8);
    EXPECT_EQ(element::to_type<int16_t>(), element::i16);
    EXPECT_EQ(element::to_type<int32_t>(), element::i32);
    EXPECT_EQ(element::to_type<int64_t>(), element::i64);
    EXPECT_EQ(element::to_type<uint8_t>(), element::u8);
    EXPECT_EQ(element::to_type<uint16_t>(), element::u16);
    EXPECT_EQ(element::to_type<uint32_t>(), element::u32);
    EXPECT_EQ(element::to_type<uint64_t>(), element::u64);
}
