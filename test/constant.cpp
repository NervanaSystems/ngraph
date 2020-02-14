//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#include <memory>

#include <gtest/gtest.h>

#include "ngraph/ngraph.hpp"

using namespace ngraph;
using namespace std;

//
// boolean
//

TEST(constant, boolean_string)
{
    Shape shape{4};
    op::Constant c(element::boolean, shape, vector<string>{"1", "0", "1", "0"});
    auto v = c.get_vector<char>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 0);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 0);

    const char* p = c.get_data_ptr<char>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 0);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 0);
}

TEST(constant, boolean_string_broadcast)
{
    Shape shape{4};
    op::Constant c(element::boolean, shape, vector<string>{"1"});
    auto v = c.get_vector<char>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 1);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 1);

    const char* p = c.get_data_ptr<char>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 1);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 1);
}

TEST(constant, boolean_vector)
{
    Shape shape{4};
    op::Constant c(element::boolean, shape, vector<char>{1, 0, 1, 0});
    auto v = c.get_vector<char>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 0);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 0);

    const char* p = c.get_data_ptr<char>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 0);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 0);
}

TEST(constant, boolean_vector_broadcast)
{
    Shape shape{4};
    op::Constant c(element::boolean, shape, vector<char>{1});
    auto v = c.get_vector<char>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 1);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 1);

    const char* p = c.get_data_ptr<char>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 1);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 1);
}

//
// float
//

TEST(constant, float_string)
{
    Shape shape{4};
    op::Constant c(element::f32, shape, vector<string>{"1", "0", "1", "0"});
    auto v = c.get_vector<float>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 0);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 0);

    const float* p = c.get_data_ptr<float>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 0);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 0);
}

TEST(constant, float_string_broadcast)
{
    Shape shape{4};
    op::Constant c(element::f32, shape, vector<string>{"1"});
    auto v = c.get_vector<float>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 1);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 1);

    const float* p = c.get_data_ptr<float>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 1);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 1);
}

TEST(constant, float_vector)
{
    Shape shape{4};
    op::Constant c(element::f32, shape, vector<float>{1, 0, 1, 0});
    auto v = c.get_vector<float>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 0);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 0);

    const float* p = c.get_data_ptr<float>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 0);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 0);
}

TEST(constant, float_vector_broadcast)
{
    Shape shape{4};
    op::Constant c(element::f32, shape, vector<float>{1});
    auto v = c.get_vector<float>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 1);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 1);

    const float* p = c.get_data_ptr<float>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 1);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 1);
}

//
// double
//

TEST(constant, double_string)
{
    Shape shape{4};
    op::Constant c(element::f64, shape, vector<string>{"1", "0", "1", "0"});
    auto v = c.get_vector<double>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 0);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 0);

    const double* p = c.get_data_ptr<double>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 0);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 0);
}

TEST(constant, double_string_broadcast)
{
    Shape shape{4};
    op::Constant c(element::f64, shape, vector<string>{"1"});
    auto v = c.get_vector<double>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 1);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 1);

    const double* p = c.get_data_ptr<double>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 1);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 1);
}

TEST(constant, double_vector)
{
    Shape shape{4};
    op::Constant c(element::f64, shape, vector<double>{1, 0, 1, 0});
    auto v = c.get_vector<double>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 0);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 0);

    const double* p = c.get_data_ptr<double>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 0);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 0);
}

TEST(constant, double_vector_broadcast)
{
    Shape shape{4};
    op::Constant c(element::f64, shape, vector<double>{1});
    auto v = c.get_vector<double>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 1);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 1);

    const double* p = c.get_data_ptr<double>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 1);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 1);
}

//
// int8
//

TEST(constant, int8_string)
{
    Shape shape{4};
    op::Constant c(element::i8, shape, vector<string>{"1", "0", "1", "0"});
    auto v = c.get_vector<int8_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 0);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 0);

    const int8_t* p = c.get_data_ptr<int8_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 0);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 0);
}

TEST(constant, int8_string_broadcast)
{
    Shape shape{4};
    op::Constant c(element::i8, shape, vector<string>{"1"});
    auto v = c.get_vector<int8_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 1);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 1);

    const int8_t* p = c.get_data_ptr<int8_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 1);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 1);
}

TEST(constant, int8_vector)
{
    Shape shape{4};
    op::Constant c(element::i8, shape, vector<int8_t>{1, 0, 1, 0});
    auto v = c.get_vector<int8_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 0);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 0);

    const int8_t* p = c.get_data_ptr<int8_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 0);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 0);
}

TEST(constant, int8_vector_broadcast)
{
    Shape shape{4};
    op::Constant c(element::i8, shape, vector<int8_t>{1});
    auto v = c.get_vector<int8_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 1);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 1);

    const int8_t* p = c.get_data_ptr<int8_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 1);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 1);
}

//
// int16
//

TEST(constant, int16_string)
{
    Shape shape{4};
    op::Constant c(element::i16, shape, vector<string>{"1", "0", "1", "0"});
    auto v = c.get_vector<int16_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 0);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 0);

    const int16_t* p = c.get_data_ptr<int16_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 0);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 0);
}

TEST(constant, int16_string_broadcast)
{
    Shape shape{4};
    op::Constant c(element::i16, shape, vector<string>{"1"});
    auto v = c.get_vector<int16_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 1);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 1);

    const int16_t* p = c.get_data_ptr<int16_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 1);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 1);
}

TEST(constant, int16_vector)
{
    Shape shape{4};
    op::Constant c(element::i16, shape, vector<int16_t>{1, 0, 1, 0});
    auto v = c.get_vector<int16_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 0);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 0);

    const int16_t* p = c.get_data_ptr<int16_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 0);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 0);
}

TEST(constant, int16_vector_broadcast)
{
    Shape shape{4};
    op::Constant c(element::i16, shape, vector<int16_t>{1});
    auto v = c.get_vector<int16_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 1);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 1);

    const int16_t* p = c.get_data_ptr<int16_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 1);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 1);
}

//
// int32
//

TEST(constant, int32_string)
{
    Shape shape{4};
    op::Constant c(element::i32, shape, vector<string>{"1", "0", "1", "0"});
    auto v = c.get_vector<int32_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 0);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 0);

    const int32_t* p = c.get_data_ptr<int32_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 0);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 0);
}

TEST(constant, int32_string_broadcast)
{
    Shape shape{4};
    op::Constant c(element::i32, shape, vector<string>{"1"});
    auto v = c.get_vector<int32_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 1);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 1);

    const int32_t* p = c.get_data_ptr<int32_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 1);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 1);
}

TEST(constant, int32_vector)
{
    Shape shape{4};
    op::Constant c(element::i32, shape, vector<int32_t>{1, 0, 1, 0});
    auto v = c.get_vector<int32_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 0);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 0);

    const int32_t* p = c.get_data_ptr<int32_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 0);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 0);
}

TEST(constant, int32_vector_broadcast)
{
    Shape shape{4};
    op::Constant c(element::i32, shape, vector<int32_t>{1});
    auto v = c.get_vector<int32_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 1);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 1);

    const int32_t* p = c.get_data_ptr<int32_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 1);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 1);
}

//
// int64
//

TEST(constant, int64_string)
{
    Shape shape{4};
    op::Constant c(element::i64, shape, vector<string>{"1", "0", "1", "0"});
    auto v = c.get_vector<int64_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 0);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 0);

    const int64_t* p = c.get_data_ptr<int64_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 0);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 0);
}

TEST(constant, int64_string_broadcast)
{
    Shape shape{4};
    op::Constant c(element::i64, shape, vector<string>{"1"});
    auto v = c.get_vector<int64_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 1);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 1);

    const int64_t* p = c.get_data_ptr<int64_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 1);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 1);
}

TEST(constant, int64_vector)
{
    Shape shape{4};
    op::Constant c(element::i64, shape, vector<int64_t>{1, 0, 1, 0});
    auto v = c.get_vector<int64_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 0);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 0);

    const int64_t* p = c.get_data_ptr<int64_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 0);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 0);
}

TEST(constant, int64_vector_broadcast)
{
    Shape shape{4};
    op::Constant c(element::i64, shape, vector<int64_t>{1});
    auto v = c.get_vector<int64_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 1);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 1);

    const int64_t* p = c.get_data_ptr<int64_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 1);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 1);
}

//
// uint8
//

TEST(constant, uint8_string)
{
    Shape shape{4};
    op::Constant c(element::u8, shape, vector<string>{"1", "0", "1", "0"});
    auto v = c.get_vector<uint8_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 0);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 0);

    const uint8_t* p = c.get_data_ptr<uint8_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 0);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 0);
}

TEST(constant, uint8_string_broadcast)
{
    Shape shape{4};
    op::Constant c(element::u8, shape, vector<string>{"1"});
    auto v = c.get_vector<uint8_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 1);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 1);

    const uint8_t* p = c.get_data_ptr<uint8_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 1);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 1);
}

TEST(constant, uint8_vector)
{
    Shape shape{4};
    op::Constant c(element::u8, shape, vector<uint8_t>{1, 0, 1, 0});
    auto v = c.get_vector<uint8_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 0);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 0);

    const uint8_t* p = c.get_data_ptr<uint8_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 0);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 0);
}

TEST(constant, uint8_vector_broadcast)
{
    Shape shape{4};
    op::Constant c(element::u8, shape, vector<uint8_t>{1});
    auto v = c.get_vector<uint8_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 1);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 1);

    const uint8_t* p = c.get_data_ptr<uint8_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 1);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 1);
}

//
// uint16
//

TEST(constant, uint16_string)
{
    Shape shape{4};
    op::Constant c(element::u16, shape, vector<string>{"1", "0", "1", "0"});
    auto v = c.get_vector<uint16_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 0);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 0);

    const uint16_t* p = c.get_data_ptr<uint16_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 0);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 0);
}

TEST(constant, uint16_string_broadcast)
{
    Shape shape{4};
    op::Constant c(element::u16, shape, vector<string>{"1"});
    auto v = c.get_vector<uint16_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 1);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 1);

    const uint16_t* p = c.get_data_ptr<uint16_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 1);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 1);
}

TEST(constant, uint16_vector)
{
    Shape shape{4};
    op::Constant c(element::u16, shape, vector<uint16_t>{1, 0, 1, 0});
    auto v = c.get_vector<uint16_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 0);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 0);

    const uint16_t* p = c.get_data_ptr<uint16_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 0);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 0);
}

TEST(constant, uint16_vector_broadcast)
{
    Shape shape{4};
    op::Constant c(element::u16, shape, vector<uint16_t>{1});
    auto v = c.get_vector<uint16_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 1);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 1);

    const uint16_t* p = c.get_data_ptr<uint16_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 1);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 1);
}

//
// uint32
//

TEST(constant, uint32_string)
{
    Shape shape{4};
    op::Constant c(element::u32, shape, vector<string>{"1", "0", "1", "0"});
    auto v = c.get_vector<uint32_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 0);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 0);

    const uint32_t* p = c.get_data_ptr<uint32_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 0);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 0);
}

TEST(constant, uint32_string_broadcast)
{
    Shape shape{4};
    op::Constant c(element::u32, shape, vector<string>{"1"});
    auto v = c.get_vector<uint32_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 1);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 1);

    const uint32_t* p = c.get_data_ptr<uint32_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 1);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 1);
}

TEST(constant, uint32_vector)
{
    Shape shape{4};
    op::Constant c(element::u32, shape, vector<uint32_t>{1, 0, 1, 0});
    auto v = c.get_vector<uint32_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 0);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 0);

    const uint32_t* p = c.get_data_ptr<uint32_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 0);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 0);
}

TEST(constant, uint32_vector_broadcast)
{
    Shape shape{4};
    op::Constant c(element::u32, shape, vector<uint32_t>{1});
    auto v = c.get_vector<uint32_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 1);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 1);

    const uint32_t* p = c.get_data_ptr<uint32_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 1);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 1);
}

//
// uint64
//

TEST(constant, uint64_string)
{
    Shape shape{4};
    op::Constant c(element::u64, shape, vector<string>{"1", "0", "1", "0"});
    auto v = c.get_vector<uint64_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 0);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 0);

    const uint64_t* p = c.get_data_ptr<uint64_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 0);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 0);
}

TEST(constant, uint64_string_broadcast)
{
    Shape shape{4};
    op::Constant c(element::u64, shape, vector<string>{"1"});
    auto v = c.get_vector<uint64_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 1);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 1);

    const uint64_t* p = c.get_data_ptr<uint64_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 1);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 1);
}

TEST(constant, uint64_vector)
{
    Shape shape{4};
    op::Constant c(element::u64, shape, vector<uint64_t>{1, 0, 1, 0});
    auto v = c.get_vector<uint64_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 0);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 0);

    const uint64_t* p = c.get_data_ptr<uint64_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 0);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 0);
}

TEST(constant, uint64_vector_broadcast)
{
    Shape shape{4};
    op::Constant c(element::u64, shape, vector<uint64_t>{1});
    auto v = c.get_vector<uint64_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 1);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 1);

    const uint64_t* p = c.get_data_ptr<uint64_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 1);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 1);
}

//
// bfloat16
//

TEST(constant, bfloat16_string)
{
    Shape shape{4};
    op::Constant c(element::bf16, shape, vector<string>{"1", "0", "1", "0"});
    auto v = c.get_vector<bfloat16>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], bfloat16(1));
    EXPECT_EQ(v[1], bfloat16(0));
    EXPECT_EQ(v[2], bfloat16(1));
    EXPECT_EQ(v[3], bfloat16(0));

    const bfloat16* p = c.get_data_ptr<bfloat16>();
    EXPECT_EQ(p[0], bfloat16(1));
    EXPECT_EQ(p[1], bfloat16(0));
    EXPECT_EQ(p[2], bfloat16(1));
    EXPECT_EQ(p[3], bfloat16(0));
}

TEST(constant, bfloat16_string_broadcast)
{
    Shape shape{4};
    op::Constant c(element::bf16, shape, vector<string>{"1"});
    auto v = c.get_vector<bfloat16>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], bfloat16(1));
    EXPECT_EQ(v[1], bfloat16(1));
    EXPECT_EQ(v[2], bfloat16(1));
    EXPECT_EQ(v[3], bfloat16(1));

    const bfloat16* p = c.get_data_ptr<bfloat16>();
    EXPECT_EQ(p[0], bfloat16(1));
    EXPECT_EQ(p[1], bfloat16(1));
    EXPECT_EQ(p[2], bfloat16(1));
    EXPECT_EQ(p[3], bfloat16(1));
}

TEST(constant, bfloat16_vector)
{
    Shape shape{4};
    op::Constant c(element::bf16, shape, vector<bfloat16>{1, 0, 1, 0});
    auto v = c.get_vector<bfloat16>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], bfloat16(1));
    EXPECT_EQ(v[1], bfloat16(0));
    EXPECT_EQ(v[2], bfloat16(1));
    EXPECT_EQ(v[3], bfloat16(0));

    const bfloat16* p = c.get_data_ptr<bfloat16>();
    EXPECT_EQ(p[0], bfloat16(1));
    EXPECT_EQ(p[1], bfloat16(0));
    EXPECT_EQ(p[2], bfloat16(1));
    EXPECT_EQ(p[3], bfloat16(0));
}

TEST(constant, bfloat16_vector_broadcast)
{
    Shape shape{4};
    op::Constant c(element::bf16, shape, vector<bfloat16>{1});
    auto v = c.get_vector<bfloat16>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], bfloat16(1));
    EXPECT_EQ(v[1], bfloat16(1));
    EXPECT_EQ(v[2], bfloat16(1));
    EXPECT_EQ(v[3], bfloat16(1));

    const bfloat16* p = c.get_data_ptr<bfloat16>();
    EXPECT_EQ(p[0], bfloat16(1));
    EXPECT_EQ(p[1], bfloat16(1));
    EXPECT_EQ(p[2], bfloat16(1));
    EXPECT_EQ(p[3], bfloat16(1));
}

//
// float16
//

TEST(constant, float16_string)
{
    Shape shape{4};
    op::Constant c(element::f16, shape, vector<string>{"1", "0", "1", "0"});
    auto v = c.get_vector<float16>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], float16(1));
    EXPECT_EQ(v[1], float16(0));
    EXPECT_EQ(v[2], float16(1));
    EXPECT_EQ(v[3], float16(0));

    const float16* p = c.get_data_ptr<float16>();
    EXPECT_EQ(p[0], float16(1));
    EXPECT_EQ(p[1], float16(0));
    EXPECT_EQ(p[2], float16(1));
    EXPECT_EQ(p[3], float16(0));
}

TEST(constant, float16_string_broadcast)
{
    Shape shape{4};
    op::Constant c(element::f16, shape, vector<string>{"1"});
    auto v = c.get_vector<float16>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], float16(1));
    EXPECT_EQ(v[1], float16(1));
    EXPECT_EQ(v[2], float16(1));
    EXPECT_EQ(v[3], float16(1));

    const float16* p = c.get_data_ptr<float16>();
    EXPECT_EQ(p[0], float16(1));
    EXPECT_EQ(p[1], float16(1));
    EXPECT_EQ(p[2], float16(1));
    EXPECT_EQ(p[3], float16(1));
}

TEST(constant, float16_vector)
{
    Shape shape{4};
    op::Constant c(element::f16, shape, vector<float16>{1, 0, 1, 0});
    auto v = c.get_vector<float16>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], float16(1));
    EXPECT_EQ(v[1], float16(0));
    EXPECT_EQ(v[2], float16(1));
    EXPECT_EQ(v[3], float16(0));

    const float16* p = c.get_data_ptr<float16>();
    EXPECT_EQ(p[0], float16(1));
    EXPECT_EQ(p[1], float16(0));
    EXPECT_EQ(p[2], float16(1));
    EXPECT_EQ(p[3], float16(0));
}

TEST(constant, float16_vector_broadcast)
{
    Shape shape{4};
    op::Constant c(element::f16, shape, vector<float16>{1});
    auto v = c.get_vector<float16>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], float16(1));
    EXPECT_EQ(v[1], float16(1));
    EXPECT_EQ(v[2], float16(1));
    EXPECT_EQ(v[3], float16(1));

    const float16* p = c.get_data_ptr<float16>();
    EXPECT_EQ(p[0], float16(1));
    EXPECT_EQ(p[1], float16(1));
    EXPECT_EQ(p[2], float16(1));
    EXPECT_EQ(p[3], float16(1));
}

TEST(constant, shared_data)
{
    Shape shape{100, 200};
    auto c1 = make_shared<op::Constant>(element::f16, shape, vector<float16>{123});
    auto c2 = static_pointer_cast<op::Constant>(c1->copy_with_new_args({}));
    const float* p1 = c1->get_data_ptr<float>();
    const float* p2 = c2->get_data_ptr<float>();
    EXPECT_EQ(p1, p2);
}
