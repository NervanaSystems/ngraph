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

#include <memory>

#include "gtest/gtest.h"

#include "ngraph/ngraph.hpp"

using namespace std;
using namespace ngraph;

TEST(tensor_map, create_default_ctor)
{
    TensorMap tm;

    ASSERT_EQ(tm.shape(), (Shape{0}));
    ASSERT_EQ(tm.element_type(), element::f32);
    ASSERT_EQ(tm.raw_buffer(), nullptr);
    ASSERT_EQ(tm.buffer<float>(), nullptr);
    ASSERT_ANY_THROW(tm.buffer<int32_t>());
}

TEST(tensor_map, create_with_et)
{
    std::vector<int32_t> values_vec{1, 2, 3, 4};
    TensorMap tm(element::i32, Shape{2, 2}, values_vec.data());
    ASSERT_EQ(tm.shape(), (Shape{2, 2}));
    ASSERT_EQ(tm.element_type(), element::i32);
    ASSERT_EQ(tm.raw_buffer(), values_vec.data());
    ASSERT_EQ(tm.buffer<int32_t>(), values_vec.data());
    ASSERT_ANY_THROW(tm.buffer<float>());
}

TEST(tensor_map, create_from_cpp_type)
{
    std::vector<int32_t> values_vec{1, 2, 3, 4};
    TensorMap tm(Shape{2, 2}, values_vec.data());
    ASSERT_EQ(tm.shape(), (Shape{2, 2}));
    ASSERT_EQ(tm.element_type(), element::i32);
    ASSERT_EQ(tm.raw_buffer(), values_vec.data());
    ASSERT_EQ(tm.buffer<int32_t>(), values_vec.data());
    ASSERT_ANY_THROW(tm.buffer<float>());
}

TEST(tensor_value, create_default_ctor)
{
    TensorValue tv;

    ASSERT_EQ(tv.shape(), (Shape{0}));
    ASSERT_EQ(tv.element_type(), element::f32);
    ASSERT_EQ(tv.raw_buffer(), tv.buffer<float>());
    ASSERT_EQ(tv.raw_buffer(), tv.buffer_unsafe<float>());
    ASSERT_ANY_THROW(tv.buffer<int32_t>());
    ASSERT_EQ(tv.raw_buffer(), tv.buffer_unsafe<int32_t>());
}

TEST(tensor_value, create_with_et)
{
    std::vector<int32_t> values_vec{1, 2, 3, 4};
    TensorValue tv(element::i32, Shape{2, 2}, values_vec.data());
    ASSERT_EQ(tv.shape(), (Shape{2, 2}));
    ASSERT_EQ(tv.element_type(), element::i32);
    ASSERT_NE(tv.raw_buffer(), values_vec.data());
    ASSERT_NE(tv.buffer<int32_t>(), values_vec.data());
    ASSERT_EQ(tv.buffer<int32_t>()[0], values_vec[0]);
    ASSERT_EQ(tv.buffer<int32_t>()[1], values_vec[1]);
    ASSERT_EQ(tv.buffer<int32_t>()[2], values_vec[2]);
    ASSERT_EQ(tv.buffer<int32_t>()[3], values_vec[3]);
    ASSERT_ANY_THROW(tv.buffer<float>());
    ASSERT_EQ(tv.buffer_unsafe<float>(), tv.raw_buffer());
}

TEST(tensor_value, create_from_cpp_type)
{
    std::vector<int32_t> values_vec{1, 2, 3, 4};
    TensorValue tv(Shape{2, 2}, values_vec.data());
    ASSERT_EQ(tv.shape(), (Shape{2, 2}));
    ASSERT_EQ(tv.element_type(), element::i32);
    ASSERT_NE(tv.raw_buffer(), values_vec.data());
    ASSERT_NE(tv.buffer<int32_t>(), values_vec.data());
    ASSERT_EQ(tv.buffer<int32_t>()[0], values_vec[0]);
    ASSERT_EQ(tv.buffer<int32_t>()[1], values_vec[1]);
    ASSERT_EQ(tv.buffer<int32_t>()[2], values_vec[2]);
    ASSERT_EQ(tv.buffer<int32_t>()[3], values_vec[3]);
    ASSERT_ANY_THROW(tv.buffer<float>());
    ASSERT_EQ(tv.buffer_unsafe<float>(), tv.raw_buffer());
}
