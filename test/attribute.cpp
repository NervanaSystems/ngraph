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

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "ngraph/ngraph.hpp"

using namespace std;
using namespace ngraph;

TEST(attribute, bool_type)
{
    Attribute<bool> b{true};
    AttributeBase* b_base = static_cast<AttributeBase*>(&b);

    ASSERT_TRUE(b.get());
    ASSERT_TRUE(b_base->get<bool>());

    b_base->set<bool>(false);
    ASSERT_FALSE(b.get());
    ASSERT_FALSE(b_base->get<bool>());

    b.set(true);
    ASSERT_TRUE(b.get());
    ASSERT_TRUE(b_base->get<bool>());

    ASSERT_ANY_THROW(b_base->get<int>());
    ASSERT_ANY_THROW(b_base->set<int>(42));
}

template <typename T>
void numeric_test()
{
    Attribute<T> x{42};
    AttributeBase* x_base = static_cast<AttributeBase*>(&x);

    ASSERT_EQ(x.get(), static_cast<T>(42));
    ASSERT_EQ(x_base->get<T>(), static_cast<T>(42));

    x_base->set<T>(static_cast<T>(33));
    ASSERT_EQ(x.get(), static_cast<T>(33));
    ASSERT_EQ(x_base->get<T>(), static_cast<T>(33));

    x.set(static_cast<T>(23));
    ASSERT_EQ(x.get(), static_cast<T>(23));
    ASSERT_EQ(x_base->get<T>(), static_cast<T>(23));

    ASSERT_ANY_THROW(x_base->get<bool>());
    ASSERT_ANY_THROW(x_base->set<bool>(false));
}

TEST(attribute, numeric_types)
{
    numeric_test<int8_t>();
    numeric_test<int16_t>();
    numeric_test<int32_t>();
    numeric_test<int64_t>();
    numeric_test<uint8_t>();
    numeric_test<uint16_t>();
    numeric_test<uint32_t>();
    numeric_test<uint64_t>();
    numeric_test<bfloat16>();
    numeric_test<float16>();
    numeric_test<double>();
    numeric_test<float>();
}

TEST(attribute, string_type)
{
    Attribute<string> s{"hello world"};
    AttributeBase* s_base = static_cast<AttributeBase*>(&s);

    ASSERT_EQ(s.get(), "hello world");
    ASSERT_EQ(s_base->get<string>(), "hello world");

    s_base->set<string>("good morning world");
    ASSERT_EQ(s.get(), "good morning world");
    ASSERT_EQ(s_base->get<string>(), "good morning world");

    s.set("good evening world");
    ASSERT_EQ(s.get(), "good evening world");
    ASSERT_EQ(s_base->get<string>(), "good evening world");

    ASSERT_ANY_THROW(s_base->get<int>());
    ASSERT_ANY_THROW(s_base->set<int>(42));
}

TEST(attribute, element_type_type)
{
    Attribute<element::Type> et{element::f32};
    AttributeBase* et_base = static_cast<AttributeBase*>(&et);

    ASSERT_EQ(et.get(), element::f32);
    ASSERT_EQ(et_base->get<element::Type>(), element::f32);

    et_base->set<element::Type>(element::f64);
    ASSERT_EQ(et.get(), element::f64);
    ASSERT_EQ(et_base->get<element::Type>(), element::f64);

    et.set(element::boolean);
    ASSERT_EQ(et.get(), element::boolean);
    ASSERT_EQ(et_base->get<element::Type>(), element::boolean);

    ASSERT_ANY_THROW(et_base->get<int>());
    ASSERT_ANY_THROW(et_base->set<int>(42));
}
