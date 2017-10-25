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

#include <sstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "ngraph/codegen/compiler.hpp"

using namespace std;

TEST(codegen, simple_return)
{
    constexpr auto name = "test.cpp";
    constexpr auto source = R"(extern "C" int test() { return 2+5; })";

    nervana::cpu::execution_state estate;
    auto module = estate.compile(source, name);
    ASSERT_NE(nullptr, module);

    estate.add_module(module);

    estate.finalize();

    auto func = estate.find_function<int()>("test");
    ASSERT_NE(nullptr, func);

    int result = func();
    EXPECT_EQ(7, result);
}

TEST(codegen, pass_args)
{
    constexpr auto name = "test.cpp";
    constexpr auto source = R"(extern "C" int test(int a, int b) { return a+b; })";

    nervana::cpu::execution_state estate;
    auto module = estate.compile(source, name);
    ASSERT_NE(nullptr, module);

    estate.add_module(module);

    estate.finalize();

    auto func = estate.find_function<int(int, int)>("test");
    ASSERT_NE(nullptr, func);

    int result = func(20, 22);
    EXPECT_EQ(42, result);
}

TEST(codegen, include)
{
    constexpr auto name = "test.cpp";
    constexpr auto source =
        R"(
        #include <cmath>
        extern "C" int test(int a, int b)
        {
            return (int)pow((double)a,(double)b);
        }
    )";

    nervana::cpu::execution_state estate;
    auto module = estate.compile(source, name);
    ASSERT_NE(nullptr, module);

    estate.add_module(module);

    estate.finalize();

    auto func = estate.find_function<int(int, int)>("test");
    ASSERT_NE(nullptr, func);

    int result = func(20, 2);
    EXPECT_EQ(400, result);
}
