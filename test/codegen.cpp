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
#include "ngraph/codegen/execution_engine.hpp"

using namespace std;
using namespace ngraph;

TEST(DISABLED_codegen, simple_return)
{
    constexpr auto source = R"(extern "C" int test() { return 2+5; })";

    codegen::Compiler compiler;
    codegen::ExecutionEngine execution_engine;

    auto module = compiler.compile(source);
    ASSERT_NE(nullptr, module);

    execution_engine.add_module(module);

    execution_engine.finalize();

    auto func = execution_engine.find_function<int()>("test");
    ASSERT_NE(nullptr, func);

    int result = func();
    EXPECT_EQ(7, result);
}

TEST(DISABLED_codegen, pass_args)
{
    constexpr auto source = R"(extern "C" int test(int a, int b) { return a+b; })";

    codegen::Compiler compiler;
    codegen::ExecutionEngine execution_engine;

    auto module = compiler.compile(source);
    ASSERT_NE(nullptr, module);

    execution_engine.add_module(module);

    execution_engine.finalize();

    auto func = execution_engine.find_function<int(int, int)>("test");
    ASSERT_NE(nullptr, func);

    int result = func(20, 22);
    EXPECT_EQ(42, result);
}

TEST(DISABLED_codegen, include)
{
    constexpr auto source =
        R"(
        #include <cmath>
        extern "C" int test(int a, int b)
        {
            return (int)pow((double)a,(double)b);
        }
    )";

    codegen::Compiler compiler;
    codegen::ExecutionEngine execution_engine;

    auto module = compiler.compile(source);
    ASSERT_NE(nullptr, module);

    execution_engine.add_module(module);

    execution_engine.finalize();

    auto func = execution_engine.find_function<int(int, int)>("test");
    ASSERT_NE(nullptr, func);

    int result = func(20, 2);
    EXPECT_EQ(400, result);
}
