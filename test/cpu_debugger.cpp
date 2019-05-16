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

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <list>
#include <memory>

#include "gtest/gtest.h"
#include "ngraph/autodiff/adjoints.hpp"
#include "ngraph/file_util.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/runtime/cpu/cpu_backend.hpp"
#include "ngraph/runtime/cpu/cpu_call_frame.hpp"
#include "ngraph/runtime/cpu/cpu_debugger.hpp"
#include "ngraph/runtime/cpu/cpu_layout_descriptor.hpp"
#include "ngraph/runtime/cpu/cpu_tensor_view.hpp"
#include "ngraph/runtime/cpu/op/sigmoid_mul.hpp"
#include "ngraph/util.hpp"
#include "util/test_tools.hpp"

using namespace ngraph;
using namespace std;

bool static is_codegen_mode()
{
    static bool codegen_set = false;
    static bool codegen_mode = false;
    if (!codegen_set)
    {
        const char* ngraph_codegen = std::getenv("NGRAPH_CODEGEN");
        codegen_mode = (ngraph_codegen != nullptr) && std::string(ngraph_codegen) != "0";
        codegen_set = true;
    }
    return codegen_mode;
}

// These tests are for DEX mode only.
TEST(debugger, add_breakpoint)
{
    if (is_codegen_mode())
    {
        //TODO change to skip when there is a new release of gtest
        NGRAPH_WARN << "This test is skipped for CODEGEN mode.";
        return;
    }

    Shape shape{};
    auto A = make_shared<op::Parameter>(element::i32, shape);
    auto B = make_shared<op::Parameter>(element::i32, shape);

    auto add = make_shared<op::Add>(A, B);
    auto absn = make_shared<op::Abs>(add);
    auto neg = make_shared<op::Negative>(absn);

    auto f = make_shared<Function>(neg, ParameterVector{A, B});

    shared_ptr<runtime::Backend> backend = runtime::Backend::create("CPU");

    shared_ptr<runtime::Tensor> a = backend->create_tensor(element::i32, shape);
    shared_ptr<runtime::Tensor> b = backend->create_tensor(element::i32, shape);
    shared_ptr<runtime::Tensor> result = backend->create_tensor(element::i32, shape);

    vector<int> dataA{-1};
    vector<int> dataB{-776};
    copy_data(a, dataA);
    copy_data(b, dataB);

    shared_ptr<runtime::Executable> handle = backend->compile(f);
    auto cf = dynamic_pointer_cast<runtime::cpu::CPU_Executable>(handle)->get_call_frame();

    ngraph::runtime::cpu::CPU_Debugger dbg(*cf);

    dbg.add_breakpoint(neg);
    dbg.call({result}, {a, b});

    ASSERT_EQ(*static_cast<int*>(dbg.inspect(add)), -777);
    ASSERT_EQ(*static_cast<int*>(dbg.inspect(absn)), 777);
    dbg.step();
    ASSERT_EQ(*static_cast<int*>(dbg.inspect(neg)), -777);
}

TEST(debugger, stepping)
{
    if (is_codegen_mode())
    {
        //TODO change to skip when there is a new release of gtest
        NGRAPH_WARN << "This test is skipped for CODEGEN mode.";
        return;
    }

    Shape shape{};
    auto A = make_shared<op::Parameter>(element::i32, shape);
    auto B = make_shared<op::Parameter>(element::i32, shape);

    auto add = make_shared<op::Add>(A, B);
    auto absn = make_shared<op::Abs>(add);
    auto neg = make_shared<op::Negative>(absn);

    auto f = make_shared<Function>(neg, ParameterVector{A, B});

    shared_ptr<runtime::Backend> backend = runtime::Backend::create("CPU");

    shared_ptr<runtime::Tensor> a = backend->create_tensor(element::i32, shape);
    shared_ptr<runtime::Tensor> b = backend->create_tensor(element::i32, shape);
    shared_ptr<runtime::Tensor> result = backend->create_tensor(element::i32, shape);

    vector<int> dataA{-1};
    vector<int> dataB{-776};
    copy_data(a, dataA);
    copy_data(b, dataB);

    shared_ptr<runtime::Executable> handle = backend->compile(f);
    auto cf = dynamic_pointer_cast<runtime::cpu::CPU_Executable>(handle)->get_call_frame();

    ngraph::runtime::cpu::CPU_Debugger dbg(*cf);

    dbg.add_breakpoint(add);
    dbg.call({result}, {a, b});

    ASSERT_EQ(*static_cast<int*>(dbg.inspect(add)), -777);
    dbg.step();
    ASSERT_EQ(*static_cast<int*>(dbg.inspect(absn)), 777);
    dbg.step();
    ASSERT_EQ(*static_cast<int*>(dbg.inspect(neg)), -777);
}

TEST(debugger, delete_breakpoint)
{
    if (is_codegen_mode())
    {
        //TODO change to skip when there is new release of gtest
        NGRAPH_WARN << "This test is skipped for CODEGEN mode.";
        return;
    }

    Shape shape{};
    auto A = make_shared<op::Parameter>(element::i32, shape);
    auto B = make_shared<op::Parameter>(element::i32, shape);

    auto add = make_shared<op::Add>(A, B);
    auto absn = make_shared<op::Abs>(add);
    auto neg = make_shared<op::Negative>(absn);

    auto f = make_shared<Function>(neg, ParameterVector{A, B});

    shared_ptr<runtime::Backend> backend = runtime::Backend::create("CPU");

    shared_ptr<runtime::Tensor> a = backend->create_tensor(element::i32, shape);
    shared_ptr<runtime::Tensor> b = backend->create_tensor(element::i32, shape);
    shared_ptr<runtime::Tensor> result = backend->create_tensor(element::i32, shape);

    vector<int> dataA{-1};
    vector<int> dataB{-776};
    copy_data(a, dataA);
    copy_data(b, dataB);

    shared_ptr<runtime::Executable> handle = backend->compile(f);
    auto cf = dynamic_pointer_cast<runtime::cpu::CPU_Executable>(handle)->get_call_frame();

    ngraph::runtime::cpu::CPU_Debugger dbg(*cf);

    dbg.add_breakpoint(add);
    dbg.add_breakpoint(absn);
    dbg.add_breakpoint(neg);
    dbg.delete_breakpoint(add);
    dbg.delete_breakpoint(absn);
    dbg.delete_breakpoint(neg);
    dbg.call({result}, {a, b});

    ASSERT_EQ(*static_cast<int*>(dbg.inspect(add)), -777);
    ASSERT_EQ(*static_cast<int*>(dbg.inspect(absn)), 777);
    ASSERT_EQ(*static_cast<int*>(dbg.inspect(neg)), -777);
}

TEST(debugger, while_stepping)
{
    if (is_codegen_mode())
    {
        //TODO change to skip when there is new release of gtest
        NGRAPH_WARN << "This test is skipped for CODEGEN mode.";
        return;
    }

    Shape shape{};
    auto A = make_shared<op::Parameter>(element::i32, shape);
    auto B = make_shared<op::Parameter>(element::i32, shape);

    auto add = make_shared<op::Add>(A, B);
    auto absn = make_shared<op::Abs>(add);
    auto neg = make_shared<op::Negative>(absn);

    auto f = make_shared<Function>(neg, ParameterVector{A, B});

    shared_ptr<runtime::Backend> backend = runtime::Backend::create("CPU");

    shared_ptr<runtime::Tensor> a = backend->create_tensor(element::i32, shape);
    shared_ptr<runtime::Tensor> b = backend->create_tensor(element::i32, shape);
    shared_ptr<runtime::Tensor> result = backend->create_tensor(element::i32, shape);

    vector<int> dataA{-1};
    vector<int> dataB{-776};
    copy_data(a, dataA);
    copy_data(b, dataB);

    shared_ptr<runtime::Executable> handle = backend->compile(f);
    auto cf = dynamic_pointer_cast<runtime::cpu::CPU_Executable>(handle)->get_call_frame();

    ngraph::runtime::cpu::CPU_Debugger dbg(*cf);

    dbg.call({result}, {a, b});
    dbg.add_breakpoint(add);
    while (dbg.step())
    {
    };

    ASSERT_EQ(*static_cast<int*>(dbg.inspect(add)), -777);
    ASSERT_EQ(*static_cast<int*>(dbg.inspect(absn)), 777);
    ASSERT_EQ(*static_cast<int*>(dbg.inspect(neg)), -777);
}

TEST(debugger, resume)
{
    if (is_codegen_mode())
    {
        //TODO change to skip when there is new release of gtest
        NGRAPH_WARN << "This test is skipped for CODEGEN mode.";
        return;
    }

    Shape shape{};
    auto A = make_shared<op::Parameter>(element::i32, shape);
    auto B = make_shared<op::Parameter>(element::i32, shape);

    auto add = make_shared<op::Add>(A, B);
    auto absn = make_shared<op::Abs>(add);
    auto neg = make_shared<op::Negative>(absn);

    auto f = make_shared<Function>(neg, ParameterVector{A, B});

    shared_ptr<runtime::Backend> backend = runtime::Backend::create("CPU");

    shared_ptr<runtime::Tensor> a = backend->create_tensor(element::i32, shape);
    shared_ptr<runtime::Tensor> b = backend->create_tensor(element::i32, shape);
    shared_ptr<runtime::Tensor> result = backend->create_tensor(element::i32, shape);

    vector<int> dataA{-1};
    vector<int> dataB{-776};
    copy_data(a, dataA);
    copy_data(b, dataB);

    shared_ptr<runtime::Executable> handle = backend->compile(f);
    auto cf = dynamic_pointer_cast<runtime::cpu::CPU_Executable>(handle)->get_call_frame();

    ngraph::runtime::cpu::CPU_Debugger dbg(*cf);

    dbg.add_breakpoint(absn);
    dbg.call({result}, {a, b});

    ASSERT_EQ(*static_cast<int*>(dbg.inspect(add)), -777);
    dbg.resume();
    ASSERT_EQ(*static_cast<int*>(dbg.inspect(absn)), 777);
    ASSERT_EQ(*static_cast<int*>(dbg.inspect(neg)), -777);
}

TEST(tracer, basic)
{
    Shape shape{};
    auto A = make_shared<op::Parameter>(element::i32, shape);
    auto B = make_shared<op::Parameter>(element::i32, shape);

    auto add = make_shared<op::Add>(A, B);
    auto absn = make_shared<op::Abs>(add);
    auto neg = make_shared<op::Negative>(absn);

    auto f = make_shared<Function>(neg, ParameterVector{A, B});

    shared_ptr<runtime::Backend> backend = runtime::Backend::create("CPU");

    shared_ptr<runtime::Tensor> a = backend->create_tensor(element::i32, shape);
    shared_ptr<runtime::Tensor> b = backend->create_tensor(element::i32, shape);
    shared_ptr<runtime::Tensor> result = backend->create_tensor(element::i32, shape);

    vector<int> dataA{-1};
    vector<int> dataB{-776};
    copy_data(a, dataA);
    copy_data(b, dataB);

    shared_ptr<runtime::Executable> handle = backend->compile(f);
    auto cf = dynamic_pointer_cast<runtime::cpu::CPU_Executable>(handle)->get_call_frame();

    ngraph::runtime::cpu::CPU_Debugger dbg(*cf);

    int good_or_bad_value = -777;
    auto add_tracer = [&good_or_bad_value](void** values, const std::string& name) {
        ASSERT_EQ(static_cast<int*>(values[0])[0], good_or_bad_value);
    };

    dbg.add_tracepoint(add, add_tracer);
    dbg.call({result}, {a, b});
    dbg.delete_tracepoint(add);
    good_or_bad_value = 777;
    dbg.call({result}, {a, b});
}

TEST(tracer, count_tracepoint)
{
    Shape shape{};
    auto A = make_shared<op::Parameter>(element::i32, shape);
    auto B = make_shared<op::Parameter>(element::i32, shape);

    auto add = make_shared<op::Add>(A, B);

    auto f = make_shared<Function>(add, ParameterVector{A, B});

    shared_ptr<runtime::Backend> backend = runtime::Backend::create("CPU");

    shared_ptr<runtime::Tensor> a = backend->create_tensor(element::i32, shape);
    shared_ptr<runtime::Tensor> b = backend->create_tensor(element::i32, shape);
    shared_ptr<runtime::Tensor> result = backend->create_tensor(element::i32, shape);

    shared_ptr<runtime::Executable> handle = backend->compile(f);
    auto cf = dynamic_pointer_cast<runtime::cpu::CPU_Executable>(handle)->get_call_frame();

    ngraph::runtime::cpu::CPU_Debugger dbg(*cf);

    size_t num_iterations = 10;
    size_t offset = 5;

    std::function<void(void**, const std::string&)> callback =
        [&num_iterations, offset](void** values, const std::string& name) {
            ASSERT_EQ(static_cast<int*>(values[0])[0], num_iterations - 1 + offset);
        };

    ngraph::runtime::cpu::CPU_CountTracepoint count_tracepoint(callback, 10);
    for (size_t i = 0; i < num_iterations; i++)
    {
        dbg.add_tracepoint(add, count_tracepoint);
        vector<int> dataA{static_cast<int>(offset)};
        vector<int> dataB{static_cast<int>(i)};
        copy_data(a, dataA);
        copy_data(b, dataB);
        dbg.call({result}, {a, b});
    }
}

TEST(tracer, conditional_tracepoint)
{
    Shape shape{};
    auto A = make_shared<op::Parameter>(element::i32, shape);
    auto B = make_shared<op::Parameter>(element::i32, shape);

    auto add = make_shared<op::Add>(A, B);

    auto f = make_shared<Function>(add, ParameterVector{A, B});

    shared_ptr<runtime::Backend> backend = runtime::Backend::create("CPU");

    shared_ptr<runtime::Tensor> a = backend->create_tensor(element::i32, shape);
    shared_ptr<runtime::Tensor> b = backend->create_tensor(element::i32, shape);
    shared_ptr<runtime::Tensor> result = backend->create_tensor(element::i32, shape);

    shared_ptr<runtime::Executable> handle = backend->compile(f);
    auto cf = dynamic_pointer_cast<runtime::cpu::CPU_Executable>(handle)->get_call_frame();

    ngraph::runtime::cpu::CPU_Debugger dbg(*cf);

    size_t num_iterations = 10;
    size_t offset = 5;
    int countdown = num_iterations;

    auto add_tracer = [&countdown, num_iterations, offset](void** values, const std::string& name) {
        if (countdown-- == 0)
        {
            ASSERT_EQ(static_cast<int*>(values[0])[0], num_iterations - 1 + offset);
        }
    };

    for (size_t i = 0; i < num_iterations; i++)
    {
        dbg.add_tracepoint(add, add_tracer);
        vector<int> dataA{static_cast<int>(offset)};
        vector<int> dataB{static_cast<int>(i)};
        copy_data(a, dataA);
        copy_data(b, dataB);
        dbg.call({result}, {a, b});
    }
}
