//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
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
#include "ngraph/runtime/cpu/cpu_layout_descriptor.hpp"
#include "ngraph/runtime/cpu/cpu_tensor_view.hpp"
#include "ngraph/runtime/cpu/op/sigmoid_mul.hpp"
#include "ngraph/util.hpp"
#include "util/test_tools.hpp"

using namespace ngraph;
using namespace std;

TEST(debugger, stepping)
{
    Shape shape{};
    auto A = make_shared<op::Parameter>(element::i32, shape);
    auto B = make_shared<op::Parameter>(element::i32, shape);

    auto add = make_shared<op::Add>(A, B);
    auto absn = make_shared<op::Abs>(add);
    auto neg = make_shared<op::Negative>(absn);

    auto f = make_shared<Function>(neg, op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("CPU");

    shared_ptr<runtime::Tensor> a = backend->create_tensor(element::i32, shape);
    shared_ptr<runtime::Tensor> b = backend->create_tensor(element::i32, shape);
    shared_ptr<runtime::Tensor> result = backend->create_tensor(element::i32, shape);

    vector<int> dataA{-1};
    vector<int> dataB{-776};
    copy_data(a, dataA);
    copy_data(b, dataB);

    auto cf =
        std::dynamic_pointer_cast<ngraph::runtime::cpu::CPU_Backend>(backend)->get_call_frame(f);

    cf->step({result}, {a, b});
    ASSERT_EQ(*static_cast<int*>(cf->inspect(add)), -777);
    cf->step({result}, {a, b});
    ASSERT_EQ(*static_cast<int*>(cf->inspect(absn)), 777);
    cf->step({result}, {a, b});
    ASSERT_EQ(*static_cast<int*>(cf->inspect(neg)), -777);
}

TEST(debugger, add_breakpoint)
{
    Shape shape{};
    auto A = make_shared<op::Parameter>(element::i32, shape);
    auto B = make_shared<op::Parameter>(element::i32, shape);

    auto add = make_shared<op::Add>(A, B);
    auto absn = make_shared<op::Abs>(add);
    auto neg = make_shared<op::Negative>(absn);

    auto f = make_shared<Function>(neg, op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("CPU");

    shared_ptr<runtime::Tensor> a = backend->create_tensor(element::i32, shape);
    shared_ptr<runtime::Tensor> b = backend->create_tensor(element::i32, shape);
    shared_ptr<runtime::Tensor> result = backend->create_tensor(element::i32, shape);

    vector<int> dataA{-1};
    vector<int> dataB{-776};
    copy_data(a, dataA);
    copy_data(b, dataB);

    auto cf =
        std::dynamic_pointer_cast<ngraph::runtime::cpu::CPU_Backend>(backend)->get_call_frame(f);

    cf->add_breakpoint(neg);
    cf->call({result}, {a, b});
    ASSERT_EQ(*static_cast<int*>(cf->inspect(add)), -777);
    ASSERT_EQ(*static_cast<int*>(cf->inspect(absn)), 777);
    cf->step({result}, {a, b});
    ASSERT_EQ(*static_cast<int*>(cf->inspect(neg)), -777);
}

TEST(debugger, delete_breakpoint)
{
    Shape shape{};
    auto A = make_shared<op::Parameter>(element::i32, shape);
    auto B = make_shared<op::Parameter>(element::i32, shape);

    auto add = make_shared<op::Add>(A, B);
    auto absn = make_shared<op::Abs>(add);
    auto neg = make_shared<op::Negative>(absn);

    auto f = make_shared<Function>(neg, op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("CPU");

    shared_ptr<runtime::Tensor> a = backend->create_tensor(element::i32, shape);
    shared_ptr<runtime::Tensor> b = backend->create_tensor(element::i32, shape);
    shared_ptr<runtime::Tensor> result = backend->create_tensor(element::i32, shape);

    vector<int> dataA{-1};
    vector<int> dataB{-776};
    copy_data(a, dataA);
    copy_data(b, dataB);

    auto cf =
        std::dynamic_pointer_cast<ngraph::runtime::cpu::CPU_Backend>(backend)->get_call_frame(f);

    cf->add_breakpoint(add);
    cf->add_breakpoint(absn);
    cf->add_breakpoint(neg);
    cf->delete_breakpoint(add);
    cf->delete_breakpoint(absn);
    cf->delete_breakpoint(neg);
    cf->call({result}, {a, b});
    ASSERT_EQ(*static_cast<int*>(cf->inspect(add)), -777);
    ASSERT_EQ(*static_cast<int*>(cf->inspect(absn)), 777);
    ASSERT_EQ(*static_cast<int*>(cf->inspect(neg)), -777);
}

TEST(debugger, while_stepping)
{
    Shape shape{};
    auto A = make_shared<op::Parameter>(element::i32, shape);
    auto B = make_shared<op::Parameter>(element::i32, shape);

    auto add = make_shared<op::Add>(A, B);
    auto absn = make_shared<op::Abs>(add);
    auto neg = make_shared<op::Negative>(absn);

    auto f = make_shared<Function>(neg, op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("CPU");

    shared_ptr<runtime::Tensor> a = backend->create_tensor(element::i32, shape);
    shared_ptr<runtime::Tensor> b = backend->create_tensor(element::i32, shape);
    shared_ptr<runtime::Tensor> result = backend->create_tensor(element::i32, shape);

    vector<int> dataA{-1};
    vector<int> dataB{-776};
    copy_data(a, dataA);
    copy_data(b, dataB);

    auto cf =
        std::dynamic_pointer_cast<ngraph::runtime::cpu::CPU_Backend>(backend)->get_call_frame(f);

    while (cf->step({result}, {a, b}))
    {
    };
    ASSERT_EQ(*static_cast<int*>(cf->inspect(add)), -777);
    ASSERT_EQ(*static_cast<int*>(cf->inspect(absn)), 777);
    ASSERT_EQ(*static_cast<int*>(cf->inspect(neg)), -777);
    cf->call({result}, {a, b});
}

TEST(debugger, kontinue)
{
    Shape shape{};
    auto A = make_shared<op::Parameter>(element::i32, shape);
    auto B = make_shared<op::Parameter>(element::i32, shape);

    auto add = make_shared<op::Add>(A, B);
    auto absn = make_shared<op::Abs>(add);
    auto neg = make_shared<op::Negative>(absn);

    auto f = make_shared<Function>(neg, op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("CPU");

    shared_ptr<runtime::Tensor> a = backend->create_tensor(element::i32, shape);
    shared_ptr<runtime::Tensor> b = backend->create_tensor(element::i32, shape);
    shared_ptr<runtime::Tensor> result = backend->create_tensor(element::i32, shape);

    vector<int> dataA{-1};
    vector<int> dataB{-776};
    copy_data(a, dataA);
    copy_data(b, dataB);

    auto cf =
        std::dynamic_pointer_cast<ngraph::runtime::cpu::CPU_Backend>(backend)->get_call_frame(f);

    cf->add_breakpoint(absn);
    cf->call({result}, {a, b});
    ASSERT_EQ(*static_cast<int*>(cf->inspect(add)), -777);
    cf->kontinue({result}, {a, b});
    ASSERT_EQ(*static_cast<int*>(cf->inspect(absn)), 777);
    ASSERT_EQ(*static_cast<int*>(cf->inspect(neg)), -777);
}
