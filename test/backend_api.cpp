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

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/runtime/backend.hpp"
#include "ngraph/util.hpp"
#include "util/all_close_f.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

TEST(backend_api, registered_devices)
{
    vector<string> devices = runtime::Backend::get_registered_devices();
    EXPECT_GE(devices.size(), 0);

    EXPECT_TRUE(find(devices.begin(), devices.end(), "INTERPRETER") != devices.end());
}

TEST(backend_api, invalid_name)
{
    ASSERT_ANY_THROW(ngraph::runtime::Backend::create("COMPLETELY-BOGUS-NAME"));
}

#ifndef NGRAPH_JSON_DISABLE
TEST(backend_api, save_load)
{
    Shape shape{2, 2};
    auto A = make_shared<op::v0::Parameter>(element::f32, shape);
    auto B = make_shared<op::v0::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::v1::Add>(A, B), ParameterVector{A, B});

    auto backend = runtime::Backend::create("INTERPRETER");

    // Create some tensors for input/output
    shared_ptr<runtime::Tensor> a = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> b = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> result = backend->create_tensor(element::f32, shape);

    copy_data<float>(a, {1.f, 2.f, 3.f, 4.f});
    copy_data<float>(b, {5.f, 6.f, 7.f, 8.f});

    stringstream file;
    {
        auto handle = backend->compile(f);
        handle->save(file);
    }
    {
        auto handle = backend->load(file);
        ASSERT_NE(handle, nullptr);
        handle->call_with_validate({result}, {a, b});
        EXPECT_TRUE(test::all_close_f(read_vector<float>(result), {6.f, 8.f, 10.f, 12.f}));
    }
}
#endif

#if defined(NGRAPH_INTERPRETER_ENABLE) && defined(NGRAPH_CPU_ENABLE)
TEST(backend_api, executable_can_create_tensor)
{
    auto interpreter = runtime::Backend::create("INTERPRETER");
    auto cpu = runtime::Backend::create("CPU");

    EXPECT_TRUE(interpreter->executable_can_create_tensors());
    EXPECT_TRUE(cpu->executable_can_create_tensors());
}
#endif
