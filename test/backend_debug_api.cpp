/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "ngraph/log.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/runtime/interpreter/int_call_frame.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

TEST(INTERPRETER, nan_check_input)
{
    Shape shape{4};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Divide>(A, B), op::ParameterVector{A, B});

    auto manager = runtime::Manager::get("INTERPRETER");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    shared_ptr<runtime::interpreter::INT_CallFrame> icf =
        static_pointer_cast<runtime::interpreter::INT_CallFrame>(cf);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(a, vector<float>{2, 4, NAN, 16});
    auto b = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(b, vector<float>{1, 2, 1, 8});
    auto result = backend->make_primary_tensor_view(element::f32, shape);

    icf->set_nan_check(true);
    EXPECT_ANY_THROW(icf->call({result}, {a, b}));
}

TEST(INTERPRETER, nan_check_output)
{
    Shape shape{4};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Divide>(A, B), op::ParameterVector{A, B});

    auto manager = runtime::Manager::get("INTERPRETER");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    shared_ptr<runtime::interpreter::INT_CallFrame> icf =
        static_pointer_cast<runtime::interpreter::INT_CallFrame>(cf);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(a, vector<float>{2, 4, 0, 16});
    auto b = backend->make_primary_tensor_view(element::f32, shape);
    copy_data(b, vector<float>{1, 2, 0, 8});
    auto result = backend->make_primary_tensor_view(element::f32, shape);

    icf->set_nan_check(true);
    EXPECT_ANY_THROW(icf->call({result}, {a, b}));
}
