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

#include "ngraph/log.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/backend_manager.hpp"
#include "ngraph/runtime/hybrid/hybrid_backend.hpp"
#include "ngraph/runtime/interpreter/int_backend.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/ndarray.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

static runtime::Backend* hybrid_creator(const char* config)
{
    vector<string> unsupported_0 = {"Add"};
    vector<string> unsupported_1 = {"Multiply"};
    vector<shared_ptr<runtime::Backend>> backend_list = {
        make_shared<runtime::interpreter::INTBackend>(unsupported_0),
        make_shared<runtime::interpreter::INTBackend>(unsupported_1)};

    return new runtime::hybrid::HybridBackend(backend_list);
}

TEST(HYBRID, abc)
{
    const string backend_name = "H1";
    runtime::BackendManager::register_backend(backend_name, hybrid_creator);

    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto D = make_shared<op::Parameter>(element::f32, shape);
    auto t1 = A * B;
    auto t2 = t1 * D;
    auto C = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(((t2 + C) + A) * t1, ParameterVector{A, B, C, D});

    shared_ptr<runtime::Backend> backend = runtime::Backend::create("H1");
    static_pointer_cast<runtime::hybrid::HybridBackend>(backend)->set_debug_enabled(true);

    // Create some tensors for input/output
    shared_ptr<runtime::Tensor> a = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> b = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> c = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> d = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> result = backend->create_tensor(element::f32, shape);

    copy_data(a, vector<float>{1, 2, 3, 4});
    copy_data(b, vector<float>{5, 6, 7, 8});
    copy_data(c, vector<float>{9, 10, 11, 12});
    copy_data(d, vector<float>{4, 3, 2, 1});

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a, b, c, d});
    EXPECT_EQ(read_vector<float>(result), (vector<float>{150, 576, 1176, 1536}));
}
