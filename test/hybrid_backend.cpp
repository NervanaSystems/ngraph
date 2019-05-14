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
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/visualize_tree.hpp"
#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/backend_manager.hpp"
#include "ngraph/runtime/hybrid/hybrid_backend.hpp"
#include "ngraph/runtime/hybrid/hybrid_util.hpp"
#include "ngraph/runtime/hybrid/op/function_call.hpp"
#include "ngraph/runtime/interpreter/int_backend.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/ndarray.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

static runtime::BackendConstructor* hybrid_creator()
{
    class HybridBackendConstructor : public runtime::BackendConstructor
    {
    public:
        std::shared_ptr<runtime::Backend> create(const std::string& config) override
        {
            vector<string> unsupported_0 = {"Add", "Max"};
            vector<string> unsupported_1 = {"Multiply"};
            vector<shared_ptr<runtime::Backend>> backend_list = {
                make_shared<runtime::interpreter::INTBackend>(unsupported_0),
                runtime::Backend::create("CPU")};

            return make_shared<runtime::hybrid::HybridBackend>(backend_list);
        }
    };

    static unique_ptr<runtime::BackendConstructor> s_backend_constructor(
        new HybridBackendConstructor());
    return s_backend_constructor.get();
}

TEST(HYBRID, function_call)
{
    vector<shared_ptr<runtime::Backend>> backend_list = {
        make_shared<runtime::interpreter::INTBackend>()};
    auto backend = make_shared<runtime::hybrid::HybridBackend>(backend_list);

    Shape shape{};
    shared_ptr<Function> inner_function;
    auto inner_A = make_shared<op::Parameter>(element::f32, shape);
    auto inner_B = make_shared<op::Parameter>(element::f32, shape);
    auto inner_C = make_shared<op::Parameter>(element::f32, shape);
    auto inner_R1 = (inner_A + inner_B) * inner_C;
    auto inner_R2 = (inner_A + inner_C) * inner_C;
    NodeVector inner_Result{inner_R1, inner_R2};
    inner_function =
        make_shared<Function>(inner_Result, ParameterVector{inner_A, inner_B, inner_C});

    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto C = make_shared<op::Parameter>(element::f32, shape);
    NodeVector fcall_args{A, B, C};
    auto H = make_shared<runtime::hybrid::op::FunctionCall>(
        inner_Result, fcall_args, *inner_function, backend_list[0]);
    auto G0 = make_shared<ngraph::op::GetOutputElement>(H, 0);
    auto G1 = make_shared<ngraph::op::GetOutputElement>(H, 1);
    NodeVector out{G0, G1};
    auto J = G0 + G1;
    auto f = make_shared<Function>(out, ParameterVector{A, B, C});

    shared_ptr<runtime::Tensor> a = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> b = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> c = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> r0 = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> r1 = backend->create_tensor(element::f32, shape);

    copy_data(a, vector<float>{2});
    copy_data(b, vector<float>{3});
    copy_data(c, vector<float>{4});

    auto exec = backend->compile(f);
    exec->call({r0, r1}, {a, b, c});

    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<ngraph::pass::VisualizeTree>("test.png");
    pass_manager.run_passes(f);
}

TEST(HYBRID, abc)
{
    const string backend_name = "H1";
    runtime::BackendManager::register_backend(backend_name, hybrid_creator());

    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto C = make_shared<op::Parameter>(element::f32, shape);
    auto D = make_shared<op::Parameter>(element::f32, shape);
    auto t1 = A * B;
    auto t2 = t1 * D;
    auto t3 = (t2 + C);
    auto t4 = (t3 + A) * t1;
    NodeVector result({t3, t4});
    auto f = make_shared<Function>(result, ParameterVector{A, B, C, D});

    shared_ptr<runtime::Backend> backend = runtime::Backend::create("H1");
    static_pointer_cast<runtime::hybrid::HybridBackend>(backend)->set_debug_enabled(true);

    // Create some tensors for input/output
    shared_ptr<runtime::Tensor> a = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> b = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> c = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> d = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> result1 = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> result2 = backend->create_tensor(element::f32, shape);

    copy_data(a, vector<float>{1, 2, 3, 4});
    copy_data(b, vector<float>{5, 6, 7, 8});
    copy_data(c, vector<float>{9, 10, 11, 12});
    copy_data(d, vector<float>{4, 3, 2, 1});

    auto handle = backend->compile(f);
    handle->call_with_validate({result1, result2}, {a, b, c, d});
    EXPECT_TRUE(
        test::all_close_f(read_vector<float>(result2), (vector<float>{150, 576, 1176, 1536})));
}

TEST(HYBRID, simple)
{
    const string backend_name = "H1";
    runtime::BackendManager::register_backend(backend_name, hybrid_creator());

    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::i8, shape);
    auto f = make_shared<Function>(make_shared<op::Max>(A, AxisSet{0, 1}), ParameterVector{A});

    shared_ptr<runtime::Backend> backend = runtime::Backend::create("H1");
    static_pointer_cast<runtime::hybrid::HybridBackend>(backend)->set_debug_enabled(true);

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i8, shape);
    copy_data(a, vector<int8_t>{1, 2, 3, 4});
    auto result = backend->create_tensor(element::i8, Shape{});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_EQ((vector<int8_t>{4}), read_vector<int8_t>(result));
}
