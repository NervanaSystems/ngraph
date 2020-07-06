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

#include <gtest/gtest.h>
#include <thread>

#include "ngraph/ngraph.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/runtime/backend.hpp"
#include "ngraph/util.hpp"
#include "ngraph/op/experimental/compiled_kernel.hpp"
#include "util/test_tools.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/op/result.hpp"
#include "ngraph/op/topk.hpp"
#include "ngraph/pass/hybrid.hpp"
#include "ngraph/pass/validate.hpp"
#include "util/all_close_f.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace ngraph;
using namespace std;

// TBD tests
// add enmasse input/output tensors
// clean up tests to use function to generate functionCall
// TEST(function_call, silly_input_tensor_nnp)
// {
//     auto backend = runtime::Backend::create("NNP");

//     auto A = make_shared<op::Parameter>(element::f32, Shape{2, 3, 2});
//     auto B = make_shared<op::Parameter>(element::f32, Shape{3, 4});
//     auto f =
//         make_shared<Function>(make_shared<op::ArgMax>(A, 1, element::i32), ParameterVector{A, B});
//     auto exec = backend->compile(f);

//     // Create some tensors for input/output
//     auto a = exec->create_input_tensor(0);
//     copy_data(a, vector<float>{12, 1, 2, 13, 4, 5, 8, 7, 10, 9, 6, 11});
//     auto b = exec->create_input_tensor(1);
//     copy_data(b, vector<float>{1, 2, 4, 8, 5, 6, 7, 8, 9, 10, 11, 12});
//     auto result = exec->create_output_tensor(0);

//     exec->call_with_validate({result}, {a, b});
// }

// TEST(function_call, create_tensor_nnp)
// {
//     auto backend = runtime::Backend::create("NNP");

//     Shape shape{2, 2};
//     auto A = make_shared<op::Parameter>(element::f32, shape);
//     auto B = make_shared<op::Parameter>(element::f32, shape);
//     auto f = make_shared<Function>(make_shared<op::Subtract>(A, B), ParameterVector{A, B});

//     auto exec = backend->compile(f);

//     // Create some tensors for input/output
//     auto a = exec->create_input_tensor(0);
//     copy_data(a, vector<float>{2, 4, 8, 16});
//     auto b = exec->create_input_tensor(1);
//     copy_data(b, vector<float>{1, 2, 4, 8});
//     auto result = exec->create_output_tensor(0);
//     auto nnp_tensor = dynamic_pointer_cast<runtime::nnp::NNPTensor>(a);
//     EXPECT_NE(nnp_tensor, nullptr);
//     nnp_tensor = dynamic_pointer_cast<runtime::nnp::NNPTensor>(result);
//     EXPECT_NE(nnp_tensor, nullptr);

//     exec->call_with_validate({result}, {a, b});
//     EXPECT_TRUE(test::all_close_f((vector<float>{1, 2, 4, 8}), read_vector<float>(result)));
// }

// TEST(hybrid, create_shapes)
// {
//     Shape shape0{}, shape1{1}, shape2{1, 2};

//     auto backend = runtime::Backend::create("NNP");

//     auto A = make_shared<op::Parameter>(element::f32, shape0);
//     auto B = make_shared<op::Parameter>(element::f32, shape1);
//     auto C = make_shared<op::Parameter>(element::f32, shape2);

//     auto t1 = make_shared<op::Multiply>(A, A);
//     auto t2 = make_shared<op::Negative>(B);
//     auto t3 = make_shared<op::Add>(C, C);

//     auto f = make_shared<Function>(NodeVector{t1, t2, t3}, ParameterVector{A, B, C});
//     auto exec = backend->compile(f);
//     auto a_ = exec->create_input_tensor(0);
//     auto b_ = exec->create_input_tensor(1);
//     auto c_ = exec->create_input_tensor(2);

//     EXPECT_EQ(4, a_->get_size_in_bytes());
//     EXPECT_EQ(4, b_->get_size_in_bytes());
//     EXPECT_EQ(8, c_->get_size_in_bytes());
// }

// TEST(function_call, DISABLED_topk)
// {
//     auto backend = runtime::Backend::create("NNP");

//     // Create the graph
//     Shape shape{3};
//     auto power_input1 = make_shared<op::Parameter>(element::f32, shape);
//     auto power_input2 = make_shared<op::Parameter>(element::f32, shape);
//     auto equal_input = make_shared<op::Parameter>(element::f32, shape);

//     auto power = make_shared<op::Power>(power_input1, power_input2);
//     auto topk = make_shared<op::TopK>(power, 0, element::i32, 0, true);
//     auto out_value = make_shared<op::GetOutputElement>(topk, 1);
//     auto out_index = make_shared<op::GetOutputElement>(topk, 0);
//     auto add = make_shared<op::Add>(out_value, power_input2);

//     auto f = make_shared<Function>(NodeVector{make_shared<op::Equal>(add, equal_input), out_index},
//                                    ParameterVector{power_input1, power_input2, equal_input});

//     pass::Manager pass_manager;
//     pass_manager.register_pass<pass::VisualizeTree>("before_compile_function_call.png");
//     pass_manager.run_passes(f);

//     auto count = count_ops_of_type<op::Parameter>(f);
//     EXPECT_EQ(count, 3);
//     count = count_ops_of_type<op::Power>(f);
//     EXPECT_EQ(count, 1);
//     count = count_ops_of_type<op::FunctionCall>(f);
//     EXPECT_EQ(count, 0);
//     count = count_ops_of_type<op::TopK>(f);
//     EXPECT_EQ(count, 1);
//     count = count_ops_of_type<op::Add>(f);
//     EXPECT_EQ(count, 1);
//     count = count_ops_of_type<op::Equal>(f);
//     EXPECT_EQ(count, 1);
//     count = count_ops_of_type<op::Result>(f);
//     EXPECT_EQ(count, 2);

//     // Compile the graph
//     pass::Manager pass_manager2;
//     pass_manager2.register_pass<pass::VisualizeTree>("after_compile_function_call.png");
//     auto exec = as_nnp_executable(backend->compile(f));
//     auto compiled_f = exec->get_compiled_function();
//     pass_manager2.run_passes(compiled_f);
//     count = count_ops_of_type<op::Parameter>(compiled_f);
//     EXPECT_EQ(count, 3);
//     count = count_ops_of_type<op::Power>(compiled_f);
//     EXPECT_EQ(count, 1);
//     count = count_ops_of_type<op::FunctionCall>(compiled_f);
//     EXPECT_EQ(count, 1);
//     count = count_ops_of_type<op::TopK>(compiled_f);
//     EXPECT_EQ(count, 0);
//     count = count_ops_of_type<op::Add>(compiled_f);
//     EXPECT_EQ(count, 0);
//     count = count_ops_of_type<op::Equal>(compiled_f);
//     EXPECT_EQ(count, 0);
//     count = count_ops_of_type<op::Result>(compiled_f);
//     EXPECT_EQ(count, 2);
// }

// // backwards compatibility test for old api: create_tensor
// TEST(function_call, create_tensor_function_call)
// {
//     auto backend = runtime::Backend::create("NNP");

//     Shape shape{3};
//     auto power_input1 = make_shared<op::Parameter>(element::f32, shape);
//     auto power_input2 = make_shared<op::Parameter>(element::f32, shape);
//     auto add2_input = make_shared<op::Parameter>(element::f32, shape);

//     auto power = make_shared<op::Power>(power_input1, power_input2);
//     auto topk = make_shared<op::TopK>(power, 0, element::i32, 0, true);
//     auto out_value = make_shared<op::GetOutputElement>(topk, 1);
//     auto add = make_shared<op::Add>(out_value, power_input2);

//     auto f = make_shared<Function>(NodeVector{make_shared<op::Add>(add, add2_input), power},
//                                    ParameterVector{power_input1, power_input2, add2_input});

//     auto exec = as_nnp_executable(backend->compile(f));
//     auto compiled_f = exec->get_compiled_function();
//     // pass::Manager pass_manager2;
//     // pass_manager2.register_pass<pass::VisualizeTree>("after_compile_hybrid.png");
//     // pass_manager2.run_passes(compiled_f);

//     auto result1 = backend->create_tensor(element::f32, shape);
//     auto result2 = backend->create_tensor(element::f32, shape);
//     auto a = backend->create_tensor(element::f32, shape);
//     copy_data(a, vector<float>{1, 2, 3});
//     auto b = backend->create_tensor(element::f32, shape);
//     copy_data(b, vector<float>{4, 5, 6});
//     auto c = backend->create_tensor(element::f32, shape);
//     copy_data(c, vector<float>{9, 8, 7});

//     exec->call({result1, result2}, {a, b, c});

//     EXPECT_TRUE(test::all_close_f((vector<float>{742, 45, 14}), read_vector<float>(result1)));
// }

// TEST(function_call, create_hybrid_tensor_function_call)
// {
//     auto backend = runtime::Backend::create("NNP");

//     // Create the graph
//     Shape shape{3};
//     auto power_input1 = make_shared<op::Parameter>(element::f32, shape);
//     auto power_input2 = make_shared<op::Parameter>(element::f32, shape);
//     auto add2_input = make_shared<op::Parameter>(element::f32, shape);

//     auto power = make_shared<op::Power>(power_input1, power_input2);
//     auto topk = make_shared<op::TopK>(power, 0, element::i32, 0, true);
//     auto out_value = make_shared<op::GetOutputElement>(topk, 1);
//     auto out_index = make_shared<op::GetOutputElement>(topk, 0);
//     auto add = make_shared<op::Add>(out_value, power_input2);

//     auto f =
//         make_shared<Function>(NodeVector{make_shared<op::Add>(add, add2_input), out_index, power},
//                               ParameterVector{power_input1, power_input2, add2_input});

//     auto exec = as_nnp_executable(backend->compile(f));
//     auto compiled_f = exec->get_compiled_function();
//     // pass::Manager pass_manager2;
//     // pass_manager2.register_pass<pass::VisualizeTree>("after_compile_hybrid.png");
//     // pass_manager2.run_passes(compiled_f);
//     auto result1_ = exec->create_output_tensor(0);
//     auto result2_ = exec->create_output_tensor(1);
//     auto result3_ = exec->create_output_tensor(2);
//     auto a_ = exec->create_input_tensor(0);
//     copy_data(a_, vector<float>{1, 2, 3});
//     auto b_ = exec->create_input_tensor(1);
//     copy_data(b_, vector<float>{4, 5, 6});
//     auto c_ = exec->create_input_tensor(2);
//     copy_data(c_, vector<float>{9, 8, 7});

//     EXPECT_EQ(12, a_->get_size_in_bytes());
//     EXPECT_EQ(12, b_->get_size_in_bytes());
//     EXPECT_EQ(12, c_->get_size_in_bytes());
//     EXPECT_EQ(12, result1_->get_size_in_bytes());
//     EXPECT_EQ(12, result2_->get_size_in_bytes());
//     EXPECT_EQ(12, result3_->get_size_in_bytes());

//     exec->call_with_validate({result1_, result2_, result3_}, {a_, b_, c_});

//     EXPECT_TRUE(test::all_close_f((vector<float>{742, 45, 14}), read_vector<float>(result1_)));
//     EXPECT_EQ((vector<int32_t>{2, 1, 0}), read_vector<int32_t>(result2_));
// }

// TEST(function_call, input_coalesce)
// {
//     Shape shape{1};

//     auto A = make_shared<op::Parameter>(element::f32, shape);
//     auto B = make_shared<op::Parameter>(element::f32, shape);
//     auto C = make_shared<op::Parameter>(element::f32, shape);
//     auto D = make_shared<op::Parameter>(element::f32, shape);

//     auto t1 = make_shared<op::Multiply>(A, B);
//     auto t6 = make_shared<op::Negative>(A);
//     auto t2 = make_shared<op::Multiply>(C, t6);
//     auto t3 = make_shared<op::Multiply>(A, D);
//     auto t4 = make_shared<op::Multiply>(t1, t2);
//     auto t5 = make_shared<op::Add>(t3, t4);

//     auto func = make_shared<Function>(NodeVector{t1, t2, t3, t4, t5}, ParameterVector{A, B, C, D});

//     // This works
//     auto backend = runtime::Backend::create("NNP");

//     pass::Manager pass_manager;
//     t1->set_placement(Placement::CPU);
//     t2->set_placement(Placement::CPU);
//     t3->set_placement(Placement::CPU);
//     t4->set_placement(Placement::CPU);
//     shared_ptr<runtime::Backend> fallback_backend = runtime::Backend::create("CPU");
//     pass_manager.register_pass<runtime::nnp::pass::AssignLayout>();
//     pass_manager.register_pass<ngraph::runtime::nnp::pass::Validate>();
//     pass_manager.register_pass<runtime::nnp::pass::Hybrid>(fallback_backend);
//     pass_manager.register_pass<ngraph::runtime::nnp::pass::Validate>();
//     pass_manager.run_passes(func);

//     auto exec = backend->compile(func);

//     auto a = backend->create_tensor(element::f32, shape);
//     auto b = backend->create_tensor(element::f32, shape);
//     auto c = backend->create_tensor(element::f32, shape);
//     auto d = backend->create_tensor(element::f32, shape);

//     auto r1 = backend->create_tensor(element::f32, shape);
//     auto r2 = backend->create_tensor(element::f32, shape);
//     auto r3 = backend->create_tensor(element::f32, shape);
//     auto r4 = backend->create_tensor(element::f32, shape);
//     auto r5 = backend->create_tensor(element::f32, shape);

//     copy_data(a, vector<float>({2}));
//     copy_data(b, vector<float>({3}));
//     copy_data(c, vector<float>({4}));
//     copy_data(d, vector<float>({5}));

//     exec->call_with_validate({r1, r2, r3, r4, r5}, {a, b, c, d});

//     auto r1a = read_vector<float>(r1);
//     auto r2a = read_vector<float>(r2);
//     auto r3a = read_vector<float>(r3);
//     auto r4a = read_vector<float>(r4);
//     auto r5a = read_vector<float>(r5);

//     EXPECT_TRUE(test::all_close_f(r1a, {6}));
//     EXPECT_TRUE(test::all_close_f(r2a, {-8}));
//     EXPECT_TRUE(test::all_close_f(r3a, {10}));
//     EXPECT_TRUE(test::all_close_f(r4a, {-48}));
//     EXPECT_TRUE(test::all_close_f(r5a, {-38}));
// }

// TEST(function_call, hybrid_output)
// {
//     Shape shape{2, 2};
//     auto A = make_shared<op::Parameter>(element::f32, shape);
//     // op::Sin is selected because it is a fallback op that likely will remain a fallback op
//     // Tensor B must be CPU fallback
//     auto R1 = make_shared<op::Sin>(A);
//     auto B = make_shared<op::Parameter>(element::f32, shape);
//     auto R2 = R1 * B;
//     auto f = make_shared<Function>(OutputVector{R1, R2}, ParameterVector{A, B});

//     auto backend = runtime::Backend::create("NNP");
//     auto exec = backend->compile(f);

//     // Create some tensors for input/output
//     shared_ptr<runtime::Tensor> a = exec->create_input_tensor(0);
//     shared_ptr<runtime::Tensor> b = exec->create_input_tensor(1);
//     shared_ptr<runtime::Tensor> r1 = exec->create_output_tensor(0);
//     shared_ptr<runtime::Tensor> r2 = exec->create_output_tensor(1);

//     copy_data(a, vector<float>({1, 2, 3, 4}));
//     copy_data(b, vector<float>({5, 6, 7, 8}));

//     exec->call_with_validate({r1, r2}, {a, b});

//     vector<float> a1 = read_vector<float>(r1);
//     vector<float> a2 = read_vector<float>(r2);
//     vector<float> e1{sinf(1), sinf(2), sinf(3), sinf(4)};
//     vector<float> e2{5 * sinf(1), 6 * sinf(2), 7 * sinf(3), 8 * sinf(4)};

//     EXPECT_TRUE(test::all_close_f(read_vector<float>(r1), e1));
//     EXPECT_TRUE(test::all_close_f(read_vector<float>(r2), e2));
// }

// TEST(function_call, output_reshape)
// {
//     Shape shape{8, 384, 1};
//     Shape shape2{8, 384};
//     vector<int64_t> const_data(1);
//     auto A = make_shared<op::Parameter>(element::f32, shape);
//     // op::Sin is selected because it is a fallback op that likely will remain a fallback op
//     // Tensor B must be CPU fallback
//     auto B = make_shared<op::Sin>(A);
//     auto Br = make_shared<op::Reshape>(B, AxisVector{0, 1, 2}, shape2);
//     auto M = make_shared<op::Max>(Br, AxisSet{1});

//     auto f = make_shared<Function>(OutputVector{Br, M}, ParameterVector{A});

//     auto backend = runtime::Backend::create("NNP");
//     auto exec = backend->compile(f);

//     // Create some tensors for input/output
//     shared_ptr<runtime::Tensor> a = exec->create_input_tensor(0);
//     shared_ptr<runtime::Tensor> r1 = exec->create_output_tensor(0);
//     shared_ptr<runtime::Tensor> r2 = exec->create_output_tensor(1);

//     // This test is only checking that the graph compiles and does not need to check
//     // results.
//     exec->call_with_validate({r1, r2}, {a});
// }
