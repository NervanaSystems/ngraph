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
#include "util/all_close_f.hpp"
#include "util/ndarray.hpp"
#include "ngraph/codegen/compiler.hpp"
#include "ngraph/codegen/execution_engine.hpp"

using namespace ngraph;
using namespace std;

TEST(cpu_codegen, abc)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto C = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>((A + B) * C, ParameterVector{A, B, C});

    auto backend = runtime::Backend::create("CPU");

    // Create some tensors for input/output
    shared_ptr<runtime::Tensor> a = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> b = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> c = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> result = backend->create_tensor(element::f32, shape);

    copy_data(a, test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
    copy_data(b, test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());
    copy_data(c, test::NDArray<float, 2>({{9, 10}, {11, 12}}).get_vector());

    ngraph::pass::PassConfig pass_config;
    pass_config.set_pass_attribute("CODEGEN", true);
    auto handle = backend->compile(f, pass_config);
    handle->call_with_validate({result}, {a, b, c});
    EXPECT_TRUE(test::all_close_f(read_vector<float>(result),
                                  (test::NDArray<float, 2>({{54, 80}, {110, 144}})).get_vector(),
                                  MIN_FLOAT_TOLERANCE_BITS));

    handle->call_with_validate({result}, {b, a, c});
    EXPECT_TRUE(test::all_close_f(read_vector<float>(result),
                                  (test::NDArray<float, 2>({{54, 80}, {110, 144}})).get_vector(),
                                  MIN_FLOAT_TOLERANCE_BITS));

    handle->call_with_validate({result}, {a, c, b});
    EXPECT_TRUE(test::all_close_f(read_vector<float>(result),
                                  (test::NDArray<float, 2>({{50, 72}, {98, 128}})).get_vector(),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

TEST(benchmark, c_compile)
{
    string source = R"(#include <stdio.h>
extern "C" void test()
{
    printf("Hello world\n");
}
)";
    NGRAPH_INFO << "\n" << source;

    stopwatch create;
    create.start();
    unique_ptr<codegen::Compiler> compiler(new codegen::Compiler());
    create.stop();
    NGRAPH_INFO << "construct compiler: " << create.get_milliseconds() << "ms";

    stopwatch compile;
    compile.start();
    auto module = compiler->compile(source);
    compile.stop();
    ASSERT_TRUE(module);
    NGRAPH_INFO << "compile: " << compile.get_milliseconds() << "ms";

    codegen::ExecutionEngine ee;
    stopwatch create_exec;
    create_exec.start();
    ee.add_module(module);
    ee.finalize();
    function<void()> test_entry = ee.find_function<void()>("test");
    create_exec.stop();
    NGRAPH_INFO << "create_exec: " << create_exec.get_milliseconds() << "ms";

    ASSERT_TRUE(test_entry);

    test_entry();
}

