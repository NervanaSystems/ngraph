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
#include "ngraph/op/batch_norm.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/visualize_tree.hpp"
#include "ngraph/runtime/cpu/pass/cpu_fusion.hpp"
#include "ngraph/serializer.hpp"
#include "ngraph/util.hpp"
#include "nlohmann/json.hpp"
#include "util/all_close.hpp"
#include "util/autodiff/backprop_function.hpp"
#include "util/autodiff/numeric_compare.hpp"
#include "util/ndarray.hpp"
#include "util/random.hpp"
#include "util/test_tools.hpp"

using namespace ngraph;
using namespace std;

class UnhandledOp : public ngraph::op::Abs
{
public:
    UnhandledOp(const std::shared_ptr<Node>& arg)
        : Abs(arg)
    {
    }
};

TEST(cpu_test, unhandled_op)
{
    auto A = make_shared<op::Parameter>(element::f32, Shape{});
    auto unhandled = make_shared<UnhandledOp>(A);
    auto f = make_shared<Function>(unhandled, op::ParameterVector{A});
    auto backend = runtime::Backend::create("CPU");
    ASSERT_THROW(backend->compile(f), ngraph_error);
}

TEST(cpu_test, trivial_in_place_relu)
{
    auto A = make_shared<op::Parameter>(element::f32, Shape{16, 1});
    auto B = make_shared<op::Parameter>(element::f32, Shape{16, 1});
    auto add = A + B;
    auto relu = make_shared<op::Relu>(add);
    auto f = make_shared<Function>(relu, op::ParameterVector{A, B});
    auto backend = runtime::Backend::create("CPU");
    (backend->compile(f));
    ASSERT_EQ(relu->get_outputs().at(0).get_tensor().get_pool_offset(),
              add->get_outputs().at(0).get_tensor().get_pool_offset());
}

TEST(cpu_test, trivial_in_place_relu_fail)
{
    auto A = make_shared<op::Parameter>(element::f32, Shape{16, 1});
    auto B = make_shared<op::Parameter>(element::f32, Shape{16, 1});
    auto add = A + B;
    auto relu = make_shared<op::Relu>(add);
    auto add2 = relu + add;
    auto f = make_shared<Function>(add2, op::ParameterVector{A, B});
    auto backend = runtime::Backend::create("CPU");
    (backend->compile(f));
    ASSERT_NE(relu->get_outputs().at(0).get_tensor().get_pool_offset(),
              add->get_outputs().at(0).get_tensor().get_pool_offset());
}

#ifdef NGRAPH_TBB_ENABLE
TEST(cpu_test, abc_tbb)
{
    // Force TBB flow graph generation in the CPU backend
    // This has no effect on other backends
    bool use_tbb = (getenv("NGRAPH_CPU_USE_TBB") != nullptr);
    if (!use_tbb)
    {
        setenv("NGRAPH_CPU_USE_TBB", "1", 1);
    }

    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto C = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>((A + B) * C, op::ParameterVector{A, B, C});

    auto backend = runtime::Backend::create("CPU");

    // Create some tensors for input/output
    shared_ptr<runtime::TensorView> a = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::TensorView> b = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::TensorView> c = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::TensorView> result = backend->create_tensor(element::f32, shape);

    copy_data(a, test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
    copy_data(b, test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());
    copy_data(c, test::NDArray<float, 2>({{9, 10}, {11, 12}}).get_vector());

    backend->call(f, {result}, {a, b, c});
    EXPECT_EQ(read_vector<float>(result),
              (test::NDArray<float, 2>({{54, 80}, {110, 144}})).get_vector());

    backend->call(f, {result}, {b, a, c});
    EXPECT_EQ(read_vector<float>(result),
              (test::NDArray<float, 2>({{54, 80}, {110, 144}})).get_vector());

    backend->call(f, {result}, {a, c, b});
    EXPECT_EQ(read_vector<float>(result),
              (test::NDArray<float, 2>({{50, 72}, {98, 128}})).get_vector());

    if (!use_tbb)
    {
        unsetenv("NGRAPH_CPU_USE_TBB");
    }
}
#endif // NGRAPH_TBB_ENABLE
