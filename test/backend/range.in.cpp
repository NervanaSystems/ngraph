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
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/min_max_propagation.hpp"
#include "util/all_close_f.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

template <typename T>
struct RangeTest
{
    T start;
    T stop;
    T step;
    Shape expected_result_shape;
    std::vector<T> expected_result;
};

// TODO(amprocte): We should test this with more than just int32, but there is a bug in the
// handling of element type-changing that is currently blocking doing that easily.
NGRAPH_TEST(${BACKEND_NAME}, range)
{
    // Create a graph for f(start,stop,step) = Range(start,stop,step).
    auto start = make_shared<op::Parameter>(element::i32, Shape{});
    auto stop = make_shared<op::Parameter>(element::i32, Shape{});
    auto step = make_shared<op::Parameter>(element::i32, Shape{});

    auto range = make_shared<op::Range>(start, stop, step);

    auto f = make_shared<Function>(NodeVector{range}, ParameterVector{start, stop, step});

    auto backend = runtime::Backend::create("${BACKEND_NAME}", true);

    auto ex = backend->compile(f);

    auto t_r = backend->create_dynamic_tensor(element::i32, PartialShape::dynamic());

    std::vector<RangeTest<int32_t>> int32_tests = {
        RangeTest<int32_t>{0, 10, 1, Shape{10}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}},
        RangeTest<int32_t>{-5, 6, 3, Shape{4}, {-5, -2, 1, 4}},
        RangeTest<int32_t>{10, 0, 1, Shape{0}, {}},
        RangeTest<int32_t>{10, 5, -3, Shape{2}, {10, 7}}};

    for (auto& test : int32_tests)
    {
        auto t_start = backend->create_tensor(element::i32, Shape{});
        auto t_stop = backend->create_tensor(element::i32, Shape{});
        auto t_step = backend->create_tensor(element::i32, Shape{});

        copy_data(t_start, std::vector<int32_t>{test.start});
        copy_data(t_stop, std::vector<int32_t>{test.stop});
        copy_data(t_step, std::vector<int32_t>{test.step});

        ex->call_with_validate({t_r}, {t_start, t_stop, t_step});

        ASSERT_EQ(t_r->get_element_type(), element::i32);
        ASSERT_EQ(t_r->get_shape(), test.expected_result_shape);

        auto results = read_vector<int32_t>(t_r);

        ASSERT_EQ(results, test.expected_result);
    }
}

NGRAPH_TEST(${BACKEND_NAME}, range_subgraph)
{
    // Create a graph for f(start,stop,step) = Range(start,stop,step).
    auto start = make_shared<op::Parameter>(element::i32, Shape{});
    auto stop = make_shared<op::Parameter>(element::i32, Shape{});
    auto step = make_shared<op::Parameter>(element::i32, Shape{});
    auto start_2 = make_shared<op::Parameter>(element::i32, Shape{});
    PartialShape out_max_shape{15};
    PartialShape out_max_shape_2{10};

    // subgraph
    auto range = make_shared<op::Range>(start, stop, step);
    auto negative = make_shared<op::Negative>(range);
    auto abs = make_shared<op::Abs>(negative);
    auto sum = make_shared<op::Sum>(abs, AxisSet{0});
    auto range_2 = make_shared<op::Range>(start_2, sum, step);
    auto negative_2 = make_shared<op::Negative>(range_2);

    range->output(0).set_max_partial_shape(out_max_shape);
    range_2->output(0).set_max_partial_shape(out_max_shape_2);
    auto f =
        make_shared<Function>(NodeVector{negative_2}, ParameterVector{start, stop, step, start_2});

    auto backend = runtime::Backend::create("${BACKEND_NAME}", true);

    auto handle = backend->compile(f);
    pass::Manager passes;
    passes.register_pass<pass::MinMaxShapePropagation>();
    passes.run_passes(f);

    auto t_start = backend->create_tensor(element::i32, Shape{});
    copy_data(t_start, vector<int32_t>{0});
    auto t_stop = backend->create_tensor(element::i32, Shape{});
    copy_data(t_stop, vector<int32_t>{10});
    auto t_step = backend->create_tensor(element::i32, Shape{});
    copy_data(t_step, vector<int32_t>{1});
    auto t_start_2 = backend->create_tensor(element::i32, Shape{});
    copy_data(t_start_2, vector<int32_t>{40});
    auto result = backend->create_dynamic_tensor(element::i32, PartialShape::dynamic());

    vector<int32_t> expected_result{-40, -41, -42, -43, -44};

    EXPECT_EQ(out_max_shape, range->get_output_shape(0));
    EXPECT_EQ(out_max_shape, range->output(0).get_max_partial_shape());
    EXPECT_EQ(out_max_shape, negative->get_output_shape(0));

    EXPECT_EQ(out_max_shape, negative->output(0).get_max_partial_shape());
    EXPECT_EQ(out_max_shape, abs->output(0).get_max_partial_shape());
    EXPECT_EQ(out_max_shape, sum->output(0).get_max_partial_shape());
    EXPECT_EQ(out_max_shape_2, range_2->output(0).get_max_partial_shape());
    EXPECT_EQ(out_max_shape_2, negative_2->output(0).get_max_partial_shape());
    handle->call_with_validate({result}, {t_start, t_stop, t_step, t_start_2});
    EXPECT_EQ(PartialShape{5}, result->get_shape());
    ASSERT_EQ(expected_result, read_vector<int32_t>(result));
}
