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

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <iterator>
#include <limits>
#include <random>
#include <string>

// clang-format off
#ifdef ${BACKEND_NAME}_FLOAT_TOLERANCE_BITS
#define DEFAULT_FLOAT_TOLERANCE_BITS ${BACKEND_NAME}_FLOAT_TOLERANCE_BITS
#endif
// clang-format on

#include "gtest/gtest.h"
#include "ngraph/check.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/ndarray.hpp"
#include "util/random.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, transpose)
{
    //
    // Create a graph for f(x,perm) = Transpose(x,Convert<i64>(perm)). We'll do the permutation in
    // i32 and cast it to i64, just for fun (and to mirror the TensorFlow test I am porting here).
    //
    auto x = make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto perm = make_shared<op::v0::Parameter>(element::i32, PartialShape{Dimension::dynamic()});
    auto perm_i64 = make_shared<op::v0::Convert>(perm, element::i64);

    auto x_transpose = make_shared<op::v1::Transpose>(x, perm_i64);

    auto f = make_shared<Function>(OutputVector{x_transpose}, ParameterVector{x, perm});

    auto backend = runtime::Backend::create("${BACKEND_NAME}", true);

    auto ex = backend->compile(f);

    auto t_r = backend->create_dynamic_tensor(element::f32, PartialShape::dynamic());

    std::vector<Shape> x_shapes{Shape{2, 3}, Shape{2, 3}, Shape{2, 2, 3}};
    std::vector<std::vector<int32_t>> perms{{0, 1}, {1, 0}, {2, 1, 0}};
    std::vector<std::vector<float>> inputs{
        {1, 2, 3, 4, 5, 6}, {1, 2, 3, 4, 5, 6}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}};
    std::vector<Shape> expected_result_shapes{Shape{2, 3}, Shape{3, 2}, {3, 2, 2}};
    // Generated with numpy, so don't worry. :)
    std::vector<std::vector<float>> expected_results{
        {1, 2, 3, 4, 5, 6}, {1, 4, 2, 5, 3, 6}, {1, 7, 4, 10, 2, 8, 5, 11, 3, 9, 6, 12}};

    for (size_t i = 0; i < x_shapes.size(); i++)
    {
        auto t_x = backend->create_tensor(element::f32, x_shapes[i]);
        auto t_perm = backend->create_tensor(element::i32, Shape{perms[i].size()});

        copy_data(t_x, inputs[i]);
        copy_data(t_perm, perms[i]);

        ex->call_with_validate({t_r}, {t_x, t_perm});

        ASSERT_EQ(t_r->get_shape(), expected_result_shapes[i]);

        auto results = read_vector<float>(t_r);

        ASSERT_TRUE(test::all_close_f(results, expected_results[i], MIN_FLOAT_TOLERANCE_BITS));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, space_to_batch)
{
    auto data = make_shared<op::v0::Parameter>(element::f32, Shape{1, 2, 2, 3});
    auto block_shape =
        make_shared<op::v0::Constant>(element::i64, Shape{4}, vector<int64_t>{1, 2, 3, 2});
    auto pads_begin =
        make_shared<op::v0::Constant>(element::i64, Shape{4}, vector<int64_t>{0, 0, 1, 0});
    auto pads_end =
        make_shared<op::v0::Constant>(element::i64, Shape{4}, vector<int64_t>{0, 0, 0, 1});
    auto space_to_batch =
        make_shared<op::v1::SpaceToBatch>(data, block_shape, pads_begin, pads_end);
    auto function = make_shared<Function>(OutputVector{space_to_batch}, ParameterVector{data});
    auto test_case = test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_input<float>({0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f});
    test_case.add_expected_output<float>(Shape{12, 1, 1, 2},
                                         {
                                             0.f, 0.f, 0.f, 0.f, 0.f, 2.f,  1.f,  0.f,
                                             3.f, 5.f, 4.f, 0.f, 0.f, 0.f,  0.f,  0.f,
                                             6.f, 8.f, 7.f, 0.f, 9.f, 11.f, 10.f, 0.f,
                                         });
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, batch_to_space)
{
    auto data = make_shared<op::v0::Parameter>(element::f32, Shape{12, 1, 1, 2});
    auto block_shape =
        make_shared<op::v0::Constant>(element::i64, Shape{4}, vector<int64_t>{1, 2, 3, 2});
    auto pads_begin =
        make_shared<op::v0::Constant>(element::i64, Shape{4}, vector<int64_t>{0, 0, 1, 0});
    auto pads_end =
        make_shared<op::v0::Constant>(element::i64, Shape{4}, vector<int64_t>{0, 0, 0, 1});
    auto batch_to_space =
        make_shared<op::v1::BatchToSpace>(data, block_shape, pads_begin, pads_end);
    auto function = make_shared<Function>(OutputVector{batch_to_space}, ParameterVector{data});

    auto test_case = test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_input<float>({
        0.f, 0.f, 0.f, 0.f, 0.f, 2.f, 1.f, 0.f, 3.f, 5.f,  4.f,  0.f,
        0.f, 0.f, 0.f, 0.f, 6.f, 8.f, 7.f, 0.f, 9.f, 11.f, 10.f, 0.f,
    });
    test_case.add_expected_output<float>(
        Shape{1, 2, 2, 3}, {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, space_to_depth_block_first)
{
    auto A = make_shared<op::v0::Parameter>(element::f32, Shape{1, 2, 4, 4});
    const auto mode = ngraph::op::v0::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST;
    auto space_to_depth = make_shared<op::v0::SpaceToDepth>(A, mode, 2);
    auto function = make_shared<Function>(OutputVector{space_to_depth}, ParameterVector{A});

    auto test_case = test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_input<float>({0.f,  1.f,  2.f,  3.f,  4.f,  5.f,  6.f,  7.f,  8.f,  9.f,  10.f,
                                11.f, 12.f, 13.f, 14.f, 15.f, 16.f, 17.f, 18.f, 19.f, 20.f, 21.f,
                                22.f, 23.f, 24.f, 25.f, 26.f, 27.f, 28.f, 29.f, 30.f, 31.f});
    test_case.add_expected_output<float>(Shape{1, 8, 2, 2},
                                         {
                                             0.f, 2.f, 8.f,  10.f, 16.f, 18.f, 24.f, 26.f,
                                             1.f, 3.f, 9.f,  11.f, 17.f, 19.f, 25.f, 27.f,
                                             4.f, 6.f, 12.f, 14.f, 20.f, 22.f, 28.f, 30.f,
                                             5.f, 7.f, 13.f, 15.f, 21.f, 23.f, 29.f, 31.f,
                                         });
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, space_to_depth_depth_first)
{
    auto A = make_shared<op::v0::Parameter>(element::f32, Shape{1, 2, 4, 4});
    const auto mode = ngraph::op::v0::SpaceToDepth::SpaceToDepthMode::DEPTH_FIRST;
    auto space_to_depth = make_shared<op::v0::SpaceToDepth>(A, mode, 2);
    auto function = make_shared<Function>(OutputVector{space_to_depth}, ParameterVector{A});

    auto test_case = test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_input<float>({0.f,  16.f, 2.f,  18.f, 1.f,  17.f, 3.f,  19.f, 8.f,  24.f, 10.f,
                                26.f, 9.f,  25.f, 11.f, 27.f, 4.f,  20.f, 6.f,  22.f, 5.f,  21.f,
                                7.f,  23.f, 12.f, 28.f, 14.f, 30.f, 13.f, 29.f, 15.f, 31.f});
    test_case.add_expected_output<float>(
        Shape{1, 8, 2, 2}, {0.f,  2.f,  8.f,  10.f, 16.f, 18.f, 24.f, 26.f, 1.f,  3.f,  9.f,
                            11.f, 17.f, 19.f, 25.f, 27.f, 4.f,  6.f,  12.f, 14.f, 20.f, 22.f,
                            28.f, 30.f, 5.f,  7.f,  13.f, 15.f, 21.f, 23.f, 29.f, 31.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, depth_to_space_block_first)
{
    auto A = make_shared<op::v0::Parameter>(element::f32, Shape{1, 8, 2, 2});
    auto depth_to_space = make_shared<op::v0::DepthToSpace>(
        A, op::v0::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST, 2);
    auto function = make_shared<Function>(OutputVector{depth_to_space}, ParameterVector{A});

    auto test_case = test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_input<float>({
        0.f, 2.f, 8.f,  10.f, 16.f, 18.f, 24.f, 26.f, 1.f, 3.f, 9.f,  11.f, 17.f, 19.f, 25.f, 27.f,
        4.f, 6.f, 12.f, 14.f, 20.f, 22.f, 28.f, 30.f, 5.f, 7.f, 13.f, 15.f, 21.f, 23.f, 29.f, 31.f,
    });
    test_case.add_expected_output<float>(
        Shape{1, 2, 4, 4}, {0.f,  1.f,  2.f,  3.f,  4.f,  5.f,  6.f,  7.f,  8.f,  9.f,  10.f,
                            11.f, 12.f, 13.f, 14.f, 15.f, 16.f, 17.f, 18.f, 19.f, 20.f, 21.f,
                            22.f, 23.f, 24.f, 25.f, 26.f, 27.f, 28.f, 29.f, 30.f, 31.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, depth_to_space_depth_first)
{
    auto A = make_shared<op::v0::Parameter>(element::f32, Shape{1, 8, 2, 2});
    auto depth_to_space = make_shared<op::v0::DepthToSpace>(
        A, op::v0::DepthToSpace::DepthToSpaceMode::DEPTH_FIRST, 2);
    auto function = make_shared<Function>(OutputVector{depth_to_space}, ParameterVector{A});

    auto test_case = test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_input<float>({
        0.f, 2.f, 8.f,  10.f, 16.f, 18.f, 24.f, 26.f, 1.f, 3.f, 9.f,  11.f, 17.f, 19.f, 25.f, 27.f,
        4.f, 6.f, 12.f, 14.f, 20.f, 22.f, 28.f, 30.f, 5.f, 7.f, 13.f, 15.f, 21.f, 23.f, 29.f, 31.f,
    });
    test_case.add_expected_output<float>(
        Shape{1, 2, 4, 4}, {0.f,  16.f, 2.f,  18.f, 1.f,  17.f, 3.f,  19.f, 8.f,  24.f, 10.f,
                            26.f, 9.f,  25.f, 11.f, 27.f, 4.f,  20.f, 6.f,  22.f, 5.f,  21.f,
                            7.f,  23.f, 12.f, 28.f, 14.f, 30.f, 13.f, 29.f, 15.f, 31.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, depth_to_space_space_to_depth_block_first)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    Shape dts_input_shape{2, 32, 2, 4, 2, 4};
    size_t block_size = 2;

    auto dts_input = make_shared<op::v0::Parameter>(element::f32, dts_input_shape);
    auto depth_to_space = make_shared<op::v0::DepthToSpace>(
        dts_input, op::v0::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST, block_size);
    auto dts_func = make_shared<Function>(OutputVector{depth_to_space}, ParameterVector{dts_input});

    auto dts_input_tensor = backend->create_tensor(element::f32, dts_input_shape);
    const auto data_size = shape_size(dts_input_shape);
    vector<float> data(data_size);
    std::iota(data.begin(), data.end(), 0);
    copy_data(dts_input_tensor, data);
    const auto dts_output_shape = depth_to_space->get_output_shape(0);
    auto dts_output_tensor = backend->create_tensor(element::f32, dts_output_shape);
    auto handle = backend->compile(dts_func);
    handle->call_with_validate({dts_output_tensor}, {dts_input_tensor});
    auto dts_result = read_vector<float>(dts_output_tensor);

    // use depth_to_space output as space_to_depth input
    auto std_input = make_shared<op::v0::Parameter>(element::f32, dts_output_shape);
    auto space_to_depth = make_shared<op::v0::SpaceToDepth>(
        std_input, op::v0::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST, block_size);
    auto std_func = make_shared<Function>(OutputVector{space_to_depth}, ParameterVector{std_input});

    auto std_input_tensor = backend->create_tensor(element::f32, dts_output_shape);
    copy_data(std_input_tensor, dts_result);
    auto std_output_tensor = backend->create_tensor(element::f32, dts_input_shape);
    handle = backend->compile(std_func);
    handle->call_with_validate({std_output_tensor}, {std_input_tensor});
    auto std_result = read_vector<float>(std_output_tensor);

    // expected output of space_to_depth is input of depth_to_space
    ASSERT_EQ(dts_input_shape, space_to_depth->get_output_shape(0));
    EXPECT_TRUE(test::all_close_f(std_result, data, data_size));
}

NGRAPH_TEST(${BACKEND_NAME}, depth_to_space_space_to_depth_depth_first)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    Shape dts_input_shape{2, 32, 2, 4, 2, 4};
    size_t block_size = 2;

    auto dts_input = make_shared<op::v0::Parameter>(element::f32, dts_input_shape);
    auto depth_to_space = make_shared<op::v0::DepthToSpace>(
        dts_input, op::v0::DepthToSpace::DepthToSpaceMode::DEPTH_FIRST, block_size);
    auto dts_func = make_shared<Function>(OutputVector{depth_to_space}, ParameterVector{dts_input});

    auto dts_input_tensor = backend->create_tensor(element::f32, dts_input_shape);
    const auto data_size = shape_size(dts_input_shape);
    vector<float> data(data_size);
    std::iota(data.begin(), data.end(), 0);
    copy_data(dts_input_tensor, data);
    const auto dts_output_shape = depth_to_space->get_output_shape(0);
    auto dts_output_tensor = backend->create_tensor(element::f32, dts_output_shape);
    auto handle = backend->compile(dts_func);
    handle->call_with_validate({dts_output_tensor}, {dts_input_tensor});
    auto dts_result = read_vector<float>(dts_output_tensor);

    // use depth_to_space output as space_to_depth input
    auto std_input = make_shared<op::v0::Parameter>(element::f32, dts_output_shape);
    auto space_to_depth = make_shared<op::v0::SpaceToDepth>(
        std_input, op::v0::SpaceToDepth::SpaceToDepthMode::DEPTH_FIRST, block_size);
    auto std_func = make_shared<Function>(OutputVector{space_to_depth}, ParameterVector{std_input});

    auto std_input_tensor = backend->create_tensor(element::f32, dts_output_shape);
    copy_data(std_input_tensor, dts_result);
    auto std_output_tensor = backend->create_tensor(element::f32, dts_input_shape);
    handle = backend->compile(std_func);
    handle->call_with_validate({std_output_tensor}, {std_input_tensor});
    auto std_result = read_vector<float>(std_output_tensor);

    // expected output of space_to_depth is input of depth_to_space
    ASSERT_EQ(dts_input_shape, space_to_depth->get_output_shape(0));
    EXPECT_TRUE(test::all_close_f(std_result, data, data_size));
}
