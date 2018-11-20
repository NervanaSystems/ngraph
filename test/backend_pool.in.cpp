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
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <random>
#include <string>

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/ndarray.hpp"
#include "util/random.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, max_pool_1d_1channel_1image)
{
    Shape shape_a{1, 1, 14};
    Shape window_shape{3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{1, 1, 12};
    auto f = make_shared<Function>(make_shared<op::MaxPool>(A, window_shape), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 3>{{{0, 1, 0, 2, 1, 0, 3, 2, 0, 0, 2, 0, 0, 0}}}.get_vector());
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call_with_validate(f, {result}, {a});
    EXPECT_EQ((test::NDArray<float, 3>({{{1, 2, 2, 2, 3, 3, 3, 2, 2, 2, 2, 0}}}).get_vector()),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, max_pool_1d_1channel_2image)
{
    Shape shape_a{2, 1, 14};
    Shape window_shape{3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 1, 12};
    auto f = make_shared<Function>(make_shared<op::MaxPool>(A, window_shape), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 3>({{{0, 1, 0, 2, 1, 0, 3, 2, 0, 0, 2, 0, 0, 0}},
                                       {{0, 2, 1, 1, 0, 0, 0, 2, 0, 1, 0, 0, 1, 2}}})
                  .get_vector());
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call_with_validate(f, {result}, {a});
    EXPECT_EQ((test::NDArray<float, 3>(
                   {{{1, 2, 2, 2, 3, 3, 3, 2, 2, 2, 2, 0}}, {{2, 2, 1, 1, 0, 2, 2, 2, 1, 1, 1, 2}}})
                   .get_vector()),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, max_pool_1d_2channel_2image)
{
    Shape shape_a{2, 2, 14};
    Shape window_shape{3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 2, 12};
    auto f = make_shared<Function>(make_shared<op::MaxPool>(A, window_shape), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 3>({{{0, 1, 0, 2, 1, 0, 3, 2, 0, 0, 2, 0, 0, 0},
                                        {0, 0, 0, 2, 0, 0, 2, 3, 0, 1, 2, 0, 1, 0}},

                                       {{0, 2, 1, 1, 0, 0, 0, 2, 0, 1, 0, 0, 1, 2},
                                        {2, 1, 0, 0, 1, 0, 2, 0, 0, 0, 1, 1, 2, 0}}})
                  .get_vector());
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call_with_validate(f, {result}, {a});
    EXPECT_EQ((test::NDArray<float, 3>(
                   {{{1, 2, 2, 2, 3, 3, 3, 2, 2, 2, 2, 0}, {0, 2, 2, 2, 2, 3, 3, 3, 2, 2, 2, 1}},

                    {{2, 2, 1, 1, 0, 2, 2, 2, 1, 1, 1, 2}, {2, 1, 1, 1, 2, 2, 2, 0, 1, 1, 2, 2}}})
                   .get_vector()),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, max_pool_2d_2channel_2image)
{
    Shape shape_a{2, 2, 5, 5};
    Shape window_shape{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 2, 4, 3};
    auto f = make_shared<Function>(make_shared<op::MaxPool>(A, window_shape), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 4>({{{{0, 1, 0, 2, 1}, // img 0 chan 0
                                         {0, 3, 2, 0, 0},
                                         {2, 0, 0, 0, 1},
                                         {2, 0, 1, 1, 2},
                                         {0, 2, 1, 0, 0}},

                                        {{0, 0, 0, 2, 0}, // img 0 chan 1
                                         {0, 2, 3, 0, 1},
                                         {2, 0, 1, 0, 2},
                                         {3, 1, 0, 0, 0},
                                         {2, 0, 0, 0, 0}}},

                                       {{{0, 2, 1, 1, 0}, // img 1 chan 0
                                         {0, 0, 2, 0, 1},
                                         {0, 0, 1, 2, 3},
                                         {2, 0, 0, 3, 0},
                                         {0, 0, 0, 0, 0}},

                                        {{2, 1, 0, 0, 1}, // img 1 chan 1
                                         {0, 2, 0, 0, 0},
                                         {1, 1, 2, 0, 2},
                                         {1, 1, 1, 0, 1},
                                         {1, 0, 0, 0, 2}}}})
                  .get_vector());
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call_with_validate(f, {result}, {a});
    EXPECT_EQ((test::NDArray<float, 4>({{{{3, 3, 2}, // img 0 chan 0
                                          {3, 3, 2},
                                          {2, 1, 2},
                                          {2, 2, 2}},

                                         {{3, 3, 3}, // img 0 chan 1
                                          {3, 3, 3},
                                          {3, 1, 2},
                                          {3, 1, 0}}},

                                        {{{2, 2, 2}, // img 1 chan 0
                                          {2, 2, 3},
                                          {2, 3, 3},
                                          {2, 3, 3}},

                                         {{2, 2, 1}, // img 1 chan 1
                                          {2, 2, 2},
                                          {2, 2, 2},
                                          {1, 1, 2}}}})
                   .get_vector()),
              read_vector<float>(result));
}

//this test cover the case with multiple image and with asymetric pad
//one bug been found on GPU side is covered by this test
NGRAPH_TEST(${BACKEND_NAME}, max_pool_2d_2channel_2image_asym_pad)
{
    Shape shape_a{2, 2, 4, 4};
    Shape window_shape{3, 3};
    auto window_movement_strides = Strides{2, 2};
    Shape padding_below{0, 0};
    Shape padding_above{1, 1};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 2, 2, 2};
    auto f = make_shared<Function>(
        make_shared<op::MaxPool>(
            A, window_shape, window_movement_strides, padding_below, padding_above),
        ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 4>({{{{0, 1, 0, 2}, // img 0 chan 0
                                         {0, 3, 2, 0},
                                         {2, 0, 0, 0},
                                         {0, 2, 1, 0}},

                                        {{0, 0, 0, 2}, // img 0 chan 1
                                         {0, 2, 3, 0},
                                         {2, 0, 1, 0},
                                         {2, 0, 0, 0}}},

                                       {{{0, 2, 1, 1}, // img 1 chan 0
                                         {0, 0, 2, 0},
                                         {0, 0, 1, 2},
                                         {0, 0, 0, 0}},

                                        {{2, 1, 0, 0}, // img 1 chan 1
                                         {0, 2, 0, 0},
                                         {1, 1, 2, 0},
                                         {1, 0, 0, 0}}}})
                  .get_vector());
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call_with_validate(f, {result}, {a});
    EXPECT_EQ((test::NDArray<float, 4>({{{{3, 2}, // img 0 chan 0
                                          {2, 1}},

                                         {{3, 3}, // img 0 chan 1
                                          {2, 1}}},

                                        {{{2, 2}, // img 1 chan 0
                                          {1, 2}},

                                         {{2, 2}, // img 1 chan 1
                                          {2, 2}}}})
                   .get_vector()),
              read_vector<float>(result));
}

// MaxPool2D1ChannelTests test fixture for test setup reuse
class MaxPool2D1ChannelTests : public testing::Test
{
public:
    Shape shape_a{1, 1, 5, 5};
    Shape window_shape{2, 3};
    Strides window_movement_strides{1, 1};

protected:
    virtual void SetUp() override {}
};

NGRAPH_TEST_F(${BACKEND_NAME}, MaxPool2D1ChannelTests, max_pool_2d_1channel_1image_overpadded)
{
    Shape padding_below{2, 0};
    Shape padding_above{1, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{1, 1, 7, 5};
    auto f = make_shared<Function>(
        make_shared<op::MaxPool>(
            A, window_shape, window_movement_strides, padding_below, padding_above),
        ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 4>({{{{0, 1, 0, 2, 1},
                                         {0, 3, 2, 0, 0},
                                         {2, 0, 0, 0, 1},
                                         {2, 0, 1, 1, 2},
                                         {0, 2, 1, 0, 0}}}})
                  .get_vector());
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call_with_validate(f, {result}, {a});
    auto min = std::numeric_limits<float>::lowest();
    EXPECT_TRUE(test::all_close(test::NDArray<float, 4>({{{{min, min, min, min, min},
                                                           {1, 2, 2, 2, 1},
                                                           {3, 3, 2, 2, 1},
                                                           {3, 3, 2, 1, 1},
                                                           {2, 1, 2, 2, 2},
                                                           {2, 2, 2, 2, 2},
                                                           {2, 2, 1, 0, 0}}}})
                                    .get_vector(),
                                read_vector<float>(result)));
}

NGRAPH_TEST_F(${BACKEND_NAME}, MaxPool2D1ChannelTests, max_pool_2d_1channel_1image_padded)
{
    Shape padding_below{1, 0};
    Shape padding_above{1, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{1, 1, 6, 5};
    auto f = make_shared<Function>(
        make_shared<op::MaxPool>(
            A, window_shape, window_movement_strides, padding_below, padding_above),
        ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 4>({{{{0, 1, 0, 2, 1},
                                         {0, 3, 2, 0, 0},
                                         {2, 0, 0, 0, 1},
                                         {2, 0, 1, 1, 2},
                                         {0, 2, 1, 0, 0}}}})
                  .get_vector());
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call_with_validate(f, {result}, {a});
    EXPECT_EQ((test::NDArray<float, 4>({{{{1, 2, 2, 2, 1},
                                          {3, 3, 2, 2, 1},
                                          {3, 3, 2, 1, 1},
                                          {2, 1, 2, 2, 2},
                                          {2, 2, 2, 2, 2},
                                          {2, 2, 1, 0, 0}}}})
                   .get_vector()),
              read_vector<float>(result));
}

// Test to make sure that negative elements and padding are handled properly. Added this because
// mkldnn calls its padding "zero padding" but apparently that is not technically true (negative
// values still "win" versus out-of-bounds values), which is good.
NGRAPH_TEST(${BACKEND_NAME}, max_pool_2d_1channel_1image_padded_negative_values)
{
    auto shape_a = Shape{
        1,
        1,
        1,
        14}; // 1 image, 1 channel, 1 row, 14 columns (if it's 1D we don't get mkldnn as of this writing)
    Shape window_shape{1, 3};
    auto window_movement_strides = Strides{1, 1};
    Shape padding_below{0, 1};
    Shape padding_above{0, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{1, 1, 1, 15};
    auto f = make_shared<Function>(
        make_shared<op::MaxPool>(
            A, window_shape, window_movement_strides, padding_below, padding_above),
        ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 4>{{{{-1, -2, -3, -3, -2, -1, -3, -2, -2, -2, -2, -3, -4, -5}}}}
                  .get_vector());
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call_with_validate(f, {result}, {a});
    EXPECT_EQ(
        (test::NDArray<float, 4>({{{{-1, -1, -2, -2, -1, -1, -1, -2, -2, -2, -2, -2, -3, -4, -5}}}})
             .get_vector()),
        read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, max_pool_2d_1channel_1image_strided)
{
    Shape shape_a{1, 1, 8, 8};
    Shape window_shape{2, 3};
    auto window_movement_strides = Strides{3, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{1, 1, 3, 3};
    auto f = make_shared<Function>(
        make_shared<op::MaxPool>(A, window_shape, window_movement_strides), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 4>({{{{0, 1, 0, 2, 1, 2, 0, 0},
                                         {0, 3, 2, 0, 0, 0, 1, 0},
                                         {2, 0, 0, 0, 1, 0, 0, 0},
                                         {2, 0, 1, 1, 2, 2, 3, 0},
                                         {0, 2, 1, 0, 0, 0, 1, 0},
                                         {2, 0, 3, 1, 0, 0, 0, 0},
                                         {1, 2, 0, 0, 0, 1, 2, 0},
                                         {1, 0, 2, 0, 0, 0, 1, 0}}}})
                  .get_vector());
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call_with_validate(f, {result}, {a});
    EXPECT_EQ((test::NDArray<float, 4>({{{{3, 2, 2}, {2, 2, 3}, {2, 2, 2}}}}).get_vector()),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, max_pool_3d)
{
    Shape shape_a{64, 3, 7, 8, 10};
    Shape window_shape{2, 3, 2};
    auto move_strides = Strides{2, 3, 4};
    Shape padding_below{5, 6, 4};
    Shape padding_above{6, 4, 5};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto B = make_shared<op::Parameter>(element::f32, shape_a);

    auto cpu_f = make_shared<Function>(
        make_shared<op::MaxPool>(A, window_shape, move_strides, padding_below, padding_above),
        ParameterVector{A});
    auto int_f = make_shared<Function>(
        make_shared<op::MaxPool>(B, window_shape, move_strides, padding_below, padding_above),
        ParameterVector{B});
    test::Uniform<float> rng(0.0f, 1.0f);
    vector<vector<float>> args;

    for (shared_ptr<op::Parameter> param : int_f->get_parameters())
    {
        vector<float> tensor_val(shape_size(param->get_shape()));
        rng.initialize(tensor_val);
        args.push_back(tensor_val);
    }
    auto int_results = execute(int_f, args, "INTERPRETER");
    auto cpu_results = execute(cpu_f, args, "${BACKEND_NAME}");
    for (size_t i = 0; i < cpu_results.size(); i++)
    {
        EXPECT_TRUE(test::all_close(cpu_results.at(i), int_results.at(i), 1.0e-4f, 1.0e-4f));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, avg_pool_1d_1channel_1image)
{
    Shape shape_a{1, 1, 14};
    Shape window_shape{3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{1, 1, 12};
    auto f = make_shared<Function>(make_shared<op::AvgPool>(A, window_shape), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 3>{{{0, 1, 0, 2, 1, 0, 3, 2, 0, 0, 2, 0, 0, 0}}}.get_vector());
    auto result = backend->create_tensor(element::f32, shape_r);

    float denom = 3.0;

    backend->call_with_validate(f, {result}, {a});
    EXPECT_TRUE(test::all_close_f(test::NDArray<float, 3>({{{1 / denom,
                                                             3 / denom,
                                                             3 / denom,
                                                             3 / denom,
                                                             4 / denom,
                                                             5 / denom,
                                                             5 / denom,
                                                             2 / denom,
                                                             2 / denom,
                                                             2 / denom,
                                                             2 / denom,
                                                             0 / denom}}})
                                      .get_vector(),
                                  read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, avg_pool_1d_1channel_2image)
{
    Shape shape_a{2, 1, 14};
    Shape window_shape{3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 1, 12};
    auto f = make_shared<Function>(make_shared<op::AvgPool>(A, window_shape), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 3>({{{0, 1, 0, 2, 1, 0, 3, 2, 0, 0, 2, 0, 0, 0}},
                                       {{0, 2, 1, 1, 0, 0, 0, 2, 0, 1, 0, 0, 1, 2}}})
                  .get_vector());
    auto result = backend->create_tensor(element::f32, shape_r);

    float denom = 3.0;

    backend->call_with_validate(f, {result}, {a});
    EXPECT_TRUE(test::all_close_f(test::NDArray<float, 3>({{{1 / denom,
                                                             3 / denom,
                                                             3 / denom,
                                                             3 / denom,
                                                             4 / denom,
                                                             5 / denom,
                                                             5 / denom,
                                                             2 / denom,
                                                             2 / denom,
                                                             2 / denom,
                                                             2 / denom,
                                                             0 / denom}},
                                                           {{3 / denom,
                                                             4 / denom,
                                                             2 / denom,
                                                             1 / denom,
                                                             0 / denom,
                                                             2 / denom,
                                                             2 / denom,
                                                             3 / denom,
                                                             1 / denom,
                                                             1 / denom,
                                                             1 / denom,
                                                             3 / denom}}})
                                      .get_vector(),
                                  read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, avg_pool_1d_2channel_2image)
{
    Shape shape_a{2, 2, 14};
    Shape window_shape{3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 2, 12};
    auto f = make_shared<Function>(make_shared<op::AvgPool>(A, window_shape), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 3>({{{0, 1, 0, 2, 1, 0, 3, 2, 0, 0, 2, 0, 0, 0},
                                        {0, 0, 0, 2, 0, 0, 2, 3, 0, 1, 2, 0, 1, 0}},

                                       {{0, 2, 1, 1, 0, 0, 0, 2, 0, 1, 0, 0, 1, 2},
                                        {2, 1, 0, 0, 1, 0, 2, 0, 0, 0, 1, 1, 2, 0}}})
                  .get_vector());
    auto result = backend->create_tensor(element::f32, shape_r);

    float denom = 3.0;

    backend->call_with_validate(f, {result}, {a});
    EXPECT_TRUE(test::all_close_f(test::NDArray<float, 3>({{{1 / denom,
                                                             3 / denom,
                                                             3 / denom,
                                                             3 / denom,
                                                             4 / denom,
                                                             5 / denom,
                                                             5 / denom,
                                                             2 / denom,
                                                             2 / denom,
                                                             2 / denom,
                                                             2 / denom,
                                                             0 / denom},
                                                            {0 / denom,
                                                             2 / denom,
                                                             2 / denom,
                                                             2 / denom,
                                                             2 / denom,
                                                             5 / denom,
                                                             5 / denom,
                                                             4 / denom,
                                                             3 / denom,
                                                             3 / denom,
                                                             3 / denom,
                                                             1 / denom}},

                                                           {{3 / denom,
                                                             4 / denom,
                                                             2 / denom,
                                                             1 / denom,
                                                             0 / denom,
                                                             2 / denom,
                                                             2 / denom,
                                                             3 / denom,
                                                             1 / denom,
                                                             1 / denom,
                                                             1 / denom,
                                                             3 / denom},
                                                            {3 / denom,
                                                             1 / denom,
                                                             1 / denom,
                                                             1 / denom,
                                                             3 / denom,
                                                             2 / denom,
                                                             2 / denom,
                                                             0 / denom,
                                                             1 / denom,
                                                             2 / denom,
                                                             4 / denom,
                                                             3 / denom}}})
                                      .get_vector(),
                                  read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, avg_pool_2d_2channel_2image)
{
    Shape shape_a{2, 2, 5, 5};
    Shape window_shape{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 2, 4, 3};
    auto f = make_shared<Function>(make_shared<op::AvgPool>(A, window_shape), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 4>({{{{0, 1, 0, 2, 1}, // img 0 chan 0
                                         {0, 3, 2, 0, 0},
                                         {2, 0, 0, 0, 1},
                                         {2, 0, 1, 1, 2},
                                         {0, 2, 1, 0, 0}},

                                        {{0, 0, 0, 2, 0}, // img 0 chan 1
                                         {0, 2, 3, 0, 1},
                                         {2, 0, 1, 0, 2},
                                         {3, 1, 0, 0, 0},
                                         {2, 0, 0, 0, 0}}},

                                       {{{0, 2, 1, 1, 0}, // img 1 chan 0
                                         {0, 0, 2, 0, 1},
                                         {0, 0, 1, 2, 3},
                                         {2, 0, 0, 3, 0},
                                         {0, 0, 0, 0, 0}},

                                        {{2, 1, 0, 0, 1}, // img 1 chan 1
                                         {0, 2, 0, 0, 0},
                                         {1, 1, 2, 0, 2},
                                         {1, 1, 1, 0, 1},
                                         {1, 0, 0, 0, 2}}}})
                  .get_vector());
    auto result = backend->create_tensor(element::f32, shape_r);

    float denom = 2 * 3;

    backend->call_with_validate(f, {result}, {a});

    EXPECT_TRUE(test::all_close_f(
        test::NDArray<float, 4>({{{{6 / denom, 8 / denom, 5 / denom}, // img 0 chan 0
                                   {7 / denom, 5 / denom, 3 / denom},
                                   {5 / denom, 2 / denom, 5 / denom},
                                   {6 / denom, 5 / denom, 5 / denom}},

                                  {{5 / denom, 7 / denom, 6 / denom}, // img 0 chan 1
                                   {8 / denom, 6 / denom, 7 / denom},
                                   {7 / denom, 2 / denom, 3 / denom},
                                   {6 / denom, 1 / denom, 0 / denom}}},

                                 {{{5 / denom, 6 / denom, 5 / denom}, // img 1 chan 0
                                   {3 / denom, 5 / denom, 9 / denom},
                                   {3 / denom, 6 / denom, 9 / denom},
                                   {2 / denom, 3 / denom, 3 / denom}},

                                  {{5 / denom, 3 / denom, 1 / denom}, // img 1 chan 1
                                   {6 / denom, 5 / denom, 4 / denom},
                                   {7 / denom, 5 / denom, 6 / denom},
                                   {4 / denom, 2 / denom, 4 / denom}}}})
            .get_vector(),
        read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, avg_pool_2d_1channel_1image_strided)
{
    Shape shape_a{1, 1, 8, 8};
    Shape window_shape{2, 3};
    auto window_movement_strides = Strides{3, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{1, 1, 3, 3};
    auto f = make_shared<Function>(
        make_shared<op::AvgPool>(A, window_shape, window_movement_strides), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 4>({{{{0, 1, 0, 2, 1, 2, 0, 0},
                                         {0, 3, 2, 0, 0, 0, 1, 0},
                                         {2, 0, 0, 0, 1, 0, 0, 0},
                                         {2, 0, 1, 1, 2, 2, 3, 0},
                                         {0, 2, 1, 0, 0, 0, 1, 0},
                                         {2, 0, 3, 1, 0, 0, 0, 0},
                                         {1, 2, 0, 0, 0, 1, 2, 0},
                                         {1, 0, 2, 0, 0, 0, 1, 0}}}})
                  .get_vector());
    auto result = backend->create_tensor(element::f32, shape_r);

    float denom = 2 * 3;

    backend->call_with_validate(f, {result}, {a});
    EXPECT_TRUE(test::all_close_f(test::NDArray<float, 4>({{{{6 / denom, 5 / denom, 4 / denom},
                                                             {6 / denom, 5 / denom, 8 / denom},
                                                             {6 / denom, 2 / denom, 4 / denom}}}})
                                      .get_vector(),
                                  read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, avg_pool_2d_1channel_1image_padded_do_not_include_in_computation)
{
    Shape shape_a{1, 1, 3, 3};
    Shape window_shape{2, 2};
    auto window_movement_strides = Strides{1, 1};
    Shape padding_below{1, 1};
    Shape padding_above{1, 1};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{1, 1, 4, 4};
    auto f = make_shared<Function>(
        make_shared<op::AvgPool>(
            A, window_shape, window_movement_strides, padding_below, padding_above, false),
        ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, test::NDArray<float, 4>({{{{0, 1, 0}, {0, 3, 2}, {2, 0, 0}}}}).get_vector());
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call_with_validate(f, {result}, {a});
    EXPECT_TRUE(
        test::all_close(test::NDArray<float, 4>({{{{0.0f / 1, 1.0f / 2, 1.0f / 2, 0.0f / 1},
                                                   {0.0f / 2, 4.0f / 4, 6.0f / 4, 2.0f / 2},
                                                   {2.0f / 2, 5.0f / 4, 5.0f / 4, 2.0f / 2},
                                                   {2.0f / 1, 2.0f / 2, 0.0f / 2, 0.0f / 1}}}})
                            .get_vector(),
                        read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, avg_pool_2d_1channel_1image_padded_include_in_computation)
{
    Shape shape_a{1, 1, 3, 3};
    Shape window_shape{2, 2};
    auto window_movement_strides = Strides{1, 1};
    Shape padding_below{1, 1};
    Shape padding_above{1, 1};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{1, 1, 4, 4};
    auto f = make_shared<Function>(
        make_shared<op::AvgPool>(
            A, window_shape, window_movement_strides, padding_below, padding_above, true),
        ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, test::NDArray<float, 4>({{{{0, 1, 0}, {0, 3, 2}, {2, 0, 0}}}}).get_vector());
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call_with_validate(f, {result}, {a});
    EXPECT_TRUE(
        test::all_close(test::NDArray<float, 4>({{{{0.0f / 4, 1.0f / 4, 1.0f / 4, 0.0f / 4},
                                                   {0.0f / 4, 4.0f / 4, 6.0f / 4, 2.0f / 4},
                                                   {2.0f / 4, 5.0f / 4, 5.0f / 4, 2.0f / 4},
                                                   {2.0f / 4, 2.0f / 4, 0.0f / 4, 0.0f / 4}}}})
                            .get_vector(),
                        read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, avg_pool_2d_2channel_2image_padded_do_not_include_in_computation)
{
    Shape shape_a{2, 1, 3, 3};
    Shape window_shape{2, 2};
    auto window_movement_strides = Strides{1, 1};
    Shape padding_below{1, 1};
    Shape padding_above{1, 1};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 1, 4, 4};
    auto f = make_shared<Function>(
        make_shared<op::AvgPool>(
            A, window_shape, window_movement_strides, padding_below, padding_above, false),
        ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 4>(
                  {{{{0, 1, 0}, {0, 3, 2}, {2, 0, 0}}, {{3, 5, 2}, {2, 0, 9}, {3, 6, 5}}}})
                  .get_vector());
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call_with_validate(f, {result}, {a});
    EXPECT_TRUE(
        test::all_close(test::NDArray<float, 4>({{{{0.0f / 1, 1.0f / 2, 1.0f / 2, 0.0f / 1},
                                                   {0.0f / 2, 4.0f / 4, 6.0f / 4, 2.0f / 2},
                                                   {2.0f / 2, 5.0f / 4, 5.0f / 4, 2.0f / 2},
                                                   {2.0f / 1, 2.0f / 2, 0.0f / 2, 0.0f / 1}},
                                                  {{3.0f / 1, 8.0f / 2, 7.0f / 2, 2.0f / 1},
                                                   {5.0f / 2, 10.0f / 4, 16.0f / 4, 11.0f / 2},
                                                   {5.0f / 2, 11.0f / 4, 20.0f / 4, 14.0f / 2},
                                                   {3.0f / 1, 9.0f / 2, 11.0f / 2, 5.0f / 1}}}})
                            .get_vector(),
                        read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, avg_pool_2d_2channel_2image_padded_include_in_computation)
{
    Shape shape_a{2, 1, 3, 3};
    Shape window_shape{2, 2};
    auto window_movement_strides = Strides{1, 1};
    Shape padding_below{1, 1};
    Shape padding_above{1, 1};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 1, 4, 4};
    auto f = make_shared<Function>(
        make_shared<op::AvgPool>(
            A, window_shape, window_movement_strides, padding_below, padding_above, true),
        ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 4>(
                  {{{{0, 1, 0}, {0, 3, 2}, {2, 0, 0}}, {{3, 5, 2}, {2, 0, 9}, {3, 6, 5}}}})
                  .get_vector());
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call_with_validate(f, {result}, {a});
    EXPECT_TRUE(
        test::all_close(test::NDArray<float, 4>({{{{0.0f / 4, 1.0f / 4, 1.0f / 4, 0.0f / 4},
                                                   {0.0f / 4, 4.0f / 4, 6.0f / 4, 2.0f / 4},
                                                   {2.0f / 4, 5.0f / 4, 5.0f / 4, 2.0f / 4},
                                                   {2.0f / 4, 2.0f / 4, 0.0f / 4, 0.0f / 4}},
                                                  {{3.0f / 4, 8.0f / 4, 7.0f / 4, 2.0f / 4},
                                                   {5.0f / 4, 10.0f / 4, 16.0f / 4, 11.0f / 4},
                                                   {5.0f / 4, 11.0f / 4, 20.0f / 4, 14.0f / 4},
                                                   {3.0f / 4, 9.0f / 4, 11.0f / 4, 5.0f / 4}}}})
                            .get_vector(),
                        read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME},
            avg_pool_2d_2channel_2image_padded_only_below_do_not_include_in_computation)
{
    Shape shape_a{2, 1, 3, 3};
    Shape window_shape{2, 2};
    auto window_movement_strides = Strides{1, 1};
    Shape padding_below{1, 1};
    Shape padding_above{0, 0};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 1, 3, 3};
    auto f = make_shared<Function>(
        make_shared<op::AvgPool>(
            A, window_shape, window_movement_strides, padding_below, padding_above, false),
        ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 4>(
                  {{{{0, 1, 0}, {0, 3, 2}, {2, 0, 0}}, {{3, 5, 2}, {2, 0, 9}, {3, 6, 5}}}})
                  .get_vector());
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call_with_validate(f, {result}, {a});
    EXPECT_TRUE(test::all_close(test::NDArray<float, 4>({{{{0.0f / 1, 1.0f / 2, 1.0f / 2},
                                                           {0.0f / 2, 4.0f / 4, 6.0f / 4},
                                                           {2.0f / 2, 5.0f / 4, 5.0f / 4}},
                                                          {{3.0f / 1, 8.0f / 2, 7.0f / 2},
                                                           {5.0f / 2, 10.0f / 4, 16.0f / 4},
                                                           {5.0f / 2, 11.0f / 4, 20.0f / 4}}}})
                                    .get_vector(),
                                read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, avg_pool_2d_2channel_2image_padded_only_below_include_in_computation)
{
    Shape shape_a{2, 1, 3, 3};
    Shape window_shape{2, 2};
    auto window_movement_strides = Strides{1, 1};
    Shape padding_below{1, 1};
    Shape padding_above{0, 0};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 1, 3, 3};
    auto f = make_shared<Function>(
        make_shared<op::AvgPool>(
            A, window_shape, window_movement_strides, padding_below, padding_above, true),
        ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 4>(
                  {{{{0, 1, 0}, {0, 3, 2}, {2, 0, 0}}, {{3, 5, 2}, {2, 0, 9}, {3, 6, 5}}}})
                  .get_vector());
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call_with_validate(f, {result}, {a});
    EXPECT_TRUE(test::all_close(test::NDArray<float, 4>({{{{0.0f / 4, 1.0f / 4, 1.0f / 4},
                                                           {0.0f / 4, 4.0f / 4, 6.0f / 4},
                                                           {2.0f / 4, 5.0f / 4, 5.0f / 4}},
                                                          {{3.0f / 4, 8.0f / 4, 7.0f / 4},
                                                           {5.0f / 4, 10.0f / 4, 16.0f / 4},
                                                           {5.0f / 4, 11.0f / 4, 20.0f / 4}}}})
                                    .get_vector(),
                                read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME},
            avg_pool_2d_2channel_2image_padded_only_above_do_not_include_in_computation)
{
    Shape shape_a{2, 1, 3, 3};
    Shape window_shape{2, 2};
    auto window_movement_strides = Strides{1, 1};
    Shape padding_below{0, 0};
    Shape padding_above{1, 1};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 1, 3, 3};
    auto f = make_shared<Function>(
        make_shared<op::AvgPool>(
            A, window_shape, window_movement_strides, padding_below, padding_above, false),
        ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 4>(
                  {{{{0, 1, 0}, {0, 3, 2}, {2, 0, 0}}, {{3, 5, 2}, {2, 0, 9}, {3, 6, 5}}}})
                  .get_vector());
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call_with_validate(f, {result}, {a});
    EXPECT_TRUE(test::all_close(test::NDArray<float, 4>({{{{4.0f / 4, 6.0f / 4, 2.0f / 2},
                                                           {5.0f / 4, 5.0f / 4, 2.0f / 2},
                                                           {2.0f / 2, 0.0f / 2, 0.0f / 1}},
                                                          {{10.0f / 4, 16.0f / 4, 11.0f / 2},
                                                           {11.0f / 4, 20.0f / 4, 14.0f / 2},
                                                           {9.0f / 2, 11.0f / 2, 5.0f / 1}}}})
                                    .get_vector(),
                                read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, avg_pool_2d_2channel_2image_padded_only_above_include_in_computation)
{
    Shape shape_a{2, 1, 3, 3};
    Shape window_shape{2, 2};
    auto window_movement_strides = Strides{1, 1};
    Shape padding_below{0, 0};
    Shape padding_above{1, 1};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 1, 3, 3};
    auto f = make_shared<Function>(
        make_shared<op::AvgPool>(
            A, window_shape, window_movement_strides, padding_below, padding_above, true),
        ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 4>(
                  {{{{0, 1, 0}, {0, 3, 2}, {2, 0, 0}}, {{3, 5, 2}, {2, 0, 9}, {3, 6, 5}}}})
                  .get_vector());
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call_with_validate(f, {result}, {a});
    EXPECT_TRUE(test::all_close(test::NDArray<float, 4>({{{{4.0f / 4, 6.0f / 4, 2.0f / 4},
                                                           {5.0f / 4, 5.0f / 4, 2.0f / 4},
                                                           {2.0f / 4, 0.0f / 4, 0.0f / 4}},
                                                          {{10.0f / 4, 16.0f / 4, 11.0f / 4},
                                                           {11.0f / 4, 20.0f / 4, 14.0f / 4},
                                                           {9.0f / 4, 11.0f / 4, 5.0f / 4}}}})
                                    .get_vector(),
                                read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, avg_pool_2d_2channel_2image_3x3_padded_do_not_include_in_computation)
{
    Shape shape_a{2, 1, 3, 3};
    Shape window_shape{3, 3};
    auto window_movement_strides = Strides{1, 1};
    Shape padding_below{2, 2};
    Shape padding_above{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 1, 5, 5};
    auto f = make_shared<Function>(
        make_shared<op::AvgPool>(
            A, window_shape, window_movement_strides, padding_below, padding_above, false),
        ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 4>(
                  {{{{0, 1, 0}, {0, 3, 2}, {2, 0, 0}}, {{3, 5, 2}, {2, 0, 9}, {3, 6, 5}}}})
                  .get_vector());
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call_with_validate(f, {result}, {a});
    EXPECT_TRUE(test::all_close_f(
        test::NDArray<float, 4>({{{{0.0f / 1, 1.0f / 2, 1.0f / 3, 1.0f / 2, 0.0f / 1},
                                   {0.0f / 2, 4.0f / 4, 6.0f / 6, 6.0f / 4, 2.0f / 2},
                                   {2.0f / 3, 6.0f / 6, 8.0f / 9, 6.0f / 6, 2.0f / 3},
                                   {2.0f / 2, 5.0f / 4, 7.0f / 6, 5.0f / 4, 2.0f / 2},
                                   {2.0f / 1, 2.0f / 2, 2.0f / 3, 0.0f / 2, 0.0f / 1}},
                                  {{3.0f / 1, 8.0f / 2, 10.0f / 3, 7.0f / 2, 2.0f / 1},
                                   {5.0f / 2, 10.0f / 4, 21.0f / 6, 16.0f / 4, 11.0f / 2},
                                   {8.0f / 3, 19.0f / 6, 35.0f / 9, 27.0f / 6, 16.0f / 3},
                                   {5.0f / 2, 11.0f / 4, 25.0f / 6, 20.0f / 4, 14.0f / 2},
                                   {3.0f / 1, 9.0f / 2, 14.0f / 3, 11.0f / 2, 5.0f / 1}}}})
            .get_vector(),
        read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, avg_pool_2d_2channel_2image_3x3_padded_include_in_computation)
{
    Shape shape_a{2, 1, 3, 3};
    Shape window_shape{3, 3};
    auto window_movement_strides = Strides{1, 1};
    Shape padding_below{2, 2};
    Shape padding_above{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 1, 5, 5};
    auto f = make_shared<Function>(
        make_shared<op::AvgPool>(
            A, window_shape, window_movement_strides, padding_below, padding_above, true),
        ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 4>(
                  {{{{0, 1, 0}, {0, 3, 2}, {2, 0, 0}}, {{3, 5, 2}, {2, 0, 9}, {3, 6, 5}}}})
                  .get_vector());
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call_with_validate(f, {result}, {a});
    EXPECT_TRUE(test::all_close_f(
        test::NDArray<float, 4>({{{{0.0f / 9, 1.0f / 9, 1.0f / 9, 1.0f / 9, 0.0f / 9},
                                   {0.0f / 9, 4.0f / 9, 6.0f / 9, 6.0f / 9, 2.0f / 9},
                                   {2.0f / 9, 6.0f / 9, 8.0f / 9, 6.0f / 9, 2.0f / 9},
                                   {2.0f / 9, 5.0f / 9, 7.0f / 9, 5.0f / 9, 2.0f / 9},
                                   {2.0f / 9, 2.0f / 9, 2.0f / 9, 0.0f / 9, 0.0f / 9}},
                                  {{3.0f / 9, 8.0f / 9, 10.0f / 9, 7.0f / 9, 2.0f / 9},
                                   {5.0f / 9, 10.0f / 9, 21.0f / 9, 16.0f / 9, 11.0f / 9},
                                   {8.0f / 9, 19.0f / 9, 35.0f / 9, 27.0f / 9, 16.0f / 9},
                                   {5.0f / 9, 11.0f / 9, 25.0f / 9, 20.0f / 9, 14.0f / 9},
                                   {3.0f / 9, 9.0f / 9, 14.0f / 9, 11.0f / 9, 5.0f / 9}}}})
            .get_vector(),
        read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME},
            avg_pool_2d_2channel_2image_3x3_strided_padded_do_not_include_in_computation)
{
    Shape shape_a{2, 1, 3, 3};
    Shape window_shape{3, 3};
    auto window_movement_strides = Strides{2, 2};
    Shape padding_below{2, 2};
    Shape padding_above{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 1, 3, 3};
    auto f = make_shared<Function>(
        make_shared<op::AvgPool>(
            A, window_shape, window_movement_strides, padding_below, padding_above, false),
        ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 4>(
                  {{{{0, 1, 0}, {0, 3, 2}, {2, 0, 0}}, {{3, 5, 2}, {2, 0, 9}, {3, 6, 5}}}})
                  .get_vector());
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call_with_validate(f, {result}, {a});
    EXPECT_TRUE(test::all_close_f(test::NDArray<float, 4>({{{{0.0f / 1, 1.0f / 3, 0.0f / 1},
                                                             {2.0f / 3, 8.0f / 9, 2.0f / 3},
                                                             {2.0f / 1, 2.0f / 3, 0.0f / 1}},
                                                            {{3.0f / 1, 10.0f / 3, 2.0f / 1},
                                                             {8.0f / 3, 35.0f / 9, 16.0f / 3},
                                                             {3.0f / 1, 14.0f / 3, 5.0f / 1}}}})
                                      .get_vector(),
                                  read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, avg_pool_2d_2channel_2image_3x3_strided_padded_include_in_computation)
{
    Shape shape_a{2, 1, 3, 3};
    Shape window_shape{3, 3};
    auto window_movement_strides = Strides{2, 2};
    Shape padding_below{2, 2};
    Shape padding_above{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 1, 3, 3};
    auto f = make_shared<Function>(
        make_shared<op::AvgPool>(
            A, window_shape, window_movement_strides, padding_below, padding_above, true),
        ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 4>(
                  {{{{0, 1, 0}, {0, 3, 2}, {2, 0, 0}}, {{3, 5, 2}, {2, 0, 9}, {3, 6, 5}}}})
                  .get_vector());
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call_with_validate(f, {result}, {a});
    EXPECT_TRUE(test::all_close_f(test::NDArray<float, 4>({{{{0.0f / 9, 1.0f / 9, 0.0f / 9},
                                                             {2.0f / 9, 8.0f / 9, 2.0f / 9},
                                                             {2.0f / 9, 2.0f / 9, 0.0f / 9}},
                                                            {{3.0f / 9, 10.0f / 9, 2.0f / 9},
                                                             {8.0f / 9, 35.0f / 9, 16.0f / 9},
                                                             {3.0f / 9, 14.0f / 9, 5.0f / 9}}}})
                                      .get_vector(),
                                  read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME},
            avg_pool_2d_2channel_2image_3x3_strided_uneven_padded_do_not_include_in_computation)
{
    Shape shape_a{2, 1, 3, 3};
    Shape window_shape{3, 3};
    auto window_movement_strides = Strides{2, 3};
    Shape padding_below{2, 2};
    Shape padding_above{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 1, 3, 2};
    auto f = make_shared<Function>(
        make_shared<op::AvgPool>(
            A, window_shape, window_movement_strides, padding_below, padding_above, false),
        ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 4>(
                  {{{{0, 1, 0}, {0, 3, 2}, {2, 0, 0}}, {{3, 5, 2}, {2, 0, 9}, {3, 6, 5}}}})
                  .get_vector());
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call_with_validate(f, {result}, {a});
    EXPECT_TRUE(test::all_close_f(
        test::NDArray<float, 4>(
            {{{{0.0f / 1, 1.0f / 2}, {2.0f / 3, 6.0f / 6}, {2.0f / 1, 0.0f / 2}},
              {{3.0f / 1, 7.0f / 2}, {8.0f / 3, 27.0f / 6}, {3.0f / 1, 11.0f / 2}}}})
            .get_vector(),
        read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME},
            avg_pool_2d_2channel_2image_3x3_strided_uneven_padded_include_in_computation)
{
    Shape shape_a{2, 1, 3, 3};
    Shape window_shape{3, 3};
    auto window_movement_strides = Strides{2, 3};
    Shape padding_below{2, 2};
    Shape padding_above{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 1, 3, 2};
    auto f = make_shared<Function>(
        make_shared<op::AvgPool>(
            A, window_shape, window_movement_strides, padding_below, padding_above, true),
        ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 4>(
                  {{{{0, 1, 0}, {0, 3, 2}, {2, 0, 0}}, {{3, 5, 2}, {2, 0, 9}, {3, 6, 5}}}})
                  .get_vector());
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call_with_validate(f, {result}, {a});
    EXPECT_TRUE(test::all_close_f(
        test::NDArray<float, 4>(
            {{{{0.0f / 9, 1.0f / 9}, {2.0f / 9, 6.0f / 9}, {2.0f / 9, 0.0f / 9}},
              {{3.0f / 9, 7.0f / 9}, {8.0f / 9, 27.0f / 9}, {3.0f / 9, 11.0f / 9}}}})
            .get_vector(),
        read_vector<float>(result)));
}

// Params to drive avg_pool_3d testing variations
class avg_pool_3d_params : public ::testing::TestWithParam<bool>
{
protected:
    avg_pool_3d_params() { include_pad = GetParam(); }
    bool include_pad;
};

// avg_pool_3d test code using params
NGRAPH_TEST_P(${BACKEND_NAME}, avg_pool_3d_params, avg_pool_3d_uneven_strided_padded)
{
    Shape shape_a{64, 3, 12, 13, 15};
    Shape window_shape{4, 5, 4};
    auto move_strides = Strides{2, 3, 4};
    Shape padding_below{2, 3, 1};
    Shape padding_above{3, 1, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto B = make_shared<op::Parameter>(element::f32, shape_a);

    auto cpu_f = make_shared<Function>(
        make_shared<op::AvgPool>(
            A, window_shape, move_strides, padding_below, padding_above, include_pad),
        ParameterVector{A});
    auto int_f = make_shared<Function>(
        make_shared<op::AvgPool>(
            B, window_shape, move_strides, padding_below, padding_above, include_pad),
        ParameterVector{B});
    test::Uniform<float> rng(0.0f, 1.0f);
    vector<vector<float>> args;

    for (shared_ptr<op::Parameter> param : int_f->get_parameters())
    {
        vector<float> tensor_val(shape_size(param->get_shape()));
        rng.initialize(tensor_val);
        args.push_back(tensor_val);
    }
    auto int_results = execute(int_f, args, "INTERPRETER");
    auto backend_results = execute(cpu_f, args, "${BACKEND_NAME}");
    for (size_t i = 0; i < backend_results.size(); i++)
    {
        EXPECT_TRUE(test::all_close(backend_results.at(i), int_results.at(i), 1.0e-4f, 1.0e-4f));
    }
}

// avg_pool_3d case generation
NGRAPH_INSTANTIATE_TEST_CASE_P(${BACKEND_NAME}, include_pad, avg_pool_3d_params, testing::Bool());

//
// This test primarly checks that maxpool backprop functions
// correctly when the input tensor is larger than most cache sizes.
// Here the to-be-pooled tensor is rank 2 with one non-trivial
// dimension:
//
// x : [[0, 1, 0, 1, 0, 1, ... , 0, 1]]  <--- input data
//       ----  ----  ----  ...   ----    <--- pooling windows
// y : [[ 1  ,  1  ,  1  , ... ,  1]]    <--- max pooled output
//
// The pooling window is size 2 and stride 2, so the windows
// do not overlap. Thus, each window will effectively see [0, 1]
// as its input data for max pooling. The resulting output tensor
// of pooling will be sizeof(x) with all elements equal to 1 as
// seen above.
// Therefore, for the backward pooling operation with the same window shape
// and strides, the value of dy will only propogate to the positions in
// dx that correspond to a value of 1 in the corresponding input tensor x:
//
// dy : [[2, 3, ... , 4]]
// x  : [[0, 1, 0, 1, ... , 0, 1]]
// dx : [[0, 2, 0, 3, ... , 0, 4]]
//
NGRAPH_TEST(${BACKEND_NAME}, maxpool_bprop_larger_than_cache)
{
    Shape window_shape{1, 2};
    Strides move_strides{1, 2};
    Shape padding_below{0, 0};
    Shape padding_above{0, 0};

    // 200 MB tensor to exceed cache
    const size_t num_elements = 50 * 1024 * 1024;
    auto ceil_div = [](size_t x, size_t y) { return 1 + ((x - 1) / y); };
    const size_t num_pooled_elements = ceil_div(num_elements + padding_below.back() +
                                                    padding_above.back() - window_shape.back() + 1,
                                                move_strides.back());
    Shape shape_x{1, 1, 1, num_elements};
    Shape shape_y{1, 1, 1, num_pooled_elements};

    auto x = make_shared<op::Parameter>(element::f32, shape_x);
    auto dy = make_shared<op::Parameter>(element::f32, shape_y);
    auto bprop =
        make_shared<Function>(make_shared<op::MaxPoolBackprop>(
                                  x, dy, window_shape, move_strides, padding_below, padding_above),
                              ParameterVector{x, dy});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // initialize x to array of alternating 0s and 1s as described above
    std::vector<float> x_data(num_elements, 0);
    for (auto i = 0u; i < num_elements; i++)
    {
        x_data[i] = (i % 2);
    }
    auto x_t = backend->create_tensor(element::f32, shape_x);
    copy_data(x_t, x_data);

    // use random data for deltas dy
    std::vector<float> dy_data(num_pooled_elements);
    test::Uniform<float> rng(0.0f, 1.0f);
    rng.initialize(dy_data);
    auto dy_t = backend->create_tensor(element::f32, shape_y);
    copy_data(dy_t, dy_data);

    // create result deltas tensor and run the backward max pooling operation
    auto dx_t = backend->create_tensor(element::f32, shape_x);
    backend->call_with_validate(bprop, {dx_t}, {x_t, dy_t});

    // expected values should be dy with 0s left inserted
    // for each delta, see test description above for details
    std::vector<float> expected_dx(num_elements, 0);
    for (auto i = 0u, j = 0u; i < num_elements; i++)
    {
        if (x_data[i])
        {
            expected_dx[i] = x_data[i] * dy_data[j++];
        }
    }
    EXPECT_EQ(expected_dx, read_vector<float>(dx_t));
}
