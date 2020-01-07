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
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/known_element_types.hpp"
#include "util/ndarray.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, shape_of_scalar)
{
    Shape input_shape{};
    Shape output_shape{0};

    auto A = std::make_shared<op::Parameter>(element::f32, input_shape);
    auto f = std::make_shared<Function>(std::make_shared<op::ShapeOf>(A), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto a = backend->create_tensor(element::f32, input_shape);
    copy_data(a, vector<float>{0});
    auto result = backend->create_tensor(element::i64, output_shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    vector<int64_t> expected{};
    EXPECT_EQ(expected, read_vector<int64_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, shape_of_vector)
{
    Shape input_shape{2};
    Shape output_shape{1};

    auto A = std::make_shared<op::Parameter>(element::f32, input_shape);
    auto f = std::make_shared<Function>(std::make_shared<op::ShapeOf>(A), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto a = backend->create_tensor(element::f32, input_shape);
    copy_data(a, vector<float>(2, 0));
    auto result = backend->create_tensor(element::i64, output_shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    vector<int64_t> expected{2};
    EXPECT_EQ(expected, read_vector<int64_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, shape_of_matrix)
{
    Shape input_shape{2, 4};
    Shape output_shape{2};

    auto A = std::make_shared<op::Parameter>(element::f32, input_shape);
    auto f = std::make_shared<Function>(std::make_shared<op::ShapeOf>(A), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto a = backend->create_tensor(element::f32, input_shape);
    copy_data(a, vector<float>(2 * 4, 0));
    auto result = backend->create_tensor(element::i64, output_shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    vector<int64_t> expected{2, 4};
    EXPECT_EQ(expected, read_vector<int64_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, shape_of_5d)
{
    Shape input_shape{2, 4, 8, 16, 32};
    Shape output_shape{5};

    auto A = std::make_shared<op::Parameter>(element::f32, input_shape);
    auto f = std::make_shared<Function>(std::make_shared<op::ShapeOf>(A), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto a = backend->create_tensor(element::f32, input_shape);
    copy_data(a, vector<float>(2 * 4 * 8 * 16 * 32, 0));
    auto result = backend->create_tensor(element::i64, output_shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    vector<int64_t> expected{2, 4, 8, 16, 32};
    EXPECT_EQ(expected, read_vector<int64_t>(result));
}
