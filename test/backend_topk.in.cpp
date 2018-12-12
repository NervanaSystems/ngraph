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

NGRAPH_TEST(${BACKEND_NAME}, topk_1d_max_all)
{
    Shape shape{6};
    Shape rshape{6};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::TopK>(A, 0, element::i32, 0, true);
    auto f0 = make_shared<Function>(make_shared<op::GetOutputElement>(B, 0), ParameterVector{A});
    auto f1 = make_shared<Function>(make_shared<op::GetOutputElement>(B, 1), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto result0 = backend->create_tensor(element::i32, rshape);
    auto result1 = backend->create_tensor(element::f32, rshape);

    auto h0 = backend->compile(f0);
    backend->call_with_validate(h0, {result0}, {a});
    EXPECT_EQ((vector<int32_t>{5, 4, 3, 2, 1, 0}), read_vector<int32_t>(result0));
    auto h1 = backend->compile(f1);
    backend->call_with_validate(h1, {result1}, {a});
    EXPECT_EQ((vector<float>{6, 5, 4, 3, 2, 1}), read_vector<float>(result1));
}

NGRAPH_TEST(${BACKEND_NAME}, topk_1d_max_partial)
{
    Shape shape{6};
    Shape rshape{3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::TopK>(A, 0, element::i32, 3, true);
    auto f0 = make_shared<Function>(make_shared<op::GetOutputElement>(B, 0), ParameterVector{A});
    auto f1 = make_shared<Function>(make_shared<op::GetOutputElement>(B, 1), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto result0 = backend->create_tensor(element::i32, rshape);
    auto result1 = backend->create_tensor(element::f32, rshape);

    auto h0 = backend->compile(f0);
    backend->call_with_validate(h0, {result0}, {a});
    EXPECT_EQ((vector<int32_t>{5, 4, 3}), read_vector<int32_t>(result0));
    auto h1 = backend->compile(f1);
    backend->call_with_validate(h1, {result1}, {a});
    EXPECT_EQ((vector<float>{6, 5, 4}), read_vector<float>(result1));
}

NGRAPH_TEST(${BACKEND_NAME}, topk_1d_max_one)
{
    Shape shape{6};
    Shape rshape{1};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::TopK>(A, 0, element::i32, 1, true);
    auto f0 = make_shared<Function>(make_shared<op::GetOutputElement>(B, 0), ParameterVector{A});
    auto f1 = make_shared<Function>(make_shared<op::GetOutputElement>(B, 1), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto result0 = backend->create_tensor(element::i32, rshape);
    auto result1 = backend->create_tensor(element::f32, rshape);

    auto h0 = backend->compile(f0);
    backend->call_with_validate(h0, {result0}, {a});
    EXPECT_EQ((vector<int32_t>{5}), read_vector<int32_t>(result0));
    auto h1 = backend->compile(f1);
    backend->call_with_validate(h1, {result1}, {a});
    EXPECT_EQ((vector<float>{6}), read_vector<float>(result1));
}

NGRAPH_TEST(${BACKEND_NAME}, topk_1d_min_all)
{
    Shape shape{6};
    Shape rshape{6};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::TopK>(A, 0, element::i32, 0, false);
    auto f0 = make_shared<Function>(make_shared<op::GetOutputElement>(B, 0), ParameterVector{A});
    auto f1 = make_shared<Function>(make_shared<op::GetOutputElement>(B, 1), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{6, 5, 4, 3, 2, 1});
    auto result0 = backend->create_tensor(element::i32, rshape);
    auto result1 = backend->create_tensor(element::f32, rshape);

    auto h0 = backend->compile(f0);
    backend->call_with_validate(h0, {result0}, {a});
    EXPECT_EQ((vector<int32_t>{5, 4, 3, 2, 1, 0}), read_vector<int32_t>(result0));
    auto h1 = backend->compile(f1);
    backend->call_with_validate(h1, {result1}, {a});
    EXPECT_EQ((vector<float>{1, 2, 3, 4, 5, 6}), read_vector<float>(result1));
}

NGRAPH_TEST(${BACKEND_NAME}, topk_1d_min_partial)
{
    Shape shape{6};
    Shape rshape{3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::TopK>(A, 0, element::i32, 3, false);
    auto f0 = make_shared<Function>(make_shared<op::GetOutputElement>(B, 0), ParameterVector{A});
    auto f1 = make_shared<Function>(make_shared<op::GetOutputElement>(B, 1), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{6, 5, 4, 3, 2, 1});
    auto result0 = backend->create_tensor(element::i32, rshape);
    auto result1 = backend->create_tensor(element::f32, rshape);

    auto h0 = backend->compile(f0);
    backend->call_with_validate(h0, {result0}, {a});
    EXPECT_EQ((vector<int32_t>{5, 4, 3}), read_vector<int32_t>(result0));
    auto h1 = backend->compile(f1);
    backend->call_with_validate(h1, {result1}, {a});
    EXPECT_EQ((vector<float>{1, 2, 3}), read_vector<float>(result1));
}

NGRAPH_TEST(${BACKEND_NAME}, topk_1d_min_one)
{
    Shape shape{6};
    Shape rshape{1};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::TopK>(A, 0, element::i32, 1, false);
    auto f0 = make_shared<Function>(make_shared<op::GetOutputElement>(B, 0), ParameterVector{A});
    auto f1 = make_shared<Function>(make_shared<op::GetOutputElement>(B, 1), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{6, 5, 4, 3, 2, 1});
    auto result0 = backend->create_tensor(element::i32, rshape);
    auto result1 = backend->create_tensor(element::f32, rshape);

    auto h0 = backend->compile(f0);
    backend->call_with_validate(h0, {result0}, {a});
    EXPECT_EQ((vector<int32_t>{5}), read_vector<int32_t>(result0));
    auto h1 = backend->compile(f1);
    backend->call_with_validate(h1, {result1}, {a});
    EXPECT_EQ((vector<float>{1}), read_vector<float>(result1));
}

NGRAPH_TEST(${BACKEND_NAME}, topk_3d_max_all)
{
    Shape shape{2, 3, 2};
    Shape rshape{2, 3, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::TopK>(A, 1, element::i32, 0, true);
    auto f0 = make_shared<Function>(make_shared<op::GetOutputElement>(B, 0), ParameterVector{A});
    auto f1 = make_shared<Function>(make_shared<op::GetOutputElement>(B, 1), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{9, 2, 10, 12, 8, 4, 6, 1, 5, 3, 11, 7});
    auto result0 = backend->create_tensor(element::i32, rshape);
    auto result1 = backend->create_tensor(element::f32, rshape);

    auto h0 = backend->compile(f0);
    backend->call_with_validate(h0, {result0}, {a});
    EXPECT_EQ((vector<int32_t>{1, 1, 0, 2, 2, 0, 2, 2, 0, 1, 1, 0}), read_vector<int32_t>(result0));
    auto h1 = backend->compile(f1);
    backend->call_with_validate(h1, {result1}, {a});
    EXPECT_EQ((vector<float>{10, 12, 9, 4, 8, 2, 11, 7, 6, 3, 5, 1}), read_vector<float>(result1));
}

NGRAPH_TEST(${BACKEND_NAME}, topk_int64)
{
    Shape shape{2, 3, 2};
    Shape rshape{2, 3, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::TopK>(A, 1, element::i64, 0, true);
    auto f0 = make_shared<Function>(make_shared<op::GetOutputElement>(B, 0), ParameterVector{A});
    auto f1 = make_shared<Function>(make_shared<op::GetOutputElement>(B, 1), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{9, 2, 10, 12, 8, 4, 6, 1, 5, 3, 11, 7});
    auto result0 = backend->create_tensor(element::i64, rshape);
    auto result1 = backend->create_tensor(element::f32, rshape);

    auto h0 = backend->compile(f0);
    backend->call_with_validate(h0, {result0}, {a});
    EXPECT_EQ((vector<int64_t>{1, 1, 0, 2, 2, 0, 2, 2, 0, 1, 1, 0}), read_vector<int64_t>(result0));
    auto h1 = backend->compile(f1);
    backend->call_with_validate(h1, {result1}, {a});
    EXPECT_EQ((vector<float>{10, 12, 9, 4, 8, 2, 11, 7, 6, 3, 5, 1}), read_vector<float>(result1));
}

NGRAPH_TEST(${BACKEND_NAME}, topk_5d_max_partial)
{
    Shape shape{2, 6, 3, 2, 4};
    Shape rshape{2, 2, 3, 2, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::TopK>(A, 1, element::i32, 2, true);
    auto f0 = make_shared<Function>(make_shared<op::GetOutputElement>(B, 0), ParameterVector{A});
    auto f1 = make_shared<Function>(make_shared<op::GetOutputElement>(B, 1), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(
        a,
        vector<float>{
            1.,   73.,  9.,   81.,  17.,  89.,  2.,   74.,  10.,  82.,  18.,  90.,  3.,   75.,
            11.,  83.,  19.,  91.,  4.,   76.,  12.,  84.,  20.,  92.,  145., 217., 153., 225.,
            161., 233., 146., 218., 154., 226., 162., 234., 147., 219., 155., 227., 163., 235.,
            148., 220., 156., 228., 164., 236., 5.,   77.,  13.,  85.,  21.,  93.,  6.,   78.,
            14.,  86.,  22.,  94.,  7.,   79.,  15.,  87.,  23.,  95.,  8.,   80.,  16.,  88.,
            24.,  96.,  149., 221., 157., 229., 165., 27.,  150., 222., 158., 230., 166., 23.,
            151., 223., 159., 231., 17.,  39.,  2.,   224., 160., 232., 168., 240., 25.,  97.,
            33.,  105., 41.,  113., 26.,  98.,  34.,  106., 42.,  114., 27.,  99.,  35.,  107.,
            43.,  115., 28.,  100., 36.,  108., 44.,  116., 169., 241., 177., 249., 185., 25.,
            170., 242., 178., 250., 186., 258., 171., 243., 179., 251., 187., 259., 172., 24.,
            180., 252., 188., 260., 29.,  101., 37.,  109., 45.,  117., 30.,  102., 38.,  10.,
            46.,  118., 31.,  103., 39.,  111., 47.,  119., 32.,  104., 40.,  112., 48.,  20.,
            173., 245., 181., 253., 189., 261., 174., 246., 182., 254., 190., 262., 175., 27.,
            183., 255., 191., 263., 176., 248., 184., 256., 192., 264., 49.,  121., 57.,  129.,
            65.,  137., 50.,  122., 58.,  130., 66.,  138., 51.,  123., 59.,  131., 67.,  139.,
            52.,  124., 60.,  132., 68.,  140., 193., 265., 201., 273., 209., 281., 194., 266.,
            202., 274., 210., 43.,  115., 28.,  100., 36.,  108., 44.,  116., 169., 241., 177.,
            212., 284., 53.,  125., 61.,  133., 69.,  141., 54.,  126., 62.,  134., 70.,  142.,
            55.,  127., 63.,  135., 71.,  143., 56.,  128., 64.,  136., 72.,  144., 197., 269.,
            205., 277., 213., 285., 198., 270., 206., 278., 214., 286., 199., 271., 207., 279.,
            215., 287., 200., 272., 208., 280., 216., 288.});

    auto result0 = backend->create_tensor(element::i32, rshape);
    auto result1 = backend->create_tensor(element::f32, rshape);

    auto h0 = backend->compile(f0);
    backend->call_with_validate(h0, {result0}, {a});
    EXPECT_EQ(
        (vector<int32_t>{5, 5, 5, 5, 5, 1, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 3, 5, 5, 5, 5,
                         3, 3, 3, 3, 3, 4, 3, 3, 3, 3, 3, 1, 3, 3, 3, 3, 1, 1, 1, 1, 3, 3, 3, 3,
                         5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 3, 5, 5, 5,
                         3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 4, 1, 1, 1, 1, 1, 1, 5, 1, 3, 3}),
        read_vector<int32_t>(result0));

    auto h1 = backend->compile(f1);
    backend->call_with_validate(h1, {result1}, {a});
    EXPECT_EQ((vector<float>{169, 241, 177, 249, 185, 233, 170, 242, 178, 250, 186, 258, 171, 243,
                             179, 251, 187, 259, 172, 224, 180, 252, 188, 260, 149, 221, 157, 229,
                             165, 113, 150, 222, 158, 230, 166, 234, 151, 223, 159, 231, 163, 235,
                             148, 220, 160, 232, 168, 240, 197, 269, 205, 277, 213, 285, 198, 270,
                             206, 278, 214, 286, 199, 271, 207, 279, 215, 287, 200, 272, 241, 280,
                             216, 288, 193, 265, 201, 273, 209, 281, 194, 266, 202, 274, 210, 262,
                             175, 127, 183, 255, 191, 263, 176, 248, 208, 256, 212, 284}),
              read_vector<float>(result1));
}

NGRAPH_TEST(${BACKEND_NAME}, topk_3d_max_partial)
{
    Shape shape{2, 3, 2};
    Shape rshape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::TopK>(A, 1, element::i32, 2, true);
    auto f0 = make_shared<Function>(make_shared<op::GetOutputElement>(B, 0), ParameterVector{A});
    auto f1 = make_shared<Function>(make_shared<op::GetOutputElement>(B, 1), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{9, 2, 10, 12, 8, 4, 6, 1, 5, 3, 11, 7});
    auto result0 = backend->create_tensor(element::i32, rshape);
    auto result1 = backend->create_tensor(element::f32, rshape);

    auto h0 = backend->compile(f0);
    backend->call_with_validate(h0, {result0}, {a});
    EXPECT_EQ((vector<int32_t>{1, 1, 0, 2, 2, 2, 0, 1}), read_vector<int32_t>(result0));
    auto h1 = backend->compile(f1);
    backend->call_with_validate(h1, {result1}, {a});
    EXPECT_EQ((vector<float>{10, 12, 9, 4, 11, 7, 6, 3}), read_vector<float>(result1));
}

NGRAPH_TEST(${BACKEND_NAME}, topk_3d_max_one)
{
    Shape shape{2, 3, 2};
    Shape rshape{2, 1, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::TopK>(A, 1, element::i32, 1, true);
    auto f0 = make_shared<Function>(make_shared<op::GetOutputElement>(B, 0), ParameterVector{A});
    auto f1 = make_shared<Function>(make_shared<op::GetOutputElement>(B, 1), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{9, 2, 10, 12, 8, 4, 6, 1, 5, 3, 11, 7});
    auto result0 = backend->create_tensor(element::i32, rshape);
    auto result1 = backend->create_tensor(element::f32, rshape);

    auto h0 = backend->compile(f0);
    backend->call_with_validate(h0, {result0}, {a});
    EXPECT_EQ((vector<int32_t>{1, 1, 2, 2}), read_vector<int32_t>(result0));
    auto h1 = backend->compile(f1);
    backend->call_with_validate(h1, {result1}, {a});
    EXPECT_EQ((vector<float>{10, 12, 11, 7}), read_vector<float>(result1));
}

NGRAPH_TEST(${BACKEND_NAME}, topk_3d_min_all)
{
    Shape shape{2, 3, 2};
    Shape rshape{2, 3, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::TopK>(A, 1, element::i32, 0, false);
    auto f0 = make_shared<Function>(make_shared<op::GetOutputElement>(B, 0), ParameterVector{A});
    auto f1 = make_shared<Function>(make_shared<op::GetOutputElement>(B, 1), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{12, 2, 10, 9, 8, 4, 6, 1, 5, 3, 11, 7});
    auto result0 = backend->create_tensor(element::i32, rshape);
    auto result1 = backend->create_tensor(element::f32, rshape);

    auto h0 = backend->compile(f0);
    backend->call_with_validate(h0, {result0}, {a});
    EXPECT_EQ((vector<int32_t>{2, 0, 1, 2, 0, 1, 1, 0, 0, 1, 2, 2}), read_vector<int32_t>(result0));
    auto h1 = backend->compile(f1);
    backend->call_with_validate(h1, {result1}, {a});
    EXPECT_EQ((vector<float>{8, 2, 10, 4, 12, 9, 5, 1, 6, 3, 11, 7}), read_vector<float>(result1));
}

NGRAPH_TEST(${BACKEND_NAME}, topk_3d_min_partial)
{
    Shape shape{2, 3, 2};
    Shape rshape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::TopK>(A, 1, element::i32, 2, false);
    auto f0 = make_shared<Function>(make_shared<op::GetOutputElement>(B, 0), ParameterVector{A});
    auto f1 = make_shared<Function>(make_shared<op::GetOutputElement>(B, 1), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{12, 2, 10, 9, 8, 4, 6, 1, 5, 3, 11, 7});
    auto result0 = backend->create_tensor(element::i32, rshape);
    auto result1 = backend->create_tensor(element::f32, rshape);

    auto h0 = backend->compile(f0);
    backend->call_with_validate(h0, {result0}, {a});
    EXPECT_EQ((vector<int32_t>{2, 0, 1, 2, 1, 0, 0, 1}), read_vector<int32_t>(result0));
    auto h1 = backend->compile(f1);
    backend->call_with_validate(h1, {result1}, {a});
    EXPECT_EQ((vector<float>{8, 2, 10, 4, 5, 1, 6, 3}), read_vector<float>(result1));
}

NGRAPH_TEST(${BACKEND_NAME}, topk_3d_min_one)
{
    Shape shape{2, 3, 2};
    Shape rshape{2, 1, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::TopK>(A, 1, element::i32, 1, false);
    auto f0 = make_shared<Function>(make_shared<op::GetOutputElement>(B, 0), ParameterVector{A});
    auto f1 = make_shared<Function>(make_shared<op::GetOutputElement>(B, 1), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{12, 2, 10, 9, 8, 4, 6, 1, 5, 3, 11, 7});
    auto result0 = backend->create_tensor(element::i32, rshape);
    auto result1 = backend->create_tensor(element::f32, rshape);

    auto h0 = backend->compile(f0);
    backend->call_with_validate(h0, {result0}, {a});
    EXPECT_EQ((vector<int32_t>{2, 0, 1, 0}), read_vector<int32_t>(result0));
    auto h1 = backend->compile(f1);
    backend->call_with_validate(h1, {result1}, {a});
    EXPECT_EQ((vector<float>{8, 2, 5, 1}), read_vector<float>(result1));
}

NGRAPH_TEST(${BACKEND_NAME}, topk_2d_max_all)
{
    Shape shape{4, 3};
    Shape rshape{4, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::TopK>(A, 0, element::i32, 4, true);
    auto f0 = make_shared<Function>(make_shared<op::GetOutputElement>(B, 0), ParameterVector{A});
    auto f1 = make_shared<Function>(make_shared<op::GetOutputElement>(B, 1), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{9, 2, 10, 12, 8, 4, 6, 1, 5, 3, 11, 7});
    auto result0 = backend->create_tensor(element::i32, rshape);
    auto result1 = backend->create_tensor(element::f32, rshape);

    auto h0 = backend->compile(f0);
    backend->call_with_validate(h0, {result0}, {a});
    EXPECT_EQ((vector<int32_t>{1, 3, 0, 0, 1, 3, 2, 0, 2, 3, 2, 1}), read_vector<int32_t>(result0));
    auto h1 = backend->compile(f1);
    backend->call_with_validate(h1, {result1}, {a});
    EXPECT_EQ((vector<float>{12, 11, 10, 9, 8, 7, 6, 2, 5, 3, 1, 4}), read_vector<float>(result1));
}

NGRAPH_TEST(${BACKEND_NAME}, topk_2d_max_partial)
{
    Shape shape{4, 3};
    Shape rshape{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::TopK>(A, 0, element::i32, 2, true);
    auto f0 = make_shared<Function>(make_shared<op::GetOutputElement>(B, 0), ParameterVector{A});
    auto f1 = make_shared<Function>(make_shared<op::GetOutputElement>(B, 1), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{9, 2, 10, 12, 8, 4, 6, 1, 5, 3, 11, 7});
    auto result0 = backend->create_tensor(element::i32, rshape);
    auto result1 = backend->create_tensor(element::f32, rshape);

    auto h0 = backend->compile(f0);
    backend->call_with_validate(h0, {result0}, {a});
    EXPECT_EQ((vector<int32_t>{1, 3, 0, 0, 1, 3}), read_vector<int32_t>(result0));
    auto h1 = backend->compile(f1);
    backend->call_with_validate(h1, {result1}, {a});
    EXPECT_EQ((vector<float>{12, 11, 10, 9, 8, 7}), read_vector<float>(result1));
}

NGRAPH_TEST(${BACKEND_NAME}, topk_2d_max_one)
{
    Shape shape{4, 3};
    Shape rshape{1, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::TopK>(A, 0, element::i32, 1, true);
    auto f0 = make_shared<Function>(make_shared<op::GetOutputElement>(B, 0), ParameterVector{A});
    auto f1 = make_shared<Function>(make_shared<op::GetOutputElement>(B, 1), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{9, 2, 10, 12, 8, 4, 6, 1, 5, 3, 11, 7});
    auto result0 = backend->create_tensor(element::i32, rshape);
    auto result1 = backend->create_tensor(element::f32, rshape);

    auto h0 = backend->compile(f0);
    backend->call_with_validate(h0, {result0}, {a});
    EXPECT_EQ((vector<int32_t>{1, 3, 0}), read_vector<int32_t>(result0));
    auto h1 = backend->compile(f1);
    backend->call_with_validate(h1, {result1}, {a});
    EXPECT_EQ((vector<float>{12, 11, 10}), read_vector<float>(result1));
}

NGRAPH_TEST(${BACKEND_NAME}, topk_2d_min_all)
{
    Shape shape{4, 3};
    Shape rshape{4, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::TopK>(A, 0, element::i32, 4, false);
    auto f0 = make_shared<Function>(make_shared<op::GetOutputElement>(B, 0), ParameterVector{A});
    auto f1 = make_shared<Function>(make_shared<op::GetOutputElement>(B, 1), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{12, 2, 10, 9, 8, 4, 6, 1, 5, 3, 11, 7});
    auto result0 = backend->create_tensor(element::i32, rshape);
    auto result1 = backend->create_tensor(element::f32, rshape);

    auto h0 = backend->compile(f0);
    backend->call_with_validate(h0, {result0}, {a});
    EXPECT_EQ((vector<int32_t>{3, 2, 1, 2, 0, 2, 1, 1, 3, 0, 3, 0}), read_vector<int32_t>(result0));
    auto h1 = backend->compile(f1);
    backend->call_with_validate(h1, {result1}, {a});
    EXPECT_EQ((vector<float>{3, 1, 4, 6, 2, 5, 9, 8, 7, 12, 11, 10}), read_vector<float>(result1));
}

NGRAPH_TEST(${BACKEND_NAME}, topk_2d_min_partial)
{
    Shape shape{4, 3};
    Shape rshape{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::TopK>(A, 0, element::i32, 2, false);
    auto f0 = make_shared<Function>(make_shared<op::GetOutputElement>(B, 0), ParameterVector{A});
    auto f1 = make_shared<Function>(make_shared<op::GetOutputElement>(B, 1), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{12, 2, 10, 9, 8, 4, 6, 1, 5, 3, 11, 7});
    auto result0 = backend->create_tensor(element::i32, rshape);
    auto result1 = backend->create_tensor(element::f32, rshape);

    auto h0 = backend->compile(f0);
    backend->call_with_validate(h0, {result0}, {a});
    EXPECT_EQ((vector<int32_t>{3, 2, 1, 2, 0, 2}), read_vector<int32_t>(result0));
    auto h1 = backend->compile(f1);
    backend->call_with_validate(h1, {result1}, {a});
    EXPECT_EQ((vector<float>{3, 1, 4, 6, 2, 5}), read_vector<float>(result1));
}

NGRAPH_TEST(${BACKEND_NAME}, topk_2d_min_one)
{
    Shape shape{4, 3};
    Shape rshape{1, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::TopK>(A, 0, element::i32, 1, false);
    auto f0 = make_shared<Function>(make_shared<op::GetOutputElement>(B, 0), ParameterVector{A});
    auto f1 = make_shared<Function>(make_shared<op::GetOutputElement>(B, 1), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{12, 2, 10, 9, 8, 4, 6, 1, 5, 3, 11, 7});
    auto result0 = backend->create_tensor(element::i32, rshape);
    auto result1 = backend->create_tensor(element::f32, rshape);

    auto h0 = backend->compile(f0);
    backend->call_with_validate(h0, {result0}, {a});
    EXPECT_EQ((vector<int32_t>{3, 2, 1}), read_vector<int32_t>(result0));
    auto h1 = backend->compile(f1);
    backend->call_with_validate(h1, {result1}, {a});
    EXPECT_EQ((vector<float>{3, 1, 4}), read_vector<float>(result1));
}

NGRAPH_TEST(${BACKEND_NAME}, topk_3d_large_input_max)
{
    Shape shape{4, 8192, 5};
    auto A = make_shared<op::Parameter>(element::f32, shape);

    auto B = make_shared<op::TopK>(A, 1, element::i32, 10, true);

    auto interp_f_0 =
        make_shared<Function>(make_shared<op::GetOutputElement>(B, 0), ParameterVector{A});
    auto interp_f_1 =
        make_shared<Function>(make_shared<op::GetOutputElement>(B, 1), ParameterVector{A});
    auto gpu_f_0 = ngraph::clone_function(*interp_f_0);
    auto gpu_f_1 = ngraph::clone_function(*interp_f_1);

    vector<vector<float>> args;
    for (shared_ptr<op::Parameter> param : interp_f_0->get_parameters())
    {
        vector<float> tensor_val(shape_size(param->get_shape()));
        iota(tensor_val.begin(), tensor_val.end(), 0.0f);
        args.push_back(tensor_val);
    }

    auto interp_results_0 = execute<float, int32_t>(interp_f_0, args, "INTERPRETER");
    auto gpu_results_0 = execute<float, int32_t>(gpu_f_0, args, "${BACKEND_NAME}");
    for (size_t i = 0; i < gpu_results_0.size(); i++)
    {
        EXPECT_EQ(gpu_results_0.at(i), interp_results_0.at(i));
    }

    auto interp_results_1 = execute(interp_f_1, args, "INTERPRETER");
    auto gpu_results_1 = execute(gpu_f_1, args, "${BACKEND_NAME}");

    for (size_t i = 0; i < gpu_results_1.size(); i++)
    {
        EXPECT_TRUE(test::all_close_f(gpu_results_1.at(i), interp_results_1.at(i), 24, 0));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, topk_3d_large_input_min)
{
    Shape shape{4, 8192, 5};
    auto A = make_shared<op::Parameter>(element::f32, shape);

    auto B = make_shared<op::TopK>(A, 1, element::i32, 10, false);

    auto interp_f_0 =
        make_shared<Function>(make_shared<op::GetOutputElement>(B, 0), ParameterVector{A});
    auto interp_f_1 =
        make_shared<Function>(make_shared<op::GetOutputElement>(B, 1), ParameterVector{A});
    auto gpu_f_0 = ngraph::clone_function(*interp_f_0);
    auto gpu_f_1 = ngraph::clone_function(*interp_f_1);

    vector<vector<float>> args;
    for (shared_ptr<op::Parameter> param : interp_f_0->get_parameters())
    {
        vector<float> tensor_val(shape_size(param->get_shape()));
        iota(tensor_val.begin(), tensor_val.end(), 0.0f);
        args.push_back(tensor_val);
    }

    auto interp_results_0 = execute<float, int32_t>(interp_f_0, args, "INTERPRETER");
    auto gpu_results_0 = execute<float, int32_t>(gpu_f_0, args, "${BACKEND_NAME}");
    for (size_t i = 0; i < gpu_results_0.size(); i++)
    {
        EXPECT_EQ(gpu_results_0.at(i), interp_results_0.at(i));
    }

    auto interp_results_1 = execute(interp_f_1, args, "INTERPRETER");
    auto gpu_results_1 = execute(gpu_f_1, args, "${BACKEND_NAME}");

    for (size_t i = 0; i < gpu_results_1.size(); i++)
    {
        EXPECT_TRUE(test::all_close_f(gpu_results_1.at(i), interp_results_1.at(i), 24, 0));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, topk_3d_single_output)
{
    Shape shape{2, 3, 2};
    Shape rshape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::TopK>(A, 1, element::i32, 2, false);
    auto f0 = make_shared<Function>(make_shared<op::GetOutputElement>(B, 0), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{12, 2, 10, 9, 8, 4, 6, 1, 5, 3, 11, 7});
    auto result0 = backend->create_tensor(element::i32, rshape);

    auto h0 = backend->compile(f0);
    backend->call_with_validate(h0, {result0}, {a});
    EXPECT_EQ((vector<int32_t>{2, 0, 1, 2, 1, 0, 0, 1}), read_vector<int32_t>(result0));
}
