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

namespace
{
    template <typename T>
    ::testing::AssertionResult clamp_test(const string& backend,
                                          const element::Type& type,
                                          const PartialShape& dynamic_shape,
                                          const Shape& static_shape,
                                          const std::vector<T>& input,
                                          double min,
                                          double max,
                                          const std::vector<T>& output)
    {
        auto data = make_shared<op::v0::Parameter>(type, dynamic_shape);
        auto clamp = make_shared<op::v0::Clamp>(data, min, max);
        auto function = make_shared<Function>(clamp, ParameterVector{data});

        auto mode = test::BackendMode::STATIC;
        if (dynamic_shape.is_dynamic())
        {
            mode = test::BackendMode::DYNAMIC;
        }
        auto test_case = test::NgraphTestCase(function, backend, mode);
        test_case.add_input<T>(static_shape, input);
        test_case.add_expected_output<T>(static_shape, output);
        return test_case.run();
    }
}

NGRAPH_TEST(${BACKEND_NAME}, clamp_double_static)
{
    auto type = element::f64;
    typedef double ctype;

    auto sshape = Shape{5, 2};
    auto dshape = PartialShape::dynamic();

    auto min = numeric_limits<ctype>::min();
    auto max = numeric_limits<ctype>::max();
    auto pinf = numeric_limits<double>::infinity();
    auto ninf = -numeric_limits<double>::infinity();

    vector<ctype> input{min, max, ninf, pinf, 9.99999, 10.0, 10.000001, 19.999999, 20.0, 20.000001};

    // static shape
    EXPECT_TRUE(clamp_test<ctype>("${BACKEND_NAME}",
                                  type,
                                  sshape,
                                  sshape,
                                  {-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8},
                                  0.2,
                                  0.6,
                                  {0.2, 0.2, 0.2, 0.2, 0.3, 0.4, 0.5, 0.6, 0.6, 0.6}));

    EXPECT_TRUE(
        clamp_test<ctype>("${BACKEND_NAME}",
                          type,
                          sshape,
                          sshape,
                          input,
                          10.0,
                          20.0,
                          {10.0, 20.0, 10.0, 20.0, 10.0, 10.0, 10.000001, 19.999999, 20.0, 20.0}));
    EXPECT_TRUE(clamp_test<ctype>(
        "${BACKEND_NAME}",
        type,
        sshape,
        sshape,
        input,
        10.0,
        pinf,
        {10.0, max, 10.0, pinf, 10.0, 10.0, 10.000001, 19.999999, 20.0, 20.000001}));
    EXPECT_TRUE(clamp_test<ctype>(
        "${BACKEND_NAME}",
        type,
        sshape,
        sshape,
        input,
        ninf,
        20.0,
        {min, 20.0, ninf, 20.0, 9.99999, 10.0, 10.000001, 19.999999, 20.0, 20.0}));
}

NGRAPH_TEST(${BACKEND_NAME}, clamp_float_static)
{
    auto type = element::f32;
    typedef float ctype;

    auto sshape = Shape{5, 2};
    auto dshape = PartialShape::dynamic();

    auto min = numeric_limits<ctype>::min();
    auto max = numeric_limits<ctype>::max();
    auto pinf = numeric_limits<float>::infinity();
    auto ninf = -numeric_limits<float>::infinity();

    vector<ctype> input{min, max, ninf, pinf, 9.99999, 10.0, 10.000001, 19.999999, 20.0, 20.000001};

    // static shape
    EXPECT_TRUE(clamp_test<ctype>("${BACKEND_NAME}",
                                  type,
                                  sshape,
                                  sshape,
                                  {-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8},
                                  0.2,
                                  0.6,
                                  {0.2, 0.2, 0.2, 0.2, 0.3, 0.4, 0.5, 0.6, 0.6, 0.6}));

    EXPECT_TRUE(
        clamp_test<ctype>("${BACKEND_NAME}",
                          type,
                          sshape,
                          sshape,
                          input,
                          10.0,
                          20.0,
                          {10.0, 20.0, 10.0, 20.0, 10.0, 10.0, 10.000001, 19.999999, 20.0, 20.0}));
    EXPECT_TRUE(clamp_test<ctype>(
        "${BACKEND_NAME}",
        type,
        sshape,
        sshape,
        input,
        10.0,
        pinf,
        {10.0, max, 10.0, pinf, 10.0, 10.0, 10.000001, 19.999999, 20.0, 20.000001}));
    EXPECT_TRUE(clamp_test<ctype>(
        "${BACKEND_NAME}",
        type,
        sshape,
        sshape,
        input,
        ninf,
        20.0,
        {min, 20.0, ninf, 20.0, 9.99999, 10.0, 10.000001, 19.999999, 20.0, 20.0}));
}

NGRAPH_TEST(${BACKEND_NAME}, clamp_int8_static)
{
    auto type = element::i8;
    typedef int8_t ctype;

    auto sshape = Shape{4, 2};
    auto dshape = PartialShape::dynamic();

    auto min = numeric_limits<ctype>::min();
    auto max = numeric_limits<ctype>::max();
    auto pinf = numeric_limits<double>::infinity();
    auto ninf = -numeric_limits<double>::infinity();

    vector<ctype> input{min, max, 9, 10, 11, 19, 20, 21};

    // static shape
    EXPECT_TRUE(clamp_test<ctype>("${BACKEND_NAME}",
                                  type,
                                  sshape,
                                  sshape,
                                  input,
                                  10.0,
                                  20.0,
                                  {10, 20, 10, 10, 11, 19, 20, 20}));
    EXPECT_TRUE(clamp_test<ctype>("${BACKEND_NAME}",
                                  type,
                                  sshape,
                                  sshape,
                                  input,
                                  10.0,
                                  pinf,
                                  {10, max, 10, 10, 11, 19, 20, 21}));
    EXPECT_TRUE(clamp_test<ctype>("${BACKEND_NAME}",
                                  type,
                                  sshape,
                                  sshape,
                                  input,
                                  ninf,
                                  20.0,
                                  {min, 20, 9, 10, 11, 19, 20, 20}));
}

NGRAPH_TEST(${BACKEND_NAME}, clamp_int16_static)
{
    auto type = element::i16;
    typedef int16_t ctype;

    auto sshape = Shape{4, 2};
    auto dshape = PartialShape::dynamic();

    auto min = numeric_limits<ctype>::min();
    auto max = numeric_limits<ctype>::max();
    auto pinf = numeric_limits<double>::infinity();
    auto ninf = -numeric_limits<double>::infinity();

    vector<ctype> input{min, max, 9, 10, 11, 19, 20, 21};

    // static shape
    EXPECT_TRUE(clamp_test<ctype>("${BACKEND_NAME}",
                                  type,
                                  sshape,
                                  sshape,
                                  input,
                                  10.0,
                                  20.0,
                                  {10, 20, 10, 10, 11, 19, 20, 20}));
    EXPECT_TRUE(clamp_test<ctype>("${BACKEND_NAME}",
                                  type,
                                  sshape,
                                  sshape,
                                  input,
                                  10.0,
                                  pinf,
                                  {10, max, 10, 10, 11, 19, 20, 21}));
    EXPECT_TRUE(clamp_test<ctype>("${BACKEND_NAME}",
                                  type,
                                  sshape,
                                  sshape,
                                  input,
                                  ninf,
                                  20.0,
                                  {min, 20, 9, 10, 11, 19, 20, 20}));
}

NGRAPH_TEST(${BACKEND_NAME}, clamp_int32_static)
{
    auto type = element::i32;
    typedef int32_t ctype;

    auto sshape = Shape{4, 2};
    auto dshape = PartialShape::dynamic();

    auto min = numeric_limits<ctype>::min();
    auto max = numeric_limits<ctype>::max();
    auto pinf = numeric_limits<double>::infinity();
    auto ninf = -numeric_limits<double>::infinity();

    vector<ctype> input{min, max, 9, 10, 11, 19, 20, 21};

    // static shape
    EXPECT_TRUE(clamp_test<ctype>("${BACKEND_NAME}",
                                  type,
                                  sshape,
                                  sshape,
                                  input,
                                  10.0,
                                  20.0,
                                  {10, 20, 10, 10, 11, 19, 20, 20}));
    EXPECT_TRUE(clamp_test<ctype>("${BACKEND_NAME}",
                                  type,
                                  sshape,
                                  sshape,
                                  input,
                                  10.0,
                                  pinf,
                                  {10, max, 10, 10, 11, 19, 20, 21}));
    EXPECT_TRUE(clamp_test<ctype>("${BACKEND_NAME}",
                                  type,
                                  sshape,
                                  sshape,
                                  input,
                                  ninf,
                                  20.0,
                                  {min, 20, 9, 10, 11, 19, 20, 20}));
}

NGRAPH_TEST(${BACKEND_NAME}, clamp_int64_static)
{
    auto type = element::i64;
    typedef int64_t ctype;

    auto sshape = Shape{4, 2};
    auto dshape = PartialShape::dynamic();

    auto min = numeric_limits<ctype>::min();
    auto max = numeric_limits<ctype>::max();
    auto pinf = numeric_limits<double>::infinity();
    auto ninf = -numeric_limits<double>::infinity();

    vector<ctype> input{min, max, 9, 10, 11, 19, 20, 21};

    // static shape
    EXPECT_TRUE(clamp_test<ctype>("${BACKEND_NAME}",
                                  type,
                                  sshape,
                                  sshape,
                                  input,
                                  10.0,
                                  20.0,
                                  {10, 20, 10, 10, 11, 19, 20, 20}));
    EXPECT_TRUE(clamp_test<ctype>("${BACKEND_NAME}",
                                  type,
                                  sshape,
                                  sshape,
                                  input,
                                  10.0,
                                  pinf,
                                  {10, max, 10, 10, 11, 19, 20, 21}));
    EXPECT_TRUE(clamp_test<ctype>("${BACKEND_NAME}",
                                  type,
                                  sshape,
                                  sshape,
                                  input,
                                  ninf,
                                  20.0,
                                  {min, 20, 9, 10, 11, 19, 20, 20}));
}

NGRAPH_TEST(${BACKEND_NAME}, clamp_uint8_static)
{
    auto type = element::u8;
    typedef uint8_t ctype;

    auto sshape = Shape{4, 2};
    auto dshape = PartialShape::dynamic();

    auto min = numeric_limits<ctype>::min();
    auto max = numeric_limits<ctype>::max();
    auto pinf = numeric_limits<double>::infinity();
    auto ninf = -numeric_limits<double>::infinity();

    vector<ctype> input{min, max, 9, 10, 11, 19, 20, 21};

    // static shape
    EXPECT_TRUE(clamp_test<ctype>("${BACKEND_NAME}",
                                  type,
                                  sshape,
                                  sshape,
                                  input,
                                  10.0,
                                  20.0,
                                  {10, 20, 10, 10, 11, 19, 20, 20}));
    EXPECT_TRUE(clamp_test<ctype>("${BACKEND_NAME}",
                                  type,
                                  sshape,
                                  sshape,
                                  input,
                                  10.0,
                                  pinf,
                                  {10, max, 10, 10, 11, 19, 20, 21}));
    EXPECT_TRUE(clamp_test<ctype>("${BACKEND_NAME}",
                                  type,
                                  sshape,
                                  sshape,
                                  input,
                                  ninf,
                                  20.0,
                                  {min, 20, 9, 10, 11, 19, 20, 20}));
}

NGRAPH_TEST(${BACKEND_NAME}, clamp_uint16_static)
{
    auto type = element::u16;
    typedef uint16_t ctype;

    auto sshape = Shape{4, 2};
    auto dshape = PartialShape::dynamic();

    auto min = numeric_limits<ctype>::min();
    auto max = numeric_limits<ctype>::max();
    auto pinf = numeric_limits<double>::infinity();
    auto ninf = -numeric_limits<double>::infinity();

    vector<ctype> input{min, max, 9, 10, 11, 19, 20, 21};

    // static shape
    EXPECT_TRUE(clamp_test<ctype>("${BACKEND_NAME}",
                                  type,
                                  sshape,
                                  sshape,
                                  input,
                                  10.0,
                                  20.0,
                                  {10, 20, 10, 10, 11, 19, 20, 20}));
    EXPECT_TRUE(clamp_test<ctype>("${BACKEND_NAME}",
                                  type,
                                  sshape,
                                  sshape,
                                  input,
                                  10.0,
                                  pinf,
                                  {10, max, 10, 10, 11, 19, 20, 21}));
    EXPECT_TRUE(clamp_test<ctype>("${BACKEND_NAME}",
                                  type,
                                  sshape,
                                  sshape,
                                  input,
                                  ninf,
                                  20.0,
                                  {min, 20, 9, 10, 11, 19, 20, 20}));
}

NGRAPH_TEST(${BACKEND_NAME}, clamp_uint32_static)
{
    auto type = element::u32;
    typedef uint32_t ctype;

    auto sshape = Shape{4, 2};
    auto dshape = PartialShape::dynamic();

    auto min = numeric_limits<ctype>::min();
    auto max = numeric_limits<ctype>::max();
    auto pinf = numeric_limits<double>::infinity();
    auto ninf = -numeric_limits<double>::infinity();

    vector<ctype> input{min, max, 9, 10, 11, 19, 20, 21};

    // static shape
    EXPECT_TRUE(clamp_test<ctype>("${BACKEND_NAME}",
                                  type,
                                  sshape,
                                  sshape,
                                  input,
                                  10.0,
                                  20.0,
                                  {10, 20, 10, 10, 11, 19, 20, 20}));
    EXPECT_TRUE(clamp_test<ctype>("${BACKEND_NAME}",
                                  type,
                                  sshape,
                                  sshape,
                                  input,
                                  10.0,
                                  pinf,
                                  {10, max, 10, 10, 11, 19, 20, 21}));
    EXPECT_TRUE(clamp_test<ctype>("${BACKEND_NAME}",
                                  type,
                                  sshape,
                                  sshape,
                                  input,
                                  ninf,
                                  20.0,
                                  {min, 20, 9, 10, 11, 19, 20, 20}));
}

NGRAPH_TEST(${BACKEND_NAME}, clamp_uint64_static)
{
    auto type = element::u64;
    typedef uint64_t ctype;

    auto sshape = Shape{4, 2};
    auto dshape = PartialShape::dynamic();

    auto min = numeric_limits<ctype>::min();
    auto max = numeric_limits<ctype>::max();
    auto pinf = numeric_limits<double>::infinity();
    auto ninf = -numeric_limits<double>::infinity();

    vector<ctype> input{min, max, 9, 10, 11, 19, 20, 21};

    // static shape
    EXPECT_TRUE(clamp_test<ctype>("${BACKEND_NAME}",
                                  type,
                                  sshape,
                                  sshape,
                                  input,
                                  10.0,
                                  20.0,
                                  {10, 20, 10, 10, 11, 19, 20, 20}));
    EXPECT_TRUE(clamp_test<ctype>("${BACKEND_NAME}",
                                  type,
                                  sshape,
                                  sshape,
                                  input,
                                  10.0,
                                  pinf,
                                  {10, max, 10, 10, 11, 19, 20, 21}));
    EXPECT_TRUE(clamp_test<ctype>("${BACKEND_NAME}",
                                  type,
                                  sshape,
                                  sshape,
                                  input,
                                  ninf,
                                  20.0,
                                  {min, 20, 9, 10, 11, 19, 20, 20}));
}

NGRAPH_TEST(${BACKEND_NAME}, clamp_float16_static)
{
    auto type = element::f16;
    typedef float16 ctype;

    auto sshape = Shape{5, 2};
    auto dshape = PartialShape::dynamic();

    auto min = numeric_limits<ctype>::min();
    auto max = numeric_limits<ctype>::max();
    auto pinf = numeric_limits<float>::infinity();
    auto ninf = -numeric_limits<float>::infinity();

    vector<ctype> input{min, max, ninf, pinf, 9.99999, 10.0, 10.000001, 19.999999, 20.0, 20.000001};

    // static shape
    EXPECT_TRUE(clamp_test<ctype>("${BACKEND_NAME}",
                                  type,
                                  sshape,
                                  sshape,
                                  {-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8},
                                  0.2,
                                  0.6,
                                  {0.2, 0.2, 0.2, 0.2, 0.3, 0.4, 0.5, 0.6, 0.6, 0.6}));

    EXPECT_TRUE(
        clamp_test<ctype>("${BACKEND_NAME}",
                          type,
                          sshape,
                          sshape,
                          input,
                          10.0,
                          20.0,
                          {10.0, 20.0, 10.0, 20.0, 10.0, 10.0, 10.000001, 19.999999, 20.0, 20.0}));
    EXPECT_TRUE(clamp_test<ctype>(
        "${BACKEND_NAME}",
        type,
        sshape,
        sshape,
        input,
        10.0,
        pinf,
        {10.0, max, 10.0, pinf, 10.0, 10.0, 10.000001, 19.999999, 20.0, 20.000001}));
    EXPECT_TRUE(clamp_test<ctype>(
        "${BACKEND_NAME}",
        type,
        sshape,
        sshape,
        input,
        ninf,
        20.0,
        {min, 20.0, ninf, 20.0, 9.99999, 10.0, 10.000001, 19.999999, 20.0, 20.0}));
}

NGRAPH_TEST(${BACKEND_NAME}, clamp_bfloat16_static)
{
    auto type = element::bf16;
    typedef bfloat16 ctype;

    auto sshape = Shape{5, 2};
    auto dshape = PartialShape::dynamic();

    auto min = numeric_limits<ctype>::min();
    auto max = numeric_limits<ctype>::max();
    auto pinf = numeric_limits<float>::infinity();
    auto ninf = -numeric_limits<float>::infinity();

    vector<ctype> input{min, max, ninf, pinf, 9.99999, 10.0, 10.000001, 19.999999, 20.0, 20.000001};

    // static shape
    EXPECT_TRUE(clamp_test<ctype>("${BACKEND_NAME}",
                                  type,
                                  sshape,
                                  sshape,
                                  {-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8},
                                  0.2,
                                  0.6,
                                  {0.2, 0.2, 0.2, 0.2, 0.3, 0.4, 0.5, 0.6, 0.6, 0.6}));

    EXPECT_TRUE(
        clamp_test<ctype>("${BACKEND_NAME}",
                          type,
                          sshape,
                          sshape,
                          input,
                          10.0,
                          20.0,
                          {10.0, 20.0, 10.0, 20.0, 10.0, 10.0, 10.000001, 19.999999, 20.0, 20.0}));
    EXPECT_TRUE(clamp_test<ctype>(
        "${BACKEND_NAME}",
        type,
        sshape,
        sshape,
        input,
        10.0,
        pinf,
        {10.0, max, 10.0, pinf, 10.0, 10.0, 10.000001, 19.999999, 20.0, 20.000001}));
    EXPECT_TRUE(clamp_test<ctype>(
        "${BACKEND_NAME}",
        type,
        sshape,
        sshape,
        input,
        ninf,
        20.0,
        {min, 20.0, ninf, 20.0, 9.99999, 10.0, 10.000001, 19.999999, 20.0, 20.0}));
}

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////

NGRAPH_TEST(${BACKEND_NAME}, clamp_double_dynamic)
{
    auto type = element::f64;
    typedef double ctype;

    auto sshape = Shape{5, 2};
    auto dshape = PartialShape::dynamic();

    auto min = numeric_limits<ctype>::min();
    auto max = numeric_limits<ctype>::max();
    auto pinf = numeric_limits<double>::infinity();
    auto ninf = -numeric_limits<double>::infinity();

    vector<ctype> input{min, max, ninf, pinf, 9.99999, 10.0, 10.000001, 19.999999, 20.0, 20.000001};

    // dynamic shape
    EXPECT_TRUE(clamp_test<ctype>("${BACKEND_NAME}",
                                  type,
                                  dshape,
                                  sshape,
                                  {-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8},
                                  0.2,
                                  0.6,
                                  {0.2, 0.2, 0.2, 0.2, 0.3, 0.4, 0.5, 0.6, 0.6, 0.6}));

    EXPECT_TRUE(
        clamp_test<ctype>("${BACKEND_NAME}",
                          type,
                          dshape,
                          sshape,
                          input,
                          10.0,
                          20.0,
                          {10.0, 20.0, 10.0, 20.0, 10.0, 10.0, 10.000001, 19.999999, 20.0, 20.0}));
    EXPECT_TRUE(clamp_test<ctype>(
        "${BACKEND_NAME}",
        type,
        dshape,
        sshape,
        input,
        10.0,
        pinf,
        {10.0, max, 10.0, pinf, 10.0, 10.0, 10.000001, 19.999999, 20.0, 20.000001}));
    EXPECT_TRUE(clamp_test<ctype>(
        "${BACKEND_NAME}",
        type,
        dshape,
        sshape,
        input,
        ninf,
        20.0,
        {min, 20.0, ninf, 20.0, 9.99999, 10.0, 10.000001, 19.999999, 20.0, 20.0}));
}

NGRAPH_TEST(${BACKEND_NAME}, clamp_float_dynamic)
{
    auto type = element::f32;
    typedef float ctype;

    auto sshape = Shape{5, 2};
    auto dshape = PartialShape::dynamic();

    auto min = numeric_limits<ctype>::min();
    auto max = numeric_limits<ctype>::max();
    auto pinf = numeric_limits<float>::infinity();
    auto ninf = -numeric_limits<float>::infinity();

    vector<ctype> input{min, max, ninf, pinf, 9.99999, 10.0, 10.000001, 19.999999, 20.0, 20.000001};

    // dynamic shape
    EXPECT_TRUE(clamp_test<ctype>("${BACKEND_NAME}",
                                  type,
                                  dshape,
                                  sshape,
                                  {-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8},
                                  0.2,
                                  0.6,
                                  {0.2, 0.2, 0.2, 0.2, 0.3, 0.4, 0.5, 0.6, 0.6, 0.6}));

    EXPECT_TRUE(
        clamp_test<ctype>("${BACKEND_NAME}",
                          type,
                          dshape,
                          sshape,
                          input,
                          10.0,
                          20.0,
                          {10.0, 20.0, 10.0, 20.0, 10.0, 10.0, 10.000001, 19.999999, 20.0, 20.0}));
    EXPECT_TRUE(clamp_test<ctype>(
        "${BACKEND_NAME}",
        type,
        dshape,
        sshape,
        input,
        10.0,
        pinf,
        {10.0, max, 10.0, pinf, 10.0, 10.0, 10.000001, 19.999999, 20.0, 20.000001}));
    EXPECT_TRUE(clamp_test<ctype>(
        "${BACKEND_NAME}",
        type,
        dshape,
        sshape,
        input,
        ninf,
        20.0,
        {min, 20.0, ninf, 20.0, 9.99999, 10.0, 10.000001, 19.999999, 20.0, 20.0}));
}

NGRAPH_TEST(${BACKEND_NAME}, clamp_int8_dynamic)
{
    auto type = element::i8;
    typedef int8_t ctype;

    auto sshape = Shape{4, 2};
    auto dshape = PartialShape::dynamic();

    auto min = numeric_limits<ctype>::min();
    auto max = numeric_limits<ctype>::max();
    auto pinf = numeric_limits<double>::infinity();
    auto ninf = -numeric_limits<double>::infinity();

    vector<ctype> input{min, max, 9, 10, 11, 19, 20, 21};

    // dynamic shape
    EXPECT_TRUE(clamp_test<ctype>("${BACKEND_NAME}",
                                  type,
                                  dshape,
                                  sshape,
                                  input,
                                  10.0,
                                  20.0,
                                  {10, 20, 10, 10, 11, 19, 20, 20}));
    EXPECT_TRUE(clamp_test<ctype>("${BACKEND_NAME}",
                                  type,
                                  dshape,
                                  sshape,
                                  input,
                                  10.0,
                                  pinf,
                                  {10, max, 10, 10, 11, 19, 20, 21}));
    EXPECT_TRUE(clamp_test<ctype>("${BACKEND_NAME}",
                                  type,
                                  dshape,
                                  sshape,
                                  input,
                                  ninf,
                                  20.0,
                                  {min, 20, 9, 10, 11, 19, 20, 20}));
}

NGRAPH_TEST(${BACKEND_NAME}, clamp_int16_dynamic)
{
    auto type = element::i16;
    typedef int16_t ctype;

    auto sshape = Shape{4, 2};
    auto dshape = PartialShape::dynamic();

    auto min = numeric_limits<ctype>::min();
    auto max = numeric_limits<ctype>::max();
    auto pinf = numeric_limits<double>::infinity();
    auto ninf = -numeric_limits<double>::infinity();

    vector<ctype> input{min, max, 9, 10, 11, 19, 20, 21};

    // dynamic shape
    EXPECT_TRUE(clamp_test<ctype>("${BACKEND_NAME}",
                                  type,
                                  dshape,
                                  sshape,
                                  input,
                                  10.0,
                                  20.0,
                                  {10, 20, 10, 10, 11, 19, 20, 20}));
    EXPECT_TRUE(clamp_test<ctype>("${BACKEND_NAME}",
                                  type,
                                  dshape,
                                  sshape,
                                  input,
                                  10.0,
                                  pinf,
                                  {10, max, 10, 10, 11, 19, 20, 21}));
    EXPECT_TRUE(clamp_test<ctype>("${BACKEND_NAME}",
                                  type,
                                  dshape,
                                  sshape,
                                  input,
                                  ninf,
                                  20.0,
                                  {min, 20, 9, 10, 11, 19, 20, 20}));
}

NGRAPH_TEST(${BACKEND_NAME}, clamp_int32_dynamic)
{
    auto type = element::i32;
    typedef int32_t ctype;

    auto sshape = Shape{4, 2};
    auto dshape = PartialShape::dynamic();

    auto min = numeric_limits<ctype>::min();
    auto max = numeric_limits<ctype>::max();
    auto pinf = numeric_limits<double>::infinity();
    auto ninf = -numeric_limits<double>::infinity();

    vector<ctype> input{min, max, 9, 10, 11, 19, 20, 21};

    // dynamic shape
    EXPECT_TRUE(clamp_test<ctype>("${BACKEND_NAME}",
                                  type,
                                  dshape,
                                  sshape,
                                  input,
                                  10.0,
                                  20.0,
                                  {10, 20, 10, 10, 11, 19, 20, 20}));
    EXPECT_TRUE(clamp_test<ctype>("${BACKEND_NAME}",
                                  type,
                                  dshape,
                                  sshape,
                                  input,
                                  10.0,
                                  pinf,
                                  {10, max, 10, 10, 11, 19, 20, 21}));
    EXPECT_TRUE(clamp_test<ctype>("${BACKEND_NAME}",
                                  type,
                                  dshape,
                                  sshape,
                                  input,
                                  ninf,
                                  20.0,
                                  {min, 20, 9, 10, 11, 19, 20, 20}));
}

NGRAPH_TEST(${BACKEND_NAME}, clamp_int64_dynamic)
{
    auto type = element::i64;
    typedef int64_t ctype;

    auto sshape = Shape{4, 2};
    auto dshape = PartialShape::dynamic();

    auto min = numeric_limits<ctype>::min();
    auto max = numeric_limits<ctype>::max();
    auto pinf = numeric_limits<double>::infinity();
    auto ninf = -numeric_limits<double>::infinity();

    vector<ctype> input{min, max, 9, 10, 11, 19, 20, 21};

    // dynamic shape
    EXPECT_TRUE(clamp_test<ctype>("${BACKEND_NAME}",
                                  type,
                                  dshape,
                                  sshape,
                                  input,
                                  10.0,
                                  20.0,
                                  {10, 20, 10, 10, 11, 19, 20, 20}));
    EXPECT_TRUE(clamp_test<ctype>("${BACKEND_NAME}",
                                  type,
                                  dshape,
                                  sshape,
                                  input,
                                  10.0,
                                  pinf,
                                  {10, max, 10, 10, 11, 19, 20, 21}));
    EXPECT_TRUE(clamp_test<ctype>("${BACKEND_NAME}",
                                  type,
                                  dshape,
                                  sshape,
                                  input,
                                  ninf,
                                  20.0,
                                  {min, 20, 9, 10, 11, 19, 20, 20}));
}

NGRAPH_TEST(${BACKEND_NAME}, clamp_uint8_dynamic)
{
    auto type = element::u8;
    typedef uint8_t ctype;

    auto sshape = Shape{4, 2};
    auto dshape = PartialShape::dynamic();

    auto min = numeric_limits<ctype>::min();
    auto max = numeric_limits<ctype>::max();
    auto pinf = numeric_limits<double>::infinity();
    auto ninf = -numeric_limits<double>::infinity();

    vector<ctype> input{min, max, 9, 10, 11, 19, 20, 21};

    // dynamic shape
    EXPECT_TRUE(clamp_test<ctype>("${BACKEND_NAME}",
                                  type,
                                  dshape,
                                  sshape,
                                  input,
                                  10.0,
                                  20.0,
                                  {10, 20, 10, 10, 11, 19, 20, 20}));
    EXPECT_TRUE(clamp_test<ctype>("${BACKEND_NAME}",
                                  type,
                                  dshape,
                                  sshape,
                                  input,
                                  10.0,
                                  pinf,
                                  {10, max, 10, 10, 11, 19, 20, 21}));
    EXPECT_TRUE(clamp_test<ctype>("${BACKEND_NAME}",
                                  type,
                                  dshape,
                                  sshape,
                                  input,
                                  ninf,
                                  20.0,
                                  {min, 20, 9, 10, 11, 19, 20, 20}));
}

NGRAPH_TEST(${BACKEND_NAME}, clamp_uint16_dynamic)
{
    auto type = element::u16;
    typedef uint16_t ctype;

    auto sshape = Shape{4, 2};
    auto dshape = PartialShape::dynamic();

    auto min = numeric_limits<ctype>::min();
    auto max = numeric_limits<ctype>::max();
    auto pinf = numeric_limits<double>::infinity();
    auto ninf = -numeric_limits<double>::infinity();

    vector<ctype> input{min, max, 9, 10, 11, 19, 20, 21};

    // dynamic shape
    EXPECT_TRUE(clamp_test<ctype>("${BACKEND_NAME}",
                                  type,
                                  dshape,
                                  sshape,
                                  input,
                                  10.0,
                                  20.0,
                                  {10, 20, 10, 10, 11, 19, 20, 20}));
    EXPECT_TRUE(clamp_test<ctype>("${BACKEND_NAME}",
                                  type,
                                  dshape,
                                  sshape,
                                  input,
                                  10.0,
                                  pinf,
                                  {10, max, 10, 10, 11, 19, 20, 21}));
    EXPECT_TRUE(clamp_test<ctype>("${BACKEND_NAME}",
                                  type,
                                  dshape,
                                  sshape,
                                  input,
                                  ninf,
                                  20.0,
                                  {min, 20, 9, 10, 11, 19, 20, 20}));
}

NGRAPH_TEST(${BACKEND_NAME}, clamp_uint32_dynamic)
{
    auto type = element::u32;
    typedef uint32_t ctype;

    auto sshape = Shape{4, 2};
    auto dshape = PartialShape::dynamic();

    auto min = numeric_limits<ctype>::min();
    auto max = numeric_limits<ctype>::max();
    auto pinf = numeric_limits<double>::infinity();
    auto ninf = -numeric_limits<double>::infinity();

    vector<ctype> input{min, max, 9, 10, 11, 19, 20, 21};

    // dynamic shape
    EXPECT_TRUE(clamp_test<ctype>("${BACKEND_NAME}",
                                  type,
                                  dshape,
                                  sshape,
                                  input,
                                  10.0,
                                  20.0,
                                  {10, 20, 10, 10, 11, 19, 20, 20}));
    EXPECT_TRUE(clamp_test<ctype>("${BACKEND_NAME}",
                                  type,
                                  dshape,
                                  sshape,
                                  input,
                                  10.0,
                                  pinf,
                                  {10, max, 10, 10, 11, 19, 20, 21}));
    EXPECT_TRUE(clamp_test<ctype>("${BACKEND_NAME}",
                                  type,
                                  dshape,
                                  sshape,
                                  input,
                                  ninf,
                                  20.0,
                                  {min, 20, 9, 10, 11, 19, 20, 20}));
}

NGRAPH_TEST(${BACKEND_NAME}, clamp_uint64_dynamic)
{
    auto type = element::u64;
    typedef uint64_t ctype;

    auto sshape = Shape{4, 2};
    auto dshape = PartialShape::dynamic();

    auto min = numeric_limits<ctype>::min();
    auto max = numeric_limits<ctype>::max();
    auto pinf = numeric_limits<double>::infinity();
    auto ninf = -numeric_limits<double>::infinity();

    vector<ctype> input{min, max, 9, 10, 11, 19, 20, 21};

    // dynamic shape
    EXPECT_TRUE(clamp_test<ctype>("${BACKEND_NAME}",
                                  type,
                                  dshape,
                                  sshape,
                                  input,
                                  10.0,
                                  20.0,
                                  {10, 20, 10, 10, 11, 19, 20, 20}));
    EXPECT_TRUE(clamp_test<ctype>("${BACKEND_NAME}",
                                  type,
                                  dshape,
                                  sshape,
                                  input,
                                  10.0,
                                  pinf,
                                  {10, max, 10, 10, 11, 19, 20, 21}));
    EXPECT_TRUE(clamp_test<ctype>("${BACKEND_NAME}",
                                  type,
                                  dshape,
                                  sshape,
                                  input,
                                  ninf,
                                  20.0,
                                  {min, 20, 9, 10, 11, 19, 20, 20}));
}

NGRAPH_TEST(${BACKEND_NAME}, clamp_float16_dynamic)
{
    auto type = element::f16;
    typedef float16 ctype;

    auto sshape = Shape{5, 2};
    auto dshape = PartialShape::dynamic();

    auto min = numeric_limits<ctype>::min();
    auto max = numeric_limits<ctype>::max();
    auto pinf = numeric_limits<float>::infinity();
    auto ninf = -numeric_limits<float>::infinity();

    vector<ctype> input{min, max, ninf, pinf, 9.99999, 10.0, 10.000001, 19.999999, 20.0, 20.000001};

    // dynamic shape
    EXPECT_TRUE(clamp_test<ctype>("${BACKEND_NAME}",
                                  type,
                                  dshape,
                                  sshape,
                                  {-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8},
                                  0.2,
                                  0.6,
                                  {0.2, 0.2, 0.2, 0.2, 0.3, 0.4, 0.5, 0.6, 0.6, 0.6}));

    EXPECT_TRUE(
        clamp_test<ctype>("${BACKEND_NAME}",
                          type,
                          dshape,
                          sshape,
                          input,
                          10.0,
                          20.0,
                          {10.0, 20.0, 10.0, 20.0, 10.0, 10.0, 10.000001, 19.999999, 20.0, 20.0}));
    EXPECT_TRUE(clamp_test<ctype>(
        "${BACKEND_NAME}",
        type,
        dshape,
        sshape,
        input,
        10.0,
        pinf,
        {10.0, max, 10.0, pinf, 10.0, 10.0, 10.000001, 19.999999, 20.0, 20.000001}));
    EXPECT_TRUE(clamp_test<ctype>(
        "${BACKEND_NAME}",
        type,
        dshape,
        sshape,
        input,
        ninf,
        20.0,
        {min, 20.0, ninf, 20.0, 9.99999, 10.0, 10.000001, 19.999999, 20.0, 20.0}));
}

NGRAPH_TEST(${BACKEND_NAME}, clamp_bfloat16_dynamic)
{
    auto type = element::bf16;
    typedef bfloat16 ctype;

    auto sshape = Shape{5, 2};
    auto dshape = PartialShape::dynamic();

    auto min = numeric_limits<ctype>::min();
    auto max = numeric_limits<ctype>::max();
    auto pinf = numeric_limits<float>::infinity();
    auto ninf = -numeric_limits<float>::infinity();

    vector<ctype> input{min, max, ninf, pinf, 9.99999, 10.0, 10.000001, 19.999999, 20.0, 20.000001};

    // dynamic shape
    EXPECT_TRUE(clamp_test<ctype>("${BACKEND_NAME}",
                                  type,
                                  dshape,
                                  sshape,
                                  {-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8},
                                  0.2,
                                  0.6,
                                  {0.2, 0.2, 0.2, 0.2, 0.3, 0.4, 0.5, 0.6, 0.6, 0.6}));

    EXPECT_TRUE(
        clamp_test<ctype>("${BACKEND_NAME}",
                          type,
                          dshape,
                          sshape,
                          input,
                          10.0,
                          20.0,
                          {10.0, 20.0, 10.0, 20.0, 10.0, 10.0, 10.000001, 19.999999, 20.0, 20.0}));
    EXPECT_TRUE(clamp_test<ctype>(
        "${BACKEND_NAME}",
        type,
        dshape,
        sshape,
        input,
        10.0,
        pinf,
        {10.0, max, 10.0, pinf, 10.0, 10.0, 10.000001, 19.999999, 20.0, 20.000001}));
    EXPECT_TRUE(clamp_test<ctype>(
        "${BACKEND_NAME}",
        type,
        dshape,
        sshape,
        input,
        ninf,
        20.0,
        {min, 20.0, ninf, 20.0, 9.99999, 10.0, 10.000001, 19.999999, 20.0, 20.0}));
}
