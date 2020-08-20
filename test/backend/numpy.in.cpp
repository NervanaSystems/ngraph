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
#include <random>
#include <string>

// clang-format off
#ifdef ${BACKEND_NAME}_FLOAT_TOLERANCE_BITS
#define DEFAULT_FLOAT_TOLERANCE_BITS ${BACKEND_NAME}_FLOAT_TOLERANCE_BITS
#endif

#ifdef ${BACKEND_NAME}_DOUBLE_TOLERANCE_BITS
#define DEFAULT_DOUBLE_TOLERANCE_BITS ${BACKEND_NAME}_DOUBLE_TOLERANCE_BITS
#endif
// clang-format on

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/ndarray.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

#include <pybind11/embed.h>
#include <pybind11/numpy.h>

using namespace std;
using namespace ngraph;

namespace py = pybind11;
using namespace py::literals;

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, numpy_abc)
{
    int a = 3, b = 4;

    auto locals = py::dict("a"_a = a, "b"_a = b);
    py::exec(R"(
        c = a + b
    )",
             py::globals(),
             locals);

    auto sum = locals["c"].cast<int>();

    EXPECT_TRUE(sum == a + b);
}

NGRAPH_TEST(${BACKEND_NAME}, numpy_add_abc)
{
    Shape shape{3, 3};
    auto A = make_shared<op::v0::Parameter>(element::i32, shape);
    auto B = make_shared<op::v0::Parameter>(element::i32, shape);
    auto f = make_shared<Function>(make_shared<op::v1::Add>(A, B), ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    shared_ptr<runtime::Tensor> t_a = backend->create_tensor(element::i32, shape);
    shared_ptr<runtime::Tensor> t_b = backend->create_tensor(element::i32, shape);
    shared_ptr<runtime::Tensor> t_result = backend->create_tensor(element::i32, shape);

    test::NDArray<int32_t, 2> a = {{1, 1, 1}, {2, 2, 2}, {3, 3, 3}};
    test::NDArray<int32_t, 2> b = {{3, 3, 3}, {2, 2, 2}, {1, 1, 1}};

    copy_data(t_a, a.get_vector());
    copy_data(t_b, b.get_vector());

    auto handle = backend->compile(f);
    handle->call_with_validate({t_result}, {t_a, t_b});

    // Steps for running embedded python
    // 1. Make a copy of test input
    auto v_a = a.get_vector();
    auto v_b = b.get_vector();
    // 2. Change format to numpy friendly
    auto n_a = py::array_t<int32_t>(a.get_shape(), v_a.data());
    auto n_b = py::array_t<int32_t>(b.get_shape(), v_b.data());
    // 3. Create variable dictionary (args) to pass to python
    auto locals = py::dict("a"_a = n_a, "b"_a = n_b);
    // 4. Execute embedded python script
    py::exec(R"(
c = a + b
    )",
             py::globals(),
             locals);
    // 5. Fetch variable from python
    auto n_c = locals["c"].cast<py::array_t<int32_t>>();
    // 6. Convert python result to a vector
    auto buf = n_c.request();
    vector<int32_t> n_result;
    n_result.assign((int32_t*)buf.ptr, (int32_t*)buf.ptr + buf.size);

    EXPECT_TRUE(test::all_close(read_vector<int32_t>(t_result), n_result));
}
