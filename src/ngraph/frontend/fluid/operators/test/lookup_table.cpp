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

#undef IN_NGRAPH_LIBRARY
#include "gtest/gtest.h"
#include "ngraph/frontend/fluid/operators/lookup_table.hpp"
#include "ngraph/ngraph.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/ndarray.hpp"
#include "util/random.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

static std::mt19937_64 random_generator;

using namespace std;
using namespace ngraph;

static string s_manifest = "test.manifest";

NGRAPH_TEST(CPU, fluid_lookup_table_v2)
{
    Shape params_shape{3, 2};
    Shape indices_shape{2, 2, 3, 4};
    Shape out_shape{2, 2, 3, 4, 2};
    auto P = make_shared<op::Parameter>(element::u8, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G = make_shared<fluid::LookupTable2>(P, I, -1);
    auto f = make_shared<Function>(G, ParameterVector{P, I});

    auto backend = runtime::Backend::create("CPU");

    // Create some tensors for input/output
    auto p = backend->create_tensor(element::u8, params_shape);
    copy_data(p, vector<uint8_t>{10, 11, 20, 21, 30, 31});
    auto i = backend->create_tensor(element::i32, indices_shape);
    copy_data(i, vector<int32_t>{0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2,
                                 0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2,
                                 0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2});
    auto result = backend->create_tensor(element::u8, out_shape);

    auto c = backend->compile(f);
    c->call_with_validate({result}, {p, i});
    EXPECT_TRUE(test::all_close(
        (vector<uint8_t>{10, 11, 20, 21, 20, 21, 30, 31, 10, 11, 20, 21, 20, 21, 30, 31,
                         10, 11, 20, 21, 20, 21, 30, 31, 10, 11, 20, 21, 20, 21, 30, 31,
                         10, 11, 20, 21, 20, 21, 30, 31, 10, 11, 20, 21, 20, 21, 30, 31,
                         10, 11, 20, 21, 20, 21, 30, 31, 10, 11, 20, 21, 20, 21, 30, 31,
                         10, 11, 20, 21, 20, 21, 30, 31, 10, 11, 20, 21, 20, 21, 30, 31,
                         10, 11, 20, 21, 20, 21, 30, 31, 10, 11, 20, 21, 20, 21, 30, 31}),
        read_vector<uint8_t>(result)));
}
