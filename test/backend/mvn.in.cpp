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

NGRAPH_TEST(${BACKEND_NAME}, mvn_mean_normalization)
{
    Shape data_shape{1, 2, 5};
    auto data = make_shared<op::v0::Parameter>(element::f32, data_shape);

    auto mvn_func = make_shared<op::v0::MVN>(data, true, false);
    auto function = make_shared<Function>(OutputVector{mvn_func}, ParameterVector{data});
    auto test_case = test::NgraphTestCase(function, "${BACKEND_NAME}");
    // data
    vector<float> data_vector(shape_size(data_shape));
    iota(begin(data_vector), end(data_vector), 0);
    test_case.add_input<float>(data_vector);

    // expected result
    test_case.add_expected_output<float>(
        data_shape, vector<float>{-4.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, mvn_mean_normalization_split_channels)
{
    Shape data_shape{1, 2, 5, 1};
    auto data = make_shared<op::v0::Parameter>(element::f32, data_shape);

    auto mvn_func = make_shared<op::v0::MVN>(data, false, false);
    auto function = make_shared<Function>(OutputVector{mvn_func}, ParameterVector{data});
    auto test_case = test::NgraphTestCase(function, "${BACKEND_NAME}");
    // data
    vector<float> data_vector(shape_size(data_shape));
    iota(begin(data_vector), end(data_vector), 0);
    test_case.add_input<float>(data_vector);

    // expected result
    test_case.add_expected_output<float>({1, 2, 5, 1},
                                         vector<float>{-2, -1, 0, 1, 2, -2, -1, 0, 1, 2});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, mvn_mean_variance_normalization)
{
    Shape data_shape{1, 2, 5};
    auto data = make_shared<op::v0::Parameter>(element::f32, data_shape);

    auto mvn_func = make_shared<op::v0::MVN>(data);
    auto function = make_shared<Function>(OutputVector{mvn_func}, ParameterVector{data});
    auto test_case = test::NgraphTestCase(function, "${BACKEND_NAME}");
    // data
    vector<float> data_vector(shape_size(data_shape));
    iota(begin(data_vector), end(data_vector), 0);
    test_case.add_input<float>(data_vector);

    // expected result
    test_case.add_expected_output<float>(data_shape,
                                         vector<float>{-1.566698903055826,
                                                       -1.2185435912656424,
                                                       -0.87038827947545883,
                                                       -0.52223296768527527,
                                                       -0.17407765589509178,
                                                       0.17407765589509178,
                                                       0.52223296768527527,
                                                       0.87038827947545883,
                                                       1.2185435912656424,
                                                       1.566698903055826});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, mvn_mean_variance_normalization_split_channels)
{
    Shape data_shape{1, 2, 5};
    auto data = make_shared<op::v0::Parameter>(element::f32, data_shape);

    auto mvn_func = make_shared<op::v0::MVN>(data, false);
    auto function = make_shared<Function>(OutputVector{mvn_func}, ParameterVector{data});
    auto test_case = test::NgraphTestCase(function, "${BACKEND_NAME}");
    // data
    vector<float> data_vector(shape_size(data_shape));
    iota(begin(data_vector), end(data_vector), 0);
    test_case.add_input<float>(data_vector);

    // expected result
    test_case.add_expected_output<float>(data_shape,
                                         vector<float>{-1.4142135613730948,
                                                       -0.70710678068654742,
                                                       0.000000000000000,
                                                       0.70710678068654742,
                                                       1.4142135613730948,
                                                       -1.4142135613730948,
                                                       -0.70710678068654742,
                                                       0.000000000000000,
                                                       0.70710678068654742,
                                                       1.4142135613730948});

    test_case.run();
}
