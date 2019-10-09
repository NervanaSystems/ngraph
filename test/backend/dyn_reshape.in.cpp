//*****************************************************************************
// Copyright 2017-2019 Intel Corporation
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
#include "util/all_close_f.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, dyn_reshape)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}", true);

    auto build_graph = [&backend](bool zero_flag) {
        // Create a graph for f(x,shape) = DynReshape(x,shape,zero_flag=zero_flag).
        auto x = make_shared<op::Parameter>(element::i32, PartialShape::dynamic());
        auto shape = make_shared<op::Parameter>(element::i64, PartialShape::dynamic(1));

        auto dyn_reshape = make_shared<op::DynReshape>(x, shape, zero_flag);
        EXPECT_TRUE(dyn_reshape->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));

        auto f = make_shared<Function>(NodeVector{dyn_reshape}, ParameterVector{x, shape});

        auto ex = backend->compile(f);

        return ex;
    };

    auto t_r = backend->create_dynamic_tensor(element::i32, PartialShape::dynamic());

    auto ex_flag_off = build_graph(false);
    auto ex_flag_on = build_graph(true);

    std::vector<std::tuple<bool, Shape, std::vector<int32_t>, std::vector<int64_t>, Shape>> tests;

    tests.emplace_back(make_tuple(
        false, Shape{2, 3}, vector<int32_t>{1, 2, 3, 4, 5, 6}, vector<int64_t>{6}, Shape{6}));
    tests.emplace_back(make_tuple(
        true, Shape{2, 3}, vector<int32_t>{1, 2, 3, 4, 5, 6}, vector<int64_t>{6}, Shape{6}));
    tests.emplace_back(make_tuple(
        false, Shape{2, 3}, vector<int32_t>{1, 2, 3, 4, 5, 6}, vector<int64_t>{-1}, Shape{6}));
    tests.emplace_back(make_tuple(false,
                                  Shape{2, 3},
                                  vector<int32_t>{1, 2, 3, 4, 5, 6},
                                  vector<int64_t>{2, -1},
                                  Shape{2, 3}));
    tests.emplace_back(make_tuple(false,
                                  Shape{2, 3},
                                  vector<int32_t>{1, 2, 3, 4, 5, 6},
                                  vector<int64_t>{3, -1},
                                  Shape{3, 2}));
    tests.emplace_back(make_tuple(false,
                                  Shape{2, 3},
                                  vector<int32_t>{1, 2, 3, 4, 5, 6},
                                  vector<int64_t>{3, 2, -1},
                                  Shape{3, 2, 1}));
    tests.emplace_back(make_tuple(true,
                                  Shape{2, 3},
                                  vector<int32_t>{1, 2, 3, 4, 5, 6},
                                  vector<int64_t>{3, 2, -1},
                                  Shape{3, 2, 1}));
    tests.emplace_back(make_tuple(true,
                                  Shape{2, 3},
                                  vector<int32_t>{1, 2, 3, 4, 5, 6},
                                  vector<int64_t>{0, 0, -1},
                                  Shape{2, 3, 1}));
    tests.emplace_back(make_tuple(true,
                                  Shape{2, 3},
                                  vector<int32_t>{1, 2, 3, 4, 5, 6},
                                  vector<int64_t>{2, 0, -1},
                                  Shape{2, 3, 1}));
    tests.emplace_back(make_tuple(
        true, Shape{0, 3, 4}, vector<int32_t>{}, vector<int64_t>{3, -1, 2}, Shape{3, 0, 2}));

    for (auto& test : tests)
    {
        bool zero_flag = get<0>(test);
        const Shape& in_shape = get<1>(test);
        const std::vector<int32_t>& data = get<2>(test);
        const std::vector<int64_t>& dims = get<3>(test);
        const Shape& out_shape = get<4>(test);

        auto t_x = backend->create_tensor(element::i32, in_shape);
        auto t_shape = backend->create_tensor(element::i64, Shape{dims.size()});

        copy_data(t_x, data);
        copy_data(t_shape, dims);

        auto ex = zero_flag ? ex_flag_on : ex_flag_off;
        ex->call_with_validate({t_r}, {t_x, t_shape});

        ASSERT_EQ(t_r->get_element_type(), element::i32);
        ASSERT_EQ(t_r->get_shape(), out_shape);

        auto results = read_vector<int32_t>(t_r);

        ASSERT_EQ(results, data);
    }
}

NGRAPH_TEST(${BACKEND_NAME}, reshape_v1)
{
    auto arg = std::make_shared<op::Parameter>(element::i64, PartialShape::dynamic());
    auto pattern = make_shared<op::Parameter>(element::i64, PartialShape::dynamic(1));
    auto reshape_v1 = std::make_shared<op::v1::Reshape>(arg, pattern);

    auto f = std::make_shared<Function>(NodeVector{reshape_v1}, ParameterVector{arg, pattern});

    auto backend = runtime::Backend::create("${BACKEND_NAME}", true);
    auto ex = backend->compile(f);

    auto arg_data = vector<int64_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    auto pattern_data = vector<int64_t>{2, 2, 3};

    auto arg_tensor = backend->create_tensor(element::i64, Shape{arg_data.size()});
    auto pattern_tensor = backend->create_tensor(element::i64, Shape{pattern_data.size()});
    copy_data(arg_tensor, arg_data);
    copy_data(pattern_tensor, pattern_data);

    auto output = backend->create_dynamic_tensor(element::i64, PartialShape::dynamic());
    ex->call_with_validate({output}, {arg_tensor, pattern_tensor});

    ASSERT_EQ(output->get_element_type(), element::i64);
    EXPECT_EQ(read_vector<int64_t>(output),
              vector<int64_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}));
}
