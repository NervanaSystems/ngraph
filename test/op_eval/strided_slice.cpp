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

#include <vector>

#include "gtest/gtest.h"

#include "ngraph/op/strided_slice.hpp"
#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/validation_util.hpp"
#include "util/test_tools.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(op_eval, strided_slice)
{
    auto A_shape = Shape{3, 2, 3};
    auto A = make_shared<op::Parameter>(element::i64, A_shape);
    auto begin = make_shared<op::Parameter>(element::i64, Shape{3});
    auto end = make_shared<op::Parameter>(element::i64, Shape{3});
    auto strides = make_shared<op::Parameter>(element::i64, Shape{3});
    auto r = make_shared<op::v1::StridedSlice>(A,
                                               begin,
                                               end,
                                               strides,
                                               vector<int64_t>(3, 0),
                                               vector<int64_t>(3, 0),
                                               vector<int64_t>(3, 0),
                                               vector<int64_t>(3, 0),
                                               vector<int64_t>(3, 0));
    auto f = make_shared<Function>(r, ParameterVector{A, begin, end, strides});

    std::vector<int64_t> A_vec(3 * 2 * 3);
    std::iota(A_vec.begin(), A_vec.end(), 0);
    std::vector<int64_t> begin_vec{1, 0, 0};
    std::vector<int64_t> end_vec{2, 1, 3};
    std::vector<int64_t> strides_vec{1, 1, 1};

    std::vector<int64_t> expected{6, 7, 8};
    auto result = make_shared<HostTensor>();

    ASSERT_TRUE(f->evaluate({result},
                            {make_host_tensor<element::Type_t::i64>(A_shape, A_vec),
                             make_host_tensor<element::Type_t::i64>(Shape{3}, begin_vec),
                             make_host_tensor<element::Type_t::i64>(Shape{3}, end_vec),
                             make_host_tensor<element::Type_t::i64>(Shape{3}, strides_vec)}));
    EXPECT_EQ(result->get_element_type(), element::i64);
    EXPECT_EQ(read_vector<int64_t>(result), expected);
}