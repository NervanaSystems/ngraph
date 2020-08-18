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

#include <numeric>
#include <vector>

#include "gtest/gtest.h"

#include "ngraph/op/mod.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "util/all_close_f.hpp"

using namespace std;
using namespace ngraph;

TEST(op_eval, mod_no_autobroadcast)
{
    auto arg0 = make_shared<op::v0::Parameter>(element::i32, PartialShape::dynamic());
    auto arg1 = make_shared<op::v0::Parameter>(element::i32, PartialShape::dynamic());
    auto mod = make_shared<op::v1::Mod>(arg0, arg1, op::AutoBroadcastType::NONE);
    auto fun = make_shared<Function>(OutputVector{mod}, ParameterVector{arg0, arg1});

    // inner vector contains shapes for arg0, arg1, mod expected result
    std::vector<std::vector<Shape>> shapes{{Shape{}, Shape{}, Shape{}},
                                           {Shape{2}, Shape{2}, Shape{2}},
                                           {Shape{2, 3}, Shape{2, 3}, Shape{2, 3}}};

    std::vector<std::vector<int32_t>> arg0_inputs{{123}, {0, 18}, {-35, -24, -9, 0, 28, 36}};
    std::vector<std::vector<int32_t>> arg1_inputs{{12}, {12, 9}, {-12, 10, -9, -5, -38, -3}};
    std::vector<std::vector<int32_t>> expected_result{{3}, {0, 0}, {-11, -4, 0, 0, 28, 0}};

    for (size_t i = 0; i < arg0_inputs.size(); i++)
    {
        auto result = make_shared<HostTensor>();
        ASSERT_TRUE(
            fun->evaluate({result},
                          {make_host_tensor<element::Type_t::i32>(shapes[i][0], arg0_inputs[i]),
                           make_host_tensor<element::Type_t::i32>(shapes[i][1], arg1_inputs[i])}));
        EXPECT_EQ(result->get_shape(), (shapes[i][2]));
        ASSERT_EQ(read_vector<int32_t>(result), expected_result[i]);
    }
}

TEST(op_eval, mod_autobroadcast_numpy)
{
    auto arg0 = make_shared<op::v0::Parameter>(element::i32, PartialShape::dynamic());
    auto arg1 = make_shared<op::v0::Parameter>(element::i32, PartialShape::dynamic());
    auto mod = make_shared<op::v1::Mod>(arg0, arg1, op::AutoBroadcastType::NUMPY);
    auto fun = make_shared<Function>(OutputVector{mod}, ParameterVector{arg0, arg1});

    // inner vector contains shapes for arg0, arg1, mod expected result
    std::vector<std::vector<Shape>> shapes{{Shape{}, Shape{2, 2}, Shape{2, 2}},
                                           {Shape{2, 3}, Shape{}, Shape{2, 3}},
                                           {Shape{3, 2}, Shape{2}, Shape{3, 2}},
                                           {Shape{2, 1}, Shape{2, 1}, Shape{2, 1}},
                                           {Shape{3, 1}, Shape{2}, Shape{3, 2}},
                                           {Shape{2, 3, 1, 1}, Shape{2, 2}, Shape{2, 3, 2, 2}}};

    std::vector<std::vector<int32_t>> arg0_inputs{{18},
                                                  {11, 12, 13, 14, 15, 16},
                                                  {21, 23, 25, 26, 32, 48},
                                                  {7, 8},
                                                  {30, 31, 32},
                                                  {40, 41, 42, 43, 44, 45}};
    std::vector<std::vector<int32_t>> arg1_inputs{
        {3, 5, 7, 10}, {7}, {5, 12}, {2, 3}, {6, 7}, {5, 6, 7, 8}};
    std::vector<std::vector<int32_t>> expected_result{
        {0, 3, 4, 8},
        {4, 5, 6, 0, 1, 2},
        {1, 11, 0, 2, 2, 0},
        {1, 2},
        {0, 2, 1, 3, 2, 4},
        {0, 4, 5, 0, 1, 5, 6, 1, 2, 0, 0, 2, 3, 1, 1, 3, 4, 2, 2, 4, 0, 3, 3, 5}};

    for (size_t i = 0; i < arg0_inputs.size(); i++)
    {
        auto result = make_shared<HostTensor>();
        ASSERT_TRUE(
            fun->evaluate({result},
                          {make_host_tensor<element::Type_t::i32>(shapes[i][0], arg0_inputs[i]),
                           make_host_tensor<element::Type_t::i32>(shapes[i][1], arg1_inputs[i])}));
        EXPECT_EQ(result->get_shape(), (shapes[i][2]));
        ASSERT_EQ(read_vector<int32_t>(result), expected_result[i]);
    }
}
