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
#include "util/type_prop.hpp"

#include <memory>

using namespace std;
using namespace ngraph;

TEST(type_prop, strided_slice_convert_mask)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 3, 4, 5, 6});
    const auto begin = op::Constant::create(element::i64, Shape{5}, {0, 1, 2, 3, 1});
    const auto end = op::Constant::create(element::i64, Shape{5}, {1, 3, 3, 5, 6});
    const auto strides = op::Constant::create(element::i64, Shape{5}, {1, 1, 1, 2, 2});

    const std::vector<int64_t> begin_mask{1, 0, 1, 0, 1};
    const std::vector<int64_t> end_mask{0, 1, 0, 1, 0};
    const std::vector<int64_t> new_axis_mask{0, 0, 0, 0, 0};
    const std::vector<int64_t> shrink_axis_mask{1, 1, 1, 1, 1};
    const std::vector<int64_t> ellipsis_mask{0, 0, 1, 0, 0};

    auto strided_slice = make_shared<op::StridedSlice>(arg,
                                                       begin,
                                                       end,
                                                       strides,
                                                       begin_mask,
                                                       end_mask,
                                                       new_axis_mask,
                                                       shrink_axis_mask,
                                                       ellipsis_mask);

    EXPECT_EQ(strided_slice->get_lower_bounds_mask(), AxisSet({0, 2, 4}));
    EXPECT_EQ(strided_slice->get_upper_bounds_mask(), AxisSet({1, 3}));
    EXPECT_EQ(strided_slice->get_new_axis(), AxisSet{});
    EXPECT_EQ(strided_slice->get_shrink_axis(), AxisSet({0, 1, 2, 3, 4}));
    EXPECT_EQ(strided_slice->get_ellipsis_mask(), AxisSet({2}));
}

TEST(type_prop, strided_slice_default_value)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 3, 4, 5, 6});
    const auto begin = op::Constant::create(element::i64, Shape{5}, {0, 1, 2, 3, 1});
    const auto end = op::Constant::create(element::i64, Shape{5}, {1, 3, 3, 5, 6});

    const std::vector<int64_t> begin_mask{1, 0, 1, 0, 1};
    const std::vector<int64_t> end_mask{0, 1, 0, 1, 0};
    const std::vector<int64_t> new_axis_mask{0, 0, 0, 1, 0};
    const std::vector<int64_t> shrink_axis_mask{1, 0, 0, 0, 0};

    auto strided_slice = make_shared<op::StridedSlice>(
        arg, begin, end, begin_mask, end_mask, new_axis_mask, shrink_axis_mask);

    auto stride_input = strided_slice->input_value(3).get_node_shared_ptr();
    auto stride_inpu_const = as_type_ptr<op::Constant>(stride_input);
    auto stride_vec = stride_inpu_const->get_vector<int64_t>();

    EXPECT_EQ(stride_vec.size(), begin_mask.size());
    EXPECT_TRUE(std::all_of(stride_vec.begin(), stride_vec.end(), [](size_t e) { return e == 1; }));
}
