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

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "ngraph/interval.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/minimum.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/op/topk.hpp"

using namespace std;
using namespace ngraph;
using ::testing::Return;

TEST(intervals, size)
{
    EXPECT_TRUE(Interval().size() > 0);
    EXPECT_TRUE(Interval(2).size() == 1);
    EXPECT_TRUE(Interval(1, 5).size() == 5);
    EXPECT_TRUE(Interval(3, 2).size() == 0);
    EXPECT_TRUE(Interval(3, 3).size() == 1);
}

TEST(intervals, contains)
{
    Interval x(3, 10);
    for (auto i = x.get_min_val(); i <= x.get_max_val(); ++i)
    {
        EXPECT_TRUE(x.contains(i));
    }
    EXPECT_FALSE(x.contains(x.get_max_val() + 1));
    EXPECT_FALSE(x.contains(x.get_min_val() - 1));
    Interval empty(1, -1);
    EXPECT_TRUE(empty.empty());
}

TEST(intervals, equals)
{
    EXPECT_TRUE(Interval(2, 5) == Interval(2, 5));
    EXPECT_FALSE(Interval(2, 5) != Interval(2, 5));
    EXPECT_FALSE(Interval(3) == Interval(5));
    EXPECT_TRUE(Interval(3) != Interval(5));
    Interval a(2);
    Interval b(a);
    EXPECT_TRUE(a == b);
    Interval c(2, 4);
    b = c;
    EXPECT_TRUE(b == c);
}

TEST(intervals, arithmetic)
{
    Interval a(7, 10);
    Interval b(1, 5);
    Interval a_plus = a;
    auto a_plus_b = a + b;
    a_plus += b;
    EXPECT_TRUE(a_plus_b == a_plus);
    Interval::value_type min_plus = numeric_limits<Interval::value_type>::max();
    Interval::value_type max_plus = numeric_limits<Interval::value_type>::min();
    auto a_minus_b = a - b;
    Interval a_minus = a;
    a_minus -= b;
    EXPECT_TRUE(a_minus_b == a_minus);
    Interval::value_type min_minus = numeric_limits<Interval::value_type>::max();
    Interval::value_type max_minus = numeric_limits<Interval::value_type>::min();
    auto a_times_b = a * b;
    Interval a_times = a;
    a_times *= b;
    EXPECT_TRUE(a_times_b == a_times);
    Interval::value_type min_times = numeric_limits<Interval::value_type>::max();
    Interval::value_type max_times = numeric_limits<Interval::value_type>::min();

    for (auto a_i = a.get_min_val(); a_i <= a.get_max_val(); ++a_i)
    {
        for (auto b_i = b.get_min_val(); b_i <= b.get_max_val(); ++b_i)
        {
            auto sum = a_i + b_i;
            EXPECT_TRUE(a_plus_b.contains(sum));
            if (sum < min_plus)
            {
                min_plus = sum;
            }
            if (sum > max_plus)
            {
                max_plus = sum;
            }
            auto minus = a_i - b_i;
            if (minus < 0)
            {
                EXPECT_FALSE(a_minus_b.contains(minus));
            }
            else
            {
                EXPECT_TRUE(a_minus_b.contains(minus));
            }
            if (minus < min_minus)
            {
                min_minus = minus;
            }
            if (minus > max_minus)
            {
                max_minus = minus;
            }
            min_minus = max(Interval::value_type(0), min_minus);

            auto times = a_i * b_i;
            EXPECT_TRUE(a_times_b.contains(times));
            if (times < min_times)
            {
                min_times = times;
            }
            if (times > max_times)
            {
                max_times = times;
            }
        }
    }
    EXPECT_TRUE(Interval(min_plus, max_plus) == a_plus_b);
    EXPECT_TRUE(Interval(min_minus, max_minus) == a_minus_b);
    EXPECT_TRUE(Interval(min_times, max_times) == a_times_b);
}

TEST(intervals, sets)
{
    Interval a(1, 5);
    Interval b(3, 7);
    Interval a_int = a;
    auto a_int_b = a & b;
    a_int &= b;
    EXPECT_TRUE(a_int_b == a_int);
    Interval::value_type min_int = numeric_limits<Interval::value_type>::max();
    Interval::value_type max_int = numeric_limits<Interval::value_type>::min();

    for (auto a_i = a.get_min_val(); a_i <= a.get_max_val(); ++a_i)
    {
        for (auto b_i = b.get_min_val(); b_i <= b.get_max_val(); ++b_i)
        {
            if (a_i == b_i)
            {
                if (a_i < min_int)
                {
                    min_int = a_i;
                }
                if (a_i > max_int)
                {
                    max_int = a_i;
                }
                EXPECT_TRUE(a_int_b.contains(a_i));
            }
        }
    }
    EXPECT_TRUE(Interval(min_int, max_int) == a_int_b);
}

TEST(intervals, topk)
{
    auto p0 = make_shared<op::Parameter>(element::i64, Shape{});
    auto p1 = make_shared<op::Parameter>(element::f32, Shape{50, 40});
    auto tk0 = make_shared<op::v1::TopK>(p1, p0, 1, "min", "none");
    // Maximum is number of elements in input
    EXPECT_EQ(40, tk0->output(0).get_partial_shape()[1].get_max_length());
    auto c = op::Constant::create<int64_t>(element::i64, Shape{}, {27});
    auto m = make_shared<op::Minimum>(c, p0);
    auto tk1 = make_shared<op::v1::TopK>(p1, m, 1, "min", "none");
    // Maximum is limited by c to 27
    EXPECT_EQ(27, tk1->output(0).get_partial_shape()[1].get_max_length());
}
