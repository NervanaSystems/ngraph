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

#include <memory>
#include <string>

#include "gtest/gtest.h"

#include "ngraph/ngraph.hpp"
#include "util/ndarray.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

TEST(coordinate, shape2d)
{
    auto ct = CoordinateTransform({2, 3});
    auto it = ct.begin();
    EXPECT_EQ(*it++, Coordinate({0, 0}));
    EXPECT_EQ(*it++, Coordinate({0, 1}));
    EXPECT_EQ(*it++, Coordinate({0, 2}));
    EXPECT_EQ(*it++, Coordinate({1, 0}));
    EXPECT_EQ(*it++, Coordinate({1, 1}));
    EXPECT_EQ(*it++, Coordinate({1, 2}));
    EXPECT_TRUE(it == ct.end());
}

TEST(coordinate, shape3d)
{
    auto ct = CoordinateTransform({2, 3, 4});
    auto it = ct.begin();
    EXPECT_EQ(*it++, Coordinate({0, 0, 0}));
    EXPECT_EQ(*it++, Coordinate({0, 0, 1}));
    EXPECT_EQ(*it++, Coordinate({0, 0, 2}));
    EXPECT_EQ(*it++, Coordinate({0, 0, 3}));
    EXPECT_EQ(*it++, Coordinate({0, 1, 0}));
    EXPECT_EQ(*it++, Coordinate({0, 1, 1}));
    EXPECT_EQ(*it++, Coordinate({0, 1, 2}));
    EXPECT_EQ(*it++, Coordinate({0, 1, 3}));
    EXPECT_EQ(*it++, Coordinate({0, 2, 0}));
    EXPECT_EQ(*it++, Coordinate({0, 2, 1}));
    EXPECT_EQ(*it++, Coordinate({0, 2, 2}));
    EXPECT_EQ(*it++, Coordinate({0, 2, 3}));
    EXPECT_EQ(*it++, Coordinate({1, 0, 0}));
    EXPECT_EQ(*it++, Coordinate({1, 0, 1}));
    EXPECT_EQ(*it++, Coordinate({1, 0, 2}));
    EXPECT_EQ(*it++, Coordinate({1, 0, 3}));
    EXPECT_EQ(*it++, Coordinate({1, 1, 0}));
    EXPECT_EQ(*it++, Coordinate({1, 1, 1}));
    EXPECT_EQ(*it++, Coordinate({1, 1, 2}));
    EXPECT_EQ(*it++, Coordinate({1, 1, 3}));
    EXPECT_EQ(*it++, Coordinate({1, 2, 0}));
    EXPECT_EQ(*it++, Coordinate({1, 2, 1}));
    EXPECT_EQ(*it++, Coordinate({1, 2, 2}));
    EXPECT_EQ(*it++, Coordinate({1, 2, 3}));
    EXPECT_TRUE(it == ct.end());
}
