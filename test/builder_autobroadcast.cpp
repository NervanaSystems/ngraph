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

#include "gtest/gtest.h"

#include "ngraph/ngraph.hpp"

using namespace std;
using namespace ngraph;

std::shared_ptr<ngraph::op::Parameter> getParamFromShape(const ngraph::Shape& shape)
{
    return std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, shape);
}

inline const ngraph::Shape& getShapeFromParam(const shared_ptr<ngraph::Node>& node)
{
    return node->get_shape();
}

// input shapes are equal so AutoBroadcast does nothing
TEST(autobroadcast, no_broadcast_equal)
{
    ngraph::Shape s2345{2, 3, 4, 5};
    auto lhs = getParamFromShape(s2345);
    auto rhs = getParamFromShape(s2345);

    auto shaped = ngraph::builder::numpy_broadcast({lhs, rhs});
    const shared_ptr<Node>& ab_lhs = shaped.first;
    const shared_ptr<Node>& ab_rhs = shaped.second;

    EXPECT_EQ(ab_lhs, lhs); // no change
    EXPECT_EQ(getShapeFromParam(ab_lhs), s2345);

    EXPECT_EQ(ab_rhs, rhs); // no change
    EXPECT_EQ(getShapeFromParam(ab_rhs), s2345);
}

// input shapes are incompatable
TEST(autobroadcast, no_broadcast_incompatable)
{
    ngraph::Shape s2345{2, 3, 4, 5};
    ngraph::Shape s6789{6, 7, 8, 9};
    auto lhs = getParamFromShape(s2345);
    auto rhs = getParamFromShape(s6789);

    EXPECT_THROW(ngraph::builder::numpy_broadcast({lhs, rhs}),
                 ngraph::builder::autobroadcast_incompatible_shapes);
}

// basic broadcast test
// 1D to 2D
// lhs broadcast to 2,3
TEST(autobroadcast, normal_broadcast_2d)
{
    ngraph::Shape s3{3};
    ngraph::Shape s23{2, 3};
    auto lhs = getParamFromShape(s3);
    auto rhs = getParamFromShape(s23);

    auto shaped = ngraph::builder::numpy_broadcast({lhs, rhs});
    const shared_ptr<Node>& ab_lhs = shaped.first;
    const shared_ptr<Node>& ab_rhs = shaped.second;

    EXPECT_NE(ab_lhs, lhs);
    EXPECT_EQ(getShapeFromParam(ab_lhs), s23);

    EXPECT_EQ(ab_rhs, rhs); // no change
    EXPECT_EQ(getShapeFromParam(ab_rhs), s23);
}

// basic broadcast test
// 2D to 3D
// lhs broadcast to 2,3,4
TEST(autobroadcast, normal_broadcast_3d)
{
    ngraph::Shape s34{3, 4};
    ngraph::Shape s234{2, 3, 4};
    auto lhs = getParamFromShape(s34);
    auto rhs = getParamFromShape(s234);

    auto shaped = ngraph::builder::numpy_broadcast({lhs, rhs});
    const shared_ptr<Node>& ab_lhs = shaped.first;
    const shared_ptr<Node>& ab_rhs = shaped.second;

    EXPECT_NE(ab_lhs, lhs);
    EXPECT_EQ(getShapeFromParam(ab_lhs), s234);

    EXPECT_EQ(ab_rhs, rhs); // no change
    EXPECT_EQ(getShapeFromParam(ab_rhs), s234);
}

// basic broadcast test
// 3D to 4D
// lhs broadcast to 2,3,4,5
TEST(autobroadcast, normal_broadcast_4d)
{
    ngraph::Shape s345{3, 4, 5};
    ngraph::Shape s2345{2, 3, 4, 5};
    auto lhs = getParamFromShape(s345);
    auto rhs = getParamFromShape(s2345);

    auto shaped = ngraph::builder::numpy_broadcast({lhs, rhs});
    const shared_ptr<Node>& ab_lhs = shaped.first;
    const shared_ptr<Node>& ab_rhs = shaped.second;

    EXPECT_NE(ab_lhs, lhs);
    EXPECT_EQ(getShapeFromParam(ab_lhs), s2345);

    EXPECT_EQ(ab_rhs, rhs); // no change
    EXPECT_EQ(getShapeFromParam(ab_rhs), s2345);
}

// basic reshape and broadcast test
// rhs reshape to 2,3,4 then
// rhs broadcast to 2,3,4,5
TEST(autobroadcast, reshape_1x_broadcast)
{
    ngraph::Shape s2345{2, 3, 4, 5};
    ngraph::Shape s2341{2, 3, 4, 1};
    auto lhs = getParamFromShape(s2345);
    auto rhs = getParamFromShape(s2341);

    auto shaped = ngraph::builder::numpy_broadcast({lhs, rhs});
    const shared_ptr<Node>& ab_lhs = shaped.first;
    const shared_ptr<Node>& ab_rhs = shaped.second;

    EXPECT_EQ(ab_lhs, lhs); // no change
    EXPECT_EQ(getShapeFromParam(ab_lhs), s2345);

    EXPECT_NE(ab_rhs, rhs);
    EXPECT_EQ(getShapeFromParam(ab_rhs), s2345);
}

// same as above, but additionally
// lhs reshape to 2,4,5 then
// lhs broadcast to 2,3,4,5
TEST(autobroadcast, reshape_2x_broadcast)
{
    ngraph::Shape s2145{2, 1, 4, 5};
    ngraph::Shape s2341{2, 3, 4, 1};
    auto lhs = getParamFromShape(s2145);
    auto rhs = getParamFromShape(s2341);

    auto shaped = ngraph::builder::numpy_broadcast({lhs, rhs});
    const shared_ptr<Node>& ab_lhs = shaped.first;
    const shared_ptr<Node>& ab_rhs = shaped.second;

    ngraph::Shape s2345{2, 3, 4, 5};

    EXPECT_NE(ab_lhs, lhs);
    EXPECT_EQ(getShapeFromParam(ab_lhs), s2345);

    EXPECT_NE(ab_rhs, rhs);
    EXPECT_EQ(getShapeFromParam(ab_rhs), s2345);
}

// matching singular dimension on axis 2
// should not require reshape of either lhs or rhs
// i.e. this should be the same as normal broadcast casse
// rhs broadcast to 2,3,1,5
TEST(autobroadcast, broadcast_with_dim1)
{
    ngraph::Shape s2315{2, 3, 1, 5};
    ngraph::Shape s315{3, 1, 5};
    auto lhs = getParamFromShape(s2315);
    auto rhs = getParamFromShape(s315);

    auto shaped = ngraph::builder::numpy_broadcast({lhs, rhs});
    const shared_ptr<Node>& ab_lhs = shaped.first;
    const shared_ptr<Node>& ab_rhs = shaped.second;

    EXPECT_EQ(ab_lhs, lhs); // no change
    EXPECT_EQ(getShapeFromParam(ab_lhs), s2315);

    EXPECT_NE(ab_rhs, rhs);
    EXPECT_EQ(getShapeFromParam(ab_rhs), s2315);
}

// reshape only test
// rhs reshape to 1,3,4,5 with no broadcast
TEST(autobroadcast, broadcast_with_leading_dim1)
{
    ngraph::Shape s1345{1, 3, 4, 5};
    ngraph::Shape s345{3, 4, 5};
    auto lhs = getParamFromShape(s1345);
    auto rhs = getParamFromShape(s345);

    auto shaped = ngraph::builder::numpy_broadcast({lhs, rhs});
    const shared_ptr<Node>& ab_lhs = shaped.first;
    const shared_ptr<Node>& ab_rhs = shaped.second;

    EXPECT_EQ(ab_lhs, lhs); // no change
    EXPECT_EQ(getShapeFromParam(ab_lhs), s1345);

    EXPECT_NE(ab_rhs, rhs);
    EXPECT_EQ(getShapeFromParam(ab_rhs), s1345);
}

TEST(autobroadcast, make_node_2_args)
{
    ngraph::Shape s21{2, 1};
    ngraph::Shape s23{2, 3};
    auto lhs = getParamFromShape(s21);
    auto rhs = getParamFromShape(s23);

    shared_ptr<Node> op = ngraph::builder::make_with_numpy_broadcast<ngraph::op::Add>(lhs, rhs);
    EXPECT_NE(op, nullptr);
}

TEST(autobroadcast, make_node_3_args)
{
    ngraph::Shape s21{2, 1};
    ngraph::Shape s23{2, 3};

    auto predicates = std::make_shared<ngraph::op::Parameter>(ngraph::element::boolean, s23);
    auto lhs = getParamFromShape(s21);
    auto rhs = getParamFromShape(s23);

    shared_ptr<Node> op =
        ngraph::builder::make_with_numpy_broadcast<ngraph::op::Select>(predicates, lhs, rhs);
    EXPECT_NE(op, nullptr);
}
