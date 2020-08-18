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

#include "gtest/gtest.h"

#include "ngraph/ngraph.hpp"

using namespace std;
using namespace ngraph;

shared_ptr<op::v0::Parameter> getParamFromShape(const Shape& shape)
{
    return make_shared<op::v0::Parameter>(element::f32, shape);
}

// input shapes are equal so AutoBroadcast does nothing
TEST(autobroadcast, no_broadcast_equal)
{
    Shape s2345{2, 3, 4, 5};
    auto lhs = getParamFromShape(s2345);
    auto rhs = getParamFromShape(s2345);

    auto shaped = builder::numpy_broadcast({lhs, rhs});
    const Output<Node> ab_lhs = shaped.first;
    const Output<Node> ab_rhs = shaped.second;

    EXPECT_EQ(ab_lhs, lhs); // no change
    EXPECT_EQ(ab_lhs.get_shape(), s2345);

    EXPECT_EQ(ab_rhs, rhs); // no change
    EXPECT_EQ(ab_rhs.get_shape(), s2345);
}

// input shapes are incompatable
TEST(autobroadcast, no_broadcast_incompatable)
{
    Shape s2345{2, 3, 4, 5};
    Shape s6789{6, 7, 8, 9};
    auto lhs = getParamFromShape(s2345);
    auto rhs = getParamFromShape(s6789);

    EXPECT_THROW(builder::numpy_broadcast({lhs, rhs}),
                 builder::numpy_autobroadcast_incompatible_shapes);
}

// basic broadcast test
// 1D to 2D
// lhs broadcast to 2,3
TEST(autobroadcast, normal_broadcast_2d)
{
    Shape s3{3};
    Shape s23{2, 3};
    auto lhs = getParamFromShape(s3);
    auto rhs = getParamFromShape(s23);

    auto shaped = builder::numpy_broadcast({lhs, rhs});
    const Output<Node> ab_lhs = shaped.first;
    const Output<Node> ab_rhs = shaped.second;

    EXPECT_NE(ab_lhs, lhs);
    EXPECT_EQ(ab_lhs.get_shape(), s23);

    EXPECT_EQ(ab_rhs, rhs); // no change
    EXPECT_EQ(ab_rhs.get_shape(), s23);
}

// basic broadcast test
// 2D to 3D
// lhs broadcast to 2,3,4
TEST(autobroadcast, normal_broadcast_3d)
{
    Shape s34{3, 4};
    Shape s234{2, 3, 4};
    auto lhs = getParamFromShape(s34);
    auto rhs = getParamFromShape(s234);

    auto shaped = builder::numpy_broadcast({lhs, rhs});
    const Output<Node> ab_lhs = shaped.first;
    const Output<Node> ab_rhs = shaped.second;

    EXPECT_NE(ab_lhs, lhs);
    EXPECT_EQ(ab_lhs.get_shape(), s234);

    EXPECT_EQ(ab_rhs, rhs); // no change
    EXPECT_EQ(ab_rhs.get_shape(), s234);
}

// basic broadcast test
// 3D to 4D
// lhs broadcast to 2,3,4,5
TEST(autobroadcast, normal_broadcast_4d)
{
    Shape s345{3, 4, 5};
    Shape s2345{2, 3, 4, 5};
    auto lhs = getParamFromShape(s345);
    auto rhs = getParamFromShape(s2345);

    auto shaped = builder::numpy_broadcast({lhs, rhs});
    const Output<Node> ab_lhs = shaped.first;
    const Output<Node> ab_rhs = shaped.second;

    EXPECT_NE(ab_lhs, lhs);
    EXPECT_EQ(ab_lhs.get_shape(), s2345);

    EXPECT_EQ(ab_rhs, rhs); // no change
    EXPECT_EQ(ab_rhs.get_shape(), s2345);
}

// basic reshape and broadcast test
// rhs reshape to 2,3,4 then
// rhs broadcast to 2,3,4,5
TEST(autobroadcast, reshape_1x_broadcast)
{
    Shape s2345{2, 3, 4, 5};
    Shape s2341{2, 3, 4, 1};
    auto lhs = getParamFromShape(s2345);
    auto rhs = getParamFromShape(s2341);

    auto shaped = builder::numpy_broadcast({lhs, rhs});
    const Output<Node> ab_lhs = shaped.first;
    const Output<Node> ab_rhs = shaped.second;

    EXPECT_EQ(ab_lhs, lhs); // no change
    EXPECT_EQ(ab_lhs.get_shape(), s2345);

    EXPECT_NE(ab_rhs, rhs);
    EXPECT_EQ(ab_rhs.get_shape(), s2345);
}

// same as above, but additionally
// lhs reshape to 2,4,5 then
// lhs broadcast to 2,3,4,5
TEST(autobroadcast, reshape_2x_broadcast)
{
    Shape s2145{2, 1, 4, 5};
    Shape s2341{2, 3, 4, 1};
    auto lhs = getParamFromShape(s2145);
    auto rhs = getParamFromShape(s2341);

    auto shaped = builder::numpy_broadcast({lhs, rhs});
    const Output<Node> ab_lhs = shaped.first;
    const Output<Node> ab_rhs = shaped.second;

    Shape s2345{2, 3, 4, 5};

    EXPECT_NE(ab_lhs, lhs);
    EXPECT_EQ(ab_lhs.get_shape(), s2345);

    EXPECT_NE(ab_rhs, rhs);
    EXPECT_EQ(ab_rhs.get_shape(), s2345);
}

// matching singular dimension on axis 2
// should not require reshape of either lhs or rhs
// i.e. this should be the same as normal broadcast casse
// rhs broadcast to 2,3,1,5
TEST(autobroadcast, broadcast_with_dim1)
{
    Shape s2315{2, 3, 1, 5};
    Shape s315{3, 1, 5};
    auto lhs = getParamFromShape(s2315);
    auto rhs = getParamFromShape(s315);

    auto shaped = builder::numpy_broadcast({lhs, rhs});
    const Output<Node> ab_lhs = shaped.first;
    const Output<Node> ab_rhs = shaped.second;

    EXPECT_EQ(ab_lhs, lhs); // no change
    EXPECT_EQ(ab_lhs.get_shape(), s2315);

    EXPECT_NE(ab_rhs, rhs);
    EXPECT_EQ(ab_rhs.get_shape(), s2315);
}

// reshape only test
// rhs reshape to 1,3,4,5 with no broadcast
TEST(autobroadcast, broadcast_with_leading_dim1)
{
    Shape s1345{1, 3, 4, 5};
    Shape s345{3, 4, 5};
    auto lhs = getParamFromShape(s1345);
    auto rhs = getParamFromShape(s345);

    auto shaped = builder::numpy_broadcast({lhs, rhs});
    const Output<Node> ab_lhs = shaped.first;
    const Output<Node> ab_rhs = shaped.second;

    EXPECT_EQ(ab_lhs, lhs); // no change
    EXPECT_EQ(ab_lhs.get_shape(), s1345);

    EXPECT_NE(ab_rhs, rhs);
    EXPECT_EQ(ab_rhs.get_shape(), s1345);
}

TEST(autobroadcast, make_node_2_args)
{
    Shape s21{2, 1};
    Shape s23{2, 3};
    auto lhs = getParamFromShape(s21);
    auto rhs = getParamFromShape(s23);

    shared_ptr<Node> op = builder::make_with_numpy_broadcast<op::v1::Add>(lhs, rhs);
    EXPECT_NE(op, nullptr);
}

TEST(autobroadcast, make_node_3_args)
{
    Shape s21{2, 1};
    Shape s23{2, 3};

    auto predicates = make_shared<op::v0::Parameter>(element::boolean, s23);
    auto lhs = getParamFromShape(s21);
    auto rhs = getParamFromShape(s23);

    shared_ptr<Node> op = builder::make_with_numpy_broadcast<op::v0::Select>(predicates, lhs, rhs);
    EXPECT_NE(op, nullptr);
}

TEST(autobroadcast, numpy_broadcast_for_matmul_op_2d)
{
    const Shape lhs{3, 1, 4, 6};
    const Shape rhs{6, 5};
    const auto lhs_node = make_shared<op::v0::Parameter>(element::f32, lhs);
    const auto rhs_node = make_shared<op::v0::Parameter>(element::f32, rhs);

    const OutputVector result = builder::numpy_broadcast_for_matmul_operation(lhs_node, rhs_node);

    EXPECT_EQ(result.at(0).get_shape(), (Shape{3, 1, 4, 6}));
    EXPECT_EQ(result.at(1).get_shape(), (Shape{3, 1, 6, 5}));
}

TEST(autobroadcast, numpy_broadcast_for_matmul_op_3d)
{
    const Shape lhs{3, 1, 4, 6};
    const Shape rhs{2, 6, 5};
    const auto lhs_node = make_shared<op::v0::Parameter>(element::f32, lhs);
    const auto rhs_node = make_shared<op::v0::Parameter>(element::f32, rhs);

    const OutputVector result = builder::numpy_broadcast_for_matmul_operation(lhs_node, rhs_node);

    EXPECT_EQ(result.at(0).get_shape(), (Shape{3, 2, 4, 6}));
    EXPECT_EQ(result.at(1).get_shape(), (Shape{3, 2, 6, 5}));
}

TEST(autobroadcast, numpy_broadcast_for_matmul_op_nop)
{
    const Shape lhs{4, 6};
    const Shape rhs{6, 5};
    const auto lhs_node = make_shared<op::v0::Parameter>(element::f32, lhs);
    const auto rhs_node = make_shared<op::v0::Parameter>(element::f32, rhs);

    const OutputVector result = builder::numpy_broadcast_for_matmul_operation(lhs_node, rhs_node);

    EXPECT_EQ(result.at(0).get_shape(), (Shape{4, 6}));
    EXPECT_EQ(result.at(1).get_shape(), (Shape{6, 5}));
}

TEST(autobroadcast, legacy_broadcast_scalar)
{
    const Shape lhs{2, 3, 4, 5};
    const Shape rhs{};
    size_t start_match_axis{3};
    const auto lhs_node = make_shared<op::v0::Parameter>(element::f32, lhs);
    const auto rhs_node = make_shared<op::v0::Parameter>(element::f32, rhs);

    const OutputVector result =
        builder::legacy_broadcast_for_binary_operation(lhs_node, rhs_node, start_match_axis);

    EXPECT_EQ(result.at(0).get_shape(), lhs);
    EXPECT_EQ(result.at(1).get_shape(), lhs);
}

TEST(autobroadcast, legacy_broadcast_1elem_tensor)
{
    const Shape lhs{2, 3, 4, 5};
    const Shape rhs{1, 1, 1};
    size_t start_match_axis{1};
    const auto lhs_node = make_shared<op::v0::Parameter>(element::f32, lhs);
    const auto rhs_node = make_shared<op::v0::Parameter>(element::f32, rhs);

    const OutputVector result =
        builder::legacy_broadcast_for_binary_operation(lhs_node, rhs_node, start_match_axis);

    EXPECT_EQ(result.at(0).get_shape(), lhs);
    EXPECT_EQ(result.at(1).get_shape(), lhs);
}

TEST(autobroadcast, legacy_broadcast_1d)
{
    const Shape lhs{2, 3, 4, 5};
    const Shape rhs{5};
    size_t start_match_axis{3};
    const auto lhs_node = make_shared<op::v0::Parameter>(element::f32, lhs);
    const auto rhs_node = make_shared<op::v0::Parameter>(element::f32, rhs);

    const OutputVector result =
        builder::legacy_broadcast_for_binary_operation(lhs_node, rhs_node, start_match_axis);

    EXPECT_EQ(result.at(0).get_shape(), lhs);
    EXPECT_EQ(result.at(1).get_shape(), lhs);
}

TEST(autobroadcast, legacy_broadcast_2d)
{
    const Shape lhs{2, 3, 4, 5};
    const Shape rhs{4, 5};
    size_t start_match_axis{2};
    const auto lhs_node = make_shared<op::v0::Parameter>(element::f32, lhs);
    const auto rhs_node = make_shared<op::v0::Parameter>(element::f32, rhs);

    const OutputVector result =
        builder::legacy_broadcast_for_binary_operation(lhs_node, rhs_node, start_match_axis);

    EXPECT_EQ(result.at(0).get_shape(), lhs);
    EXPECT_EQ(result.at(1).get_shape(), lhs);
}

TEST(autobroadcast, legacy_broadcast_2d_inside)
{
    const Shape lhs{2, 3, 4, 5};
    const Shape rhs{3, 4};
    size_t start_match_axis{1};
    const auto lhs_node = make_shared<op::v0::Parameter>(element::f32, lhs);
    const auto rhs_node = make_shared<op::v0::Parameter>(element::f32, rhs);

    const OutputVector result =
        builder::legacy_broadcast_for_binary_operation(lhs_node, rhs_node, start_match_axis);

    EXPECT_EQ(result.at(0).get_shape(), lhs);
    EXPECT_EQ(result.at(1).get_shape(), lhs);
}

TEST(autobroadcast, legacy_broadcast_1d_left)
{
    const Shape lhs{2, 3, 4, 5};
    const Shape rhs{2};
    size_t start_match_axis{0};
    const auto lhs_node = make_shared<op::v0::Parameter>(element::f32, lhs);
    const auto rhs_node = make_shared<op::v0::Parameter>(element::f32, rhs);

    const OutputVector result =
        builder::legacy_broadcast_for_binary_operation(lhs_node, rhs_node, start_match_axis);

    EXPECT_EQ(result.at(0).get_shape(), lhs);
    EXPECT_EQ(result.at(1).get_shape(), lhs);
}

TEST(autobroadcast, legacy_broadcast_identical)
{
    const Shape lhs{2, 3, 4, 5};
    size_t start_match_axis{0};
    const auto lhs_node = make_shared<op::v0::Parameter>(element::f32, lhs);
    const auto rhs_node = make_shared<op::v0::Parameter>(element::f32, lhs);

    const OutputVector result =
        builder::legacy_broadcast_for_binary_operation(lhs_node, rhs_node, start_match_axis);

    EXPECT_EQ(result.at(0).get_shape(), lhs);
    EXPECT_EQ(result.at(1).get_shape(), lhs);
}

TEST(autobroadcast, opset1_legacy_broadcast_scalar)
{
    const Shape lhs{2, 3, 4, 5};
    const Shape rhs{};
    size_t start_match_axis{3};
    const auto lhs_node = make_shared<op::v0::Parameter>(element::f32, lhs);
    const auto rhs_node = make_shared<op::v0::Parameter>(element::f32, rhs);

    const Output<Node> result = builder::opset1::legacy_broadcast_for_binary_operation(
        lhs_node, rhs_node, start_match_axis);

    EXPECT_EQ(result.get_shape(), lhs);
}

TEST(autobroadcast, opset1_legacy_broadcast_1elem_tensor)
{
    const Shape lhs{2, 3, 4, 5};
    const Shape rhs{1, 1, 1};
    size_t start_match_axis{1};
    const auto lhs_node = make_shared<op::v0::Parameter>(element::f32, lhs);
    const auto rhs_node = make_shared<op::v0::Parameter>(element::f32, rhs);

    const Output<Node> result = builder::opset1::legacy_broadcast_for_binary_operation(
        lhs_node, rhs_node, start_match_axis);

    EXPECT_EQ(result.get_shape(), lhs);
}

TEST(autobroadcast, opset1_legacy_broadcast_1d)
{
    const Shape lhs{2, 3, 4, 5};
    const Shape rhs{5};
    size_t start_match_axis{3};
    const auto lhs_node = make_shared<op::v0::Parameter>(element::f32, lhs);
    const auto rhs_node = make_shared<op::v0::Parameter>(element::f32, rhs);

    const Output<Node> result = builder::opset1::legacy_broadcast_for_binary_operation(
        lhs_node, rhs_node, start_match_axis);

    EXPECT_EQ(result.get_shape(), lhs);
}

TEST(autobroadcast, opset1_legacy_broadcast_2d)
{
    const Shape lhs{2, 3, 4, 5};
    const Shape rhs{4, 5};
    size_t start_match_axis{2};
    const auto lhs_node = make_shared<op::v0::Parameter>(element::f32, lhs);
    const auto rhs_node = make_shared<op::v0::Parameter>(element::f32, rhs);

    const Output<Node> result = builder::opset1::legacy_broadcast_for_binary_operation(
        lhs_node, rhs_node, start_match_axis);

    EXPECT_EQ(result.get_shape(), lhs);
}

TEST(autobroadcast, opset1_legacy_broadcast_2d_inside)
{
    const Shape lhs{2, 3, 4, 5};
    const Shape rhs{3, 4};
    size_t start_match_axis{1};
    const auto lhs_node = make_shared<op::v0::Parameter>(element::f32, lhs);
    const auto rhs_node = make_shared<op::v0::Parameter>(element::f32, rhs);

    const Output<Node> result = builder::opset1::legacy_broadcast_for_binary_operation(
        lhs_node, rhs_node, start_match_axis);

    EXPECT_EQ(result.get_shape(), lhs);
}

TEST(autobroadcast, opset1_legacy_broadcast_1d_left)
{
    const Shape lhs{2, 3, 4, 5};
    const Shape rhs{2};
    size_t start_match_axis{0};
    const auto lhs_node = make_shared<op::v0::Parameter>(element::f32, lhs);
    const auto rhs_node = make_shared<op::v0::Parameter>(element::f32, rhs);

    const Output<Node> result = builder::opset1::legacy_broadcast_for_binary_operation(
        lhs_node, rhs_node, start_match_axis);

    EXPECT_EQ(result.get_shape(), lhs);
}

TEST(autobroadcast, opset1_legacy_broadcast_identical)
{
    const Shape lhs{2, 3, 4, 5};
    size_t start_match_axis{0};
    const auto lhs_node = make_shared<op::v0::Parameter>(element::f32, lhs);
    const auto rhs_node = make_shared<op::v0::Parameter>(element::f32, lhs);

    const Output<Node> result = builder::opset1::legacy_broadcast_for_binary_operation(
        lhs_node, rhs_node, start_match_axis);

    EXPECT_EQ(result.get_shape(), lhs);
}

TEST(autobroadcast, axes_mapping_from_bcast_axes)
{
    const Shape output_shape{2, 3, 4, 5};
    const Shape input_shape{3, 5};
    const AxisSet broadcast_axes{0, 2};

    auto axes_mapping = builder::opset1::get_axes_mapping_output(output_shape, broadcast_axes);
    EXPECT_TRUE(axes_mapping.get_node()->is_constant());
    Shape axes_mapping_shape = as_type<op::v0::Constant>(axes_mapping.get_node())->get_shape_val();
    EXPECT_EQ(axes_mapping_shape.size(), 2);
    EXPECT_EQ(axes_mapping_shape, (Shape{1, 3}));
}

TEST(autobroadcast, axes_mapping_from_bcast_axes_scalar)
{
    const Shape output_shape{2, 3, 4, 5};
    const Shape input_shape{};
    const AxisSet broadcast_axes{0, 1, 2, 3};

    auto axes_mapping = builder::opset1::get_axes_mapping_output(output_shape, broadcast_axes);
    EXPECT_TRUE(axes_mapping.get_node()->is_constant());
    Shape axes_mapping_shape = as_type<op::v0::Constant>(axes_mapping.get_node())->get_shape_val();
    EXPECT_EQ(axes_mapping_shape.size(), 0);
    EXPECT_EQ(axes_mapping_shape, (Shape{}));
}

TEST(autobroadcast, axes_mapping_from_bcast_axes_identical)
{
    const Shape output_shape{2, 3, 4, 5};
    const Shape input_shape(output_shape);
    const AxisSet broadcast_axes{};

    auto axes_mapping = builder::opset1::get_axes_mapping_output(output_shape, broadcast_axes);
    EXPECT_TRUE(axes_mapping.get_node()->is_constant());
    Shape axes_mapping_shape = as_type<op::v0::Constant>(axes_mapping.get_node())->get_shape_val();
    EXPECT_EQ(axes_mapping_shape.size(), output_shape.size());
    EXPECT_EQ(axes_mapping_shape, (Shape{0, 1, 2, 3}));
}

TEST(autobroadcast, axes_mapping_start_match_axis)
{
    const Shape output_shape{2, 3, 4, 5};
    const Shape input_shape{3, 4};
    const std::size_t start_match_axis{1};

    auto axes_mapping =
        builder::opset1::get_axes_mapping_output(output_shape, input_shape, start_match_axis);
    EXPECT_TRUE(axes_mapping.get_node()->is_constant());
    Shape axes_mapping_shape = as_type<op::v0::Constant>(axes_mapping.get_node())->get_shape_val();
    EXPECT_EQ(axes_mapping_shape.size(), 2);
    EXPECT_EQ(axes_mapping_shape, (Shape{1, 2}));
}

TEST(autobroadcast, axes_mapping_start_match_axis_scalar)
{
    const Shape output_shape{2, 3, 4, 5};
    const Shape input_shape{};
    const std::size_t start_match_axis{4};

    auto axes_mapping =
        builder::opset1::get_axes_mapping_output(output_shape, input_shape, start_match_axis);
    EXPECT_TRUE(axes_mapping.get_node()->is_constant());
    Shape axes_mapping_shape = as_type<op::v0::Constant>(axes_mapping.get_node())->get_shape_val();
    EXPECT_EQ(axes_mapping_shape.size(), 0);
    EXPECT_EQ(axes_mapping_shape, (Shape{}));
}

TEST(autobroadcast, axes_mapping_start_match_axis_identical)
{
    const Shape output_shape{2, 3, 4, 5};
    const Shape input_shape{2, 3, 4, 5};
    const std::size_t start_match_axis{0};

    auto axes_mapping =
        builder::opset1::get_axes_mapping_output(output_shape, input_shape, start_match_axis);
    EXPECT_TRUE(axes_mapping.get_node()->is_constant());
    Shape axes_mapping_shape = as_type<op::v0::Constant>(axes_mapping.get_node())->get_shape_val();
    EXPECT_EQ(axes_mapping_shape.size(), output_shape.size());
    EXPECT_EQ(axes_mapping_shape, (Shape{0, 1, 2, 3}));
}
