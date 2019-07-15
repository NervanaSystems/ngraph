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

using namespace std;
using namespace ngraph;

TEST(type_prop, group_conv_transpose)
{
    // C x M / group x kH x kW
    auto weights = make_shared<op::Parameter>(element::f32, Shape{16, 2, 3, 3});
    // N x C x H x W
    auto data = make_shared<op::Parameter>(element::f32, Shape{1, 16, 6, 6});
    auto gct = make_shared<op::GroupConvolutionTranspose>(data,
                                                          weights,
                                                          Strides{1, 1},
                                                          Strides{1, 1},
                                                          CoordinateDiff{0, 0},
                                                          CoordinateDiff{0, 0},
                                                          CoordinateDiff{0, 0},
                                                          2);
    EXPECT_EQ(gct->get_element_type(), element::f32);
    EXPECT_EQ(gct->get_shape(), (Shape{1, 4, 8, 8}));
    EXPECT_EQ(gct->get_strides(), (Strides{1, 1}));
    EXPECT_EQ(gct->get_dilations(), (Strides{1, 1}));
    EXPECT_EQ(gct->get_padding_begin(), (CoordinateDiff{0, 0}));
    EXPECT_EQ(gct->get_padding_end(), (CoordinateDiff{0, 0}));
    EXPECT_EQ(gct->get_output_padding(), (CoordinateDiff{0, 0}));
    EXPECT_EQ(gct->get_groups(), size_t(2));
    EXPECT_EQ(gct->get_pad_type(), op::PadType::EXPLICIT);
}

TEST(type_prop, group_conv_transpose_output_shape)
{
    // N x C x H x W
    auto data = make_shared<op::Parameter>(element::f32, Shape{1, 16, 5, 5});
    // C x M / group x kH x kW
    auto weights = make_shared<op::Parameter>(element::f32, Shape{16, 2, 3, 3});
    auto gct = make_shared<op::GroupConvolutionTranspose>(
        data, weights, Strides{1, 1}, Strides{1, 1}, CoordinateDiff{0, 0}, Shape{1, 2, 3, 3}, 1);
    EXPECT_EQ(gct->get_element_type(), element::f32);
    EXPECT_EQ(gct->get_shape(), (Shape{1, 2, 3, 3}));
    EXPECT_EQ(gct->get_strides(), (Strides{1, 1}));
    EXPECT_EQ(gct->get_dilations(), (Strides{1, 1}));
    EXPECT_EQ(gct->get_padding_begin(), (CoordinateDiff{2, 2}));
    EXPECT_EQ(gct->get_padding_end(), (CoordinateDiff{2, 2}));
    EXPECT_EQ(gct->get_output_padding(), (CoordinateDiff{0, 0}));
    EXPECT_EQ(gct->get_groups(), size_t(1));
    EXPECT_EQ(gct->get_pad_type(), op::PadType::EXPLICIT);
}

TEST(type_prop, group_conv_transpose_invalid_params)
{
    // C x M / group x kH x kW
    auto weights = make_shared<op::Parameter>(element::f32, Shape{16, 20, 3, 3});
    // N x C x H x W
    auto data = make_shared<op::Parameter>(element::f32, Shape{1, 16, 5, 5});

    try
    {
        const auto gct = make_shared<op::GroupConvolutionTranspose>(data,
                                                                    weights,
                                                                    Strides{1, 1},
                                                                    Strides{1, 1},
                                                                    CoordinateDiff{2, 2},
                                                                    CoordinateDiff{2, 2},
                                                                    CoordinateDiff{0, 0},
                                                                    21);
        EXPECT_FALSE(gct.get()) << "GroupConvolutionTranspose validation did not work. "
                                   "Node was created with incorrect params.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Incorrect value of groups:"));
    }

    try
    {
        const auto gct = make_shared<op::GroupConvolutionTranspose>(data,
                                                                    weights,
                                                                    Strides{1, 1},
                                                                    Strides{1, 1},
                                                                    CoordinateDiff{2, 2},
                                                                    CoordinateDiff{2, 2},
                                                                    CoordinateDiff{0, 0},
                                                                    5);
        EXPECT_FALSE(gct.get()) << "GroupConvolutionTranspose validation did not work. "
                                   "Node was created with incorrect params.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Number of data channels not a multiple of group size."));
    }

    try
    {
        // C x M / group x kH x kW
        auto bad_weights = make_shared<op::Parameter>(element::f32, Shape{10, 20, 3, 3});
        const auto gct = make_shared<op::GroupConvolutionTranspose>(data,
                                                                    bad_weights,
                                                                    Strides{1, 1},
                                                                    Strides{1, 1},
                                                                    CoordinateDiff{2, 2},
                                                                    CoordinateDiff{2, 2},
                                                                    CoordinateDiff{0, 0},
                                                                    8);
        EXPECT_FALSE(gct.get()) << "GroupConvolutionTranspose validation did not work. "
                                   "Node was created with incorrect params.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Number of filters channels must be equal to number of ") +
                                 std::string("data channels"));
    }

    try
    {
        const auto gct = make_shared<op::GroupConvolutionTranspose>(data,
                                                                    weights,
                                                                    Strides{1, 1},
                                                                    Strides{1, 1},
                                                                    CoordinateDiff{2, 2},
                                                                    CoordinateDiff{2, 2},
                                                                    CoordinateDiff{0, 0},
                                                                    4,
                                                                    op::PadType::SAME_UPPER);
        EXPECT_FALSE(gct.get()) << "GroupConvolutionTranspose validation did not work. "
                                   "Node was created with incorrect params.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Currently only eplicit pad type is supported."));
    }

    try
    {
        const auto gct = make_shared<op::GroupConvolutionTranspose>(data,
                                                                    weights,
                                                                    Strides{1},
                                                                    Strides{1, 1},
                                                                    CoordinateDiff{2, 2},
                                                                    CoordinateDiff{2, 2},
                                                                    CoordinateDiff{0, 0},
                                                                    4);
        EXPECT_FALSE(gct.get()) << "GroupConvolutionTranspose validation did not work. "
                                   "Node was created with incorrect params.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(), std::string("Strides should be of number of input data features size."));
    }

    try
    {
        const auto gct = make_shared<op::GroupConvolutionTranspose>(data,
                                                                    weights,
                                                                    Strides{1, 1},
                                                                    Strides{1, 1, 2},
                                                                    CoordinateDiff{2, 2},
                                                                    CoordinateDiff{2, 2},
                                                                    CoordinateDiff{0, 0},
                                                                    4);
        EXPECT_FALSE(gct.get()) << "GroupConvolutionTranspose validation did not work. "
                                   "Node was created with incorrect params.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Dilations should be of number of input data features size."));
    }

    try
    {
        const auto gct = make_shared<op::GroupConvolutionTranspose>(data,
                                                                    weights,
                                                                    Strides{1, 1},
                                                                    Strides{1, 1},
                                                                    CoordinateDiff{2, 2},
                                                                    CoordinateDiff{2, 2},
                                                                    CoordinateDiff{0, 0, 1, 1},
                                                                    4);
        EXPECT_FALSE(gct.get()) << "GroupConvolutionTranspose validation did not work. "
                                   "Node was created with incorrect params.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Output padding should be of number of input data features size."));
    }
}
