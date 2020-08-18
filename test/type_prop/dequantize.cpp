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
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(type_prop, dequantize_f32_from_i8_nchw_per_channel_ok)
{
    Shape batch_shape{64, 3, 480, 640};
    Shape scale_shape{3};
    Shape zero_point_shape{3};
    element::Type unquantized_type = element::f32;
    element::Type quantized_type = element::i8;
    element::Type batch_type = quantized_type;
    element::Type scale_type = unquantized_type;
    element::Type zero_point_type = quantized_type;
    AxisSet axes{1};

    auto batch = make_shared<op::v0::Parameter>(batch_type, batch_shape);
    auto scale = make_shared<op::v0::Parameter>(scale_type, scale_shape);
    auto zero_point = make_shared<op::v0::Parameter>(zero_point_type, zero_point_shape);
    auto quant = make_shared<op::v0::Dequantize>(batch, scale, zero_point, unquantized_type, axes);

    ASSERT_EQ(quant->get_output_element_type(0), unquantized_type);
    ASSERT_EQ(quant->get_output_shape(0), batch_shape);
}

TEST(type_prop, dequantize_f32_from_i8_nchw_per_image_ok)
{
    Shape batch_shape{64, 3, 480, 640};
    Shape scale_shape{64};
    Shape zero_point_shape{64};
    element::Type unquantized_type = element::f32;
    element::Type quantized_type = element::i8;
    element::Type batch_type = quantized_type;
    element::Type scale_type = unquantized_type;
    element::Type zero_point_type = quantized_type;
    AxisSet axes{0};

    auto batch = make_shared<op::v0::Parameter>(batch_type, batch_shape);
    auto scale = make_shared<op::v0::Parameter>(scale_type, scale_shape);
    auto zero_point = make_shared<op::v0::Parameter>(zero_point_type, zero_point_shape);
    auto quant = make_shared<op::v0::Dequantize>(batch, scale, zero_point, unquantized_type, axes);

    ASSERT_EQ(quant->get_output_element_type(0), unquantized_type);
    ASSERT_EQ(quant->get_output_shape(0), batch_shape);
}

TEST(type_prop, dequantize_f32_from_i8_nchw_per_row_ok)
{
    Shape batch_shape{64, 3, 480, 640};
    Shape scale_shape{480};
    Shape zero_point_shape{480};
    element::Type unquantized_type = element::f32;
    element::Type quantized_type = element::i8;
    element::Type batch_type = quantized_type;
    element::Type scale_type = unquantized_type;
    element::Type zero_point_type = quantized_type;
    AxisSet axes{2};

    auto batch = make_shared<op::v0::Parameter>(batch_type, batch_shape);
    auto scale = make_shared<op::v0::Parameter>(scale_type, scale_shape);
    auto zero_point = make_shared<op::v0::Parameter>(zero_point_type, zero_point_shape);
    auto quant = make_shared<op::v0::Dequantize>(batch, scale, zero_point, unquantized_type, axes);

    ASSERT_EQ(quant->get_output_element_type(0), unquantized_type);
    ASSERT_EQ(quant->get_output_shape(0), batch_shape);
}

TEST(type_prop, dequantize_f32_from_i8_nchw_per_image_channel_ok)
{
    Shape batch_shape{64, 3, 480, 640};
    Shape scale_shape{64, 3};
    Shape zero_point_shape{64, 3};
    element::Type unquantized_type = element::f32;
    element::Type quantized_type = element::i8;
    element::Type batch_type = quantized_type;
    element::Type scale_type = unquantized_type;
    element::Type zero_point_type = quantized_type;
    AxisSet axes{0, 1};

    auto batch = make_shared<op::v0::Parameter>(batch_type, batch_shape);
    auto scale = make_shared<op::v0::Parameter>(scale_type, scale_shape);
    auto zero_point = make_shared<op::v0::Parameter>(zero_point_type, zero_point_shape);
    auto quant = make_shared<op::v0::Dequantize>(batch, scale, zero_point, unquantized_type, axes);

    ASSERT_EQ(quant->get_output_element_type(0), unquantized_type);
    ASSERT_EQ(quant->get_output_shape(0), batch_shape);
}

TEST(type_prop, dequantize_f32_from_i8_nchw_whole_batch_ok)
{
    Shape batch_shape{64, 3, 480, 640};
    Shape scale_shape{};
    Shape zero_point_shape{};
    element::Type unquantized_type = element::f32;
    element::Type quantized_type = element::i8;
    element::Type batch_type = quantized_type;
    element::Type scale_type = unquantized_type;
    element::Type zero_point_type = quantized_type;
    AxisSet axes{};

    auto batch = make_shared<op::v0::Parameter>(batch_type, batch_shape);
    auto scale = make_shared<op::v0::Parameter>(scale_type, scale_shape);
    auto zero_point = make_shared<op::v0::Parameter>(zero_point_type, zero_point_shape);
    auto quant = make_shared<op::v0::Dequantize>(batch, scale, zero_point, unquantized_type, axes);

    ASSERT_EQ(quant->get_output_element_type(0), unquantized_type);
    ASSERT_EQ(quant->get_output_shape(0), batch_shape);
}

TEST(type_prop, dequantize_f64_from_i8_ok)
{
    Shape batch_shape{64, 3, 480, 640};
    Shape scale_shape{};
    Shape zero_point_shape{};
    element::Type unquantized_type = element::f64;
    element::Type quantized_type = element::i8;
    element::Type batch_type = quantized_type;
    element::Type scale_type = unquantized_type;
    element::Type zero_point_type = quantized_type;
    AxisSet axes{};

    auto batch = make_shared<op::v0::Parameter>(batch_type, batch_shape);
    auto scale = make_shared<op::v0::Parameter>(scale_type, scale_shape);
    auto zero_point = make_shared<op::v0::Parameter>(zero_point_type, zero_point_shape);
    auto quant = make_shared<op::v0::Dequantize>(batch, scale, zero_point, unquantized_type, axes);

    ASSERT_EQ(quant->get_output_element_type(0), unquantized_type);
    ASSERT_EQ(quant->get_output_shape(0), batch_shape);
}

TEST(type_prop, dequantize_f64_to_u8_ok)
{
    Shape batch_shape{64, 3, 480, 640};
    Shape scale_shape{};
    Shape zero_point_shape{};
    element::Type unquantized_type = element::f64;
    element::Type quantized_type = element::u8;
    element::Type batch_type = quantized_type;
    element::Type scale_type = unquantized_type;
    element::Type zero_point_type = quantized_type;
    AxisSet axes{};

    auto batch = make_shared<op::v0::Parameter>(batch_type, batch_shape);
    auto scale = make_shared<op::v0::Parameter>(scale_type, scale_shape);
    auto zero_point = make_shared<op::v0::Parameter>(zero_point_type, zero_point_shape);
    auto quant = make_shared<op::v0::Dequantize>(batch, scale, zero_point, unquantized_type, axes);

    ASSERT_EQ(quant->get_output_element_type(0), unquantized_type);
    ASSERT_EQ(quant->get_output_shape(0), batch_shape);
}

TEST(type_prop, dequantize_i8_from_u8_fails)
{
    Shape batch_shape{64, 3, 480, 640};
    Shape scale_shape{};
    Shape zero_point_shape{};
    element::Type unquantized_type = element::i8;
    element::Type quantized_type = element::u8;
    element::Type batch_type = quantized_type;
    element::Type scale_type = unquantized_type;
    element::Type zero_point_type = quantized_type;
    AxisSet axes{};

    auto batch = make_shared<op::v0::Parameter>(batch_type, batch_shape);
    auto scale = make_shared<op::v0::Parameter>(scale_type, scale_shape);
    auto zero_point = make_shared<op::v0::Parameter>(zero_point_type, zero_point_shape);

    try
    {
        auto quant =
            make_shared<op::v0::Dequantize>(batch, scale, zero_point, unquantized_type, axes);
        FAIL() << "Attempt to dequantize to non-floating point type not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Output element type (i8) must be a floating point type");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, dequantize_f32_from_f32_fails)
{
    Shape batch_shape{64, 3, 480, 640};
    Shape scale_shape{};
    Shape zero_point_shape{};
    element::Type unquantized_type = element::f32;
    element::Type quantized_type = element::f32;
    element::Type batch_type = quantized_type;
    element::Type scale_type = unquantized_type;
    element::Type zero_point_type = quantized_type;
    AxisSet axes{};

    auto batch = make_shared<op::v0::Parameter>(batch_type, batch_shape);
    auto scale = make_shared<op::v0::Parameter>(scale_type, scale_shape);
    auto zero_point = make_shared<op::v0::Parameter>(zero_point_type, zero_point_shape);

    try
    {
        auto quant =
            make_shared<op::v0::Dequantize>(batch, scale, zero_point, unquantized_type, axes);
        FAIL() << "Attempt to dequantize from non-quantized type not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Zero point / input element type (f32) must be a quantized type");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, dequantize_batch_zero_point_type_mismatch_fails)
{
    Shape batch_shape{64, 3, 480, 640};
    Shape scale_shape{};
    Shape zero_point_shape{};
    element::Type unquantized_type = element::f32;
    element::Type quantized_type = element::i8;
    element::Type batch_type = quantized_type;
    element::Type scale_type = unquantized_type;
    element::Type zero_point_type = element::u8;
    AxisSet axes{};

    auto batch = make_shared<op::v0::Parameter>(batch_type, batch_shape);
    auto scale = make_shared<op::v0::Parameter>(scale_type, scale_shape);
    auto zero_point = make_shared<op::v0::Parameter>(zero_point_type, zero_point_shape);

    try
    {
        auto quant =
            make_shared<op::v0::Dequantize>(batch, scale, zero_point, unquantized_type, axes);
        FAIL() << "Mismatch of batch and zero point element types not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Zero point element type (u8) must match input element type (i8)");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, dequantize_scale_type_mismatch_fails)
{
    Shape batch_shape{64, 3, 480, 640};
    Shape scale_shape{};
    Shape zero_point_shape{};
    element::Type unquantized_type = element::f32;
    element::Type quantized_type = element::i8;
    element::Type batch_type = quantized_type;
    element::Type scale_type = element::f64;
    element::Type zero_point_type = quantized_type;
    AxisSet axes{};

    auto batch = make_shared<op::v0::Parameter>(batch_type, batch_shape);
    auto scale = make_shared<op::v0::Parameter>(scale_type, scale_shape);
    auto zero_point = make_shared<op::v0::Parameter>(zero_point_type, zero_point_shape);

    try
    {
        auto quant =
            make_shared<op::v0::Dequantize>(batch, scale, zero_point, unquantized_type, axes);
        FAIL() << "Mismatch of scale element type with scale argument not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Scale element type (f64) must match output element type (f32)");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, dequantize_oob_axis_fails)
{
    Shape batch_shape{64, 3, 480, 640};
    Shape scale_shape{320};
    Shape zero_point_shape{320};
    element::Type unquantized_type = element::f32;
    element::Type quantized_type = element::i8;
    element::Type batch_type = quantized_type;
    element::Type scale_type = unquantized_type;
    element::Type zero_point_type = quantized_type;
    AxisSet axes{3, 4};

    auto batch = make_shared<op::v0::Parameter>(batch_type, batch_shape);
    auto scale = make_shared<op::v0::Parameter>(scale_type, scale_shape);
    auto zero_point = make_shared<op::v0::Parameter>(zero_point_type, zero_point_shape);

    try
    {
        auto quant =
            make_shared<op::v0::Dequantize>(batch, scale, zero_point, unquantized_type, axes);
        FAIL() << "Out-of-bounds quantization axis not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Quantization axis (4) must be less than input shape rank (4)");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, dequantize_scale_shape_mismatch_same_rank_fails)
{
    Shape batch_shape{64, 3, 480, 640};
    Shape scale_shape{64, 4};
    Shape zero_point_shape{64, 3};
    element::Type unquantized_type = element::f32;
    element::Type quantized_type = element::i8;
    element::Type batch_type = quantized_type;
    element::Type scale_type = unquantized_type;
    element::Type zero_point_type = quantized_type;
    AxisSet axes{0, 1};

    auto batch = make_shared<op::v0::Parameter>(batch_type, batch_shape);
    auto scale = make_shared<op::v0::Parameter>(scale_type, scale_shape);
    auto zero_point = make_shared<op::v0::Parameter>(zero_point_type, zero_point_shape);

    try
    {
        auto quant =
            make_shared<op::v0::Dequantize>(batch, scale, zero_point, unquantized_type, axes);
        FAIL() << "Mismatch of scale argument shape with required shape not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Scale shape ({64,4}) and zero point shape ({64,3}) must match");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, dequantize_scale_shape_mismatch_different_rank_fails)
{
    Shape batch_shape{64, 3, 480, 640};
    Shape scale_shape{64, 3, 2};
    Shape zero_point_shape{64, 3};
    element::Type unquantized_type = element::f32;
    element::Type quantized_type = element::i8;
    element::Type batch_type = quantized_type;
    element::Type scale_type = unquantized_type;
    element::Type zero_point_type = quantized_type;
    AxisSet axes{0, 1};

    auto batch = make_shared<op::v0::Parameter>(batch_type, batch_shape);
    auto scale = make_shared<op::v0::Parameter>(scale_type, scale_shape);
    auto zero_point = make_shared<op::v0::Parameter>(zero_point_type, zero_point_shape);

    try
    {
        auto quant =
            make_shared<op::v0::Dequantize>(batch, scale, zero_point, unquantized_type, axes);
        FAIL() << "Mismatch of scale argument shape with required shape not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Scale shape ({64,3,2}) and zero point shape ({64,3}) must match");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, dequantize_zero_point_shape_mismatch_same_rank_fails)
{
    Shape batch_shape{64, 3, 480, 640};
    Shape scale_shape{64, 3};
    Shape zero_point_shape{64, 4};
    element::Type unquantized_type = element::f32;
    element::Type quantized_type = element::i8;
    element::Type batch_type = quantized_type;
    element::Type scale_type = unquantized_type;
    element::Type zero_point_type = quantized_type;
    AxisSet axes{0, 1};

    auto batch = make_shared<op::v0::Parameter>(batch_type, batch_shape);
    auto scale = make_shared<op::v0::Parameter>(scale_type, scale_shape);
    auto zero_point = make_shared<op::v0::Parameter>(zero_point_type, zero_point_shape);

    try
    {
        auto quant =
            make_shared<op::v0::Dequantize>(batch, scale, zero_point, unquantized_type, axes);
        FAIL() << "Mismatch of zero point argument shape with required shape not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Scale shape ({64,3}) and zero point shape ({64,4}) must match");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, dequantize_zero_point_shape_mismatch_different_rank_fails)
{
    Shape batch_shape{64, 3, 480, 640};
    Shape scale_shape{64, 3};
    Shape zero_point_shape{64, 3, 2};
    element::Type unquantized_type = element::f32;
    element::Type quantized_type = element::i8;
    element::Type batch_type = quantized_type;
    element::Type scale_type = unquantized_type;
    element::Type zero_point_type = quantized_type;
    AxisSet axes{0, 1};

    auto batch = make_shared<op::v0::Parameter>(batch_type, batch_shape);
    auto scale = make_shared<op::v0::Parameter>(scale_type, scale_shape);
    auto zero_point = make_shared<op::v0::Parameter>(zero_point_type, zero_point_shape);

    try
    {
        auto quant =
            make_shared<op::v0::Dequantize>(batch, scale, zero_point, unquantized_type, axes);
        FAIL() << "Mismatch of zero point argument shape with required shape not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Scale shape ({64,3}) and zero point shape ({64,3,2}) must match");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, dequantize_partial_all_rank_dynamic_ok)
{
    PartialShape batch_shape{PartialShape::dynamic()};
    PartialShape scale_shape{PartialShape::dynamic()};
    PartialShape zero_point_shape{PartialShape::dynamic()};
    element::Type unquantized_type = element::f32;
    element::Type quantized_type = element::i8;
    element::Type batch_type = quantized_type;
    element::Type scale_type = unquantized_type;
    element::Type zero_point_type = quantized_type;
    AxisSet axes{0, 1, 2000};

    auto batch = make_shared<op::v0::Parameter>(batch_type, batch_shape);
    auto scale = make_shared<op::v0::Parameter>(scale_type, scale_shape);
    auto zero_point = make_shared<op::v0::Parameter>(zero_point_type, zero_point_shape);
    auto quant = make_shared<op::v0::Dequantize>(batch, scale, zero_point, unquantized_type, axes);

    ASSERT_EQ(quant->get_output_element_type(0), unquantized_type);
    ASSERT_TRUE(quant->get_output_partial_shape(0).rank().is_dynamic());
}

TEST(type_prop,
     dequantize_partial_input_rank_dynamic_scale_rank_static_dynamic_zero_point_rank_dynamic_ok)
{
    PartialShape batch_shape{PartialShape::dynamic()};
    PartialShape scale_shape{64, Dimension::dynamic(), 96};
    PartialShape zero_point_shape{PartialShape::dynamic()};
    element::Type unquantized_type = element::f32;
    element::Type quantized_type = element::i8;
    element::Type batch_type = quantized_type;
    element::Type scale_type = unquantized_type;
    element::Type zero_point_type = quantized_type;
    AxisSet axes{0, 1, 2000};

    auto batch = make_shared<op::v0::Parameter>(batch_type, batch_shape);
    auto scale = make_shared<op::v0::Parameter>(scale_type, scale_shape);
    auto zero_point = make_shared<op::v0::Parameter>(zero_point_type, zero_point_shape);
    auto quant = make_shared<op::v0::Dequantize>(batch, scale, zero_point, unquantized_type, axes);

    ASSERT_EQ(quant->get_output_element_type(0), unquantized_type);
    ASSERT_TRUE(quant->get_output_partial_shape(0).rank().is_dynamic());
}

TEST(
    type_prop,
    dequantize_partial_input_rank_dynamic_scale_rank_static_dynamic_zero_point_rank_dynamic_axis_count_inconsistent)
{
    PartialShape batch_shape{PartialShape::dynamic()};
    PartialShape scale_shape{64, Dimension::dynamic(), 96};
    PartialShape zero_point_shape{PartialShape::dynamic()};
    element::Type unquantized_type = element::f32;
    element::Type quantized_type = element::i8;
    element::Type batch_type = quantized_type;
    element::Type scale_type = unquantized_type;
    element::Type zero_point_type = quantized_type;
    AxisSet axes{0, 1};

    auto batch = make_shared<op::v0::Parameter>(batch_type, batch_shape);
    auto scale = make_shared<op::v0::Parameter>(scale_type, scale_shape);
    auto zero_point = make_shared<op::v0::Parameter>(zero_point_type, zero_point_shape);

    try
    {
        auto quant =
            make_shared<op::v0::Dequantize>(batch, scale, zero_point, unquantized_type, axes);
        FAIL() << "Mismatch of scale / zero point rank with axis count not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            "Scale / zero point rank (3) does not match the number of quantization axes (2)");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(
    type_prop,
    dequantize_partial_input_rank_dynamic_scale_rank_static_dynamic_zero_point_rank_static_dynamic_ok)
{
    PartialShape batch_shape{PartialShape::dynamic()};
    PartialShape scale_shape{64, Dimension::dynamic(), 96, Dimension::dynamic()};
    PartialShape zero_point_shape{64, 22, Dimension::dynamic(), Dimension::dynamic()};
    element::Type unquantized_type = element::f32;
    element::Type quantized_type = element::i8;
    element::Type batch_type = quantized_type;
    element::Type scale_type = unquantized_type;
    element::Type zero_point_type = quantized_type;
    AxisSet axes{0, 1, 5, 88};

    auto batch = make_shared<op::v0::Parameter>(batch_type, batch_shape);
    auto scale = make_shared<op::v0::Parameter>(scale_type, scale_shape);
    auto zero_point = make_shared<op::v0::Parameter>(zero_point_type, zero_point_shape);
    auto quant = make_shared<op::v0::Dequantize>(batch, scale, zero_point, unquantized_type, axes);

    ASSERT_EQ(quant->get_output_element_type(0), unquantized_type);
    ASSERT_TRUE(quant->get_output_partial_shape(0).rank().is_dynamic());
}

TEST(
    type_prop,
    dequantize_partial_input_rank_dynamic_scale_rank_static_dynamic_zero_point_rank_static_dynamic_ranks_inconsistent)
{
    PartialShape batch_shape{PartialShape::dynamic()};
    PartialShape scale_shape{64, Dimension::dynamic(), 96, Dimension::dynamic()};
    PartialShape zero_point_shape{64, 22, Dimension::dynamic(), Dimension::dynamic(), 3};
    element::Type unquantized_type = element::f32;
    element::Type quantized_type = element::i8;
    element::Type batch_type = quantized_type;
    element::Type scale_type = unquantized_type;
    element::Type zero_point_type = quantized_type;
    AxisSet axes{0, 1, 5, 88};

    auto batch = make_shared<op::v0::Parameter>(batch_type, batch_shape);
    auto scale = make_shared<op::v0::Parameter>(scale_type, scale_shape);
    auto zero_point = make_shared<op::v0::Parameter>(zero_point_type, zero_point_shape);

    try
    {
        auto quant =
            make_shared<op::v0::Dequantize>(batch, scale, zero_point, unquantized_type, axes);
        FAIL() << "Inconsistent scale / zero point ranks not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            "Scale shape ({64,?,96,?}) and zero point shape ({64,22,?,?,3}) must match");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(
    type_prop,
    dequantize_partial_input_rank_dynamic_scale_rank_static_dynamic_zero_point_rank_static_dynamic_dims_inconsistent)
{
    PartialShape batch_shape{PartialShape::dynamic()};
    PartialShape scale_shape{64, Dimension::dynamic(), 96, Dimension::dynamic()};
    PartialShape zero_point_shape{65, 22, Dimension::dynamic(), Dimension::dynamic()};
    element::Type unquantized_type = element::f32;
    element::Type quantized_type = element::i8;
    element::Type batch_type = quantized_type;
    element::Type scale_type = unquantized_type;
    element::Type zero_point_type = quantized_type;
    AxisSet axes{0, 1, 5, 88};

    auto batch = make_shared<op::v0::Parameter>(batch_type, batch_shape);
    auto scale = make_shared<op::v0::Parameter>(scale_type, scale_shape);
    auto zero_point = make_shared<op::v0::Parameter>(zero_point_type, zero_point_shape);

    try
    {
        auto quant =
            make_shared<op::v0::Dequantize>(batch, scale, zero_point, unquantized_type, axes);
        FAIL() << "Inconsistent scale / zero point dims not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            "Scale shape ({64,?,96,?}) and zero point shape ({65,22,?,?}) must match");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(
    type_prop,
    dequantize_partial_input_static_rank_dynamic_scale_rank_static_dynamic_zero_point_rank_static_dynamic_ok)
{
    PartialShape batch_shape{2, 4, 6, Dimension::dynamic(), 10, Dimension::dynamic()};
    PartialShape scale_shape{4, Dimension::dynamic(), Dimension::dynamic()};
    PartialShape zero_point_shape{Dimension::dynamic(), 8, Dimension::dynamic()};
    element::Type unquantized_type = element::f32;
    element::Type quantized_type = element::i8;
    element::Type batch_type = quantized_type;
    element::Type scale_type = unquantized_type;
    element::Type zero_point_type = quantized_type;
    AxisSet axes{1, 3, 5};

    auto batch = make_shared<op::v0::Parameter>(batch_type, batch_shape);
    auto scale = make_shared<op::v0::Parameter>(scale_type, scale_shape);
    auto zero_point = make_shared<op::v0::Parameter>(zero_point_type, zero_point_shape);
    auto quant = make_shared<op::v0::Dequantize>(batch, scale, zero_point, unquantized_type, axes);

    ASSERT_EQ(quant->get_output_element_type(0), unquantized_type);
    ASSERT_TRUE(quant->get_output_partial_shape(0).same_scheme(
        PartialShape{2, 4, 6, 8, 10, Dimension::dynamic()}));
}

TEST(
    type_prop,
    dequantize_partial_input_static_rank_dynamic_scale_rank_static_dynamic_zero_point_rank_static_dynamic_axis_oob)
{
    PartialShape batch_shape{2, 4, 6, Dimension::dynamic(), 10, Dimension::dynamic()};
    PartialShape scale_shape{4, Dimension::dynamic(), Dimension::dynamic()};
    PartialShape zero_point_shape{Dimension::dynamic(), 8, Dimension::dynamic()};
    element::Type unquantized_type = element::f32;
    element::Type quantized_type = element::i8;
    element::Type batch_type = quantized_type;
    element::Type scale_type = unquantized_type;
    element::Type zero_point_type = quantized_type;
    AxisSet axes{1, 3, 6};

    auto batch = make_shared<op::v0::Parameter>(batch_type, batch_shape);
    auto scale = make_shared<op::v0::Parameter>(scale_type, scale_shape);
    auto zero_point = make_shared<op::v0::Parameter>(zero_point_type, zero_point_shape);

    try
    {
        auto quant =
            make_shared<op::v0::Dequantize>(batch, scale, zero_point, unquantized_type, axes);
        FAIL() << "Out-of-bound quantization axis not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Quantization axis (6) must be less than input shape rank (6)");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(
    type_prop,
    dequantize_partial_input_static_rank_dynamic_scale_rank_static_dynamic_zero_point_rank_static_dynamic_dims_inconsistent)
{
    PartialShape batch_shape{2, 5, 6, Dimension::dynamic(), 10, Dimension::dynamic()};
    PartialShape scale_shape{4, Dimension::dynamic(), Dimension::dynamic()};
    PartialShape zero_point_shape{Dimension::dynamic(), 8, Dimension::dynamic()};
    element::Type unquantized_type = element::f32;
    element::Type quantized_type = element::i8;
    element::Type batch_type = quantized_type;
    element::Type scale_type = unquantized_type;
    element::Type zero_point_type = quantized_type;
    AxisSet axes{1, 3, 5};

    auto batch = make_shared<op::v0::Parameter>(batch_type, batch_shape);
    auto scale = make_shared<op::v0::Parameter>(scale_type, scale_shape);
    auto zero_point = make_shared<op::v0::Parameter>(zero_point_type, zero_point_shape);

    try
    {
        auto quant =
            make_shared<op::v0::Dequantize>(batch, scale, zero_point, unquantized_type, axes);
        FAIL() << "Inconsistent dimensions not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            "Scale / zero point shape ({4,8,?}) must match input shape ({2,5,6,?,10,?}) "
            "at the quantization axes (AxisSet{1, 3, 5})");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}
