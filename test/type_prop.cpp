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

#include <memory>
using namespace std;
using namespace ngraph;

#define EXPECT_HAS_SUBSTRING(haystack, needle)                                                     \
    EXPECT_PRED_FORMAT2(testing::IsSubstring, needle, haystack)

TEST(type_prop, broadcast_deduce)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    Shape bc_shape{2, 3, 4};
    auto bc = make_shared<op::Broadcast>(param, bc_shape, AxisSet{1});
    ASSERT_EQ(bc->get_element_type(), element::f32);
    ASSERT_EQ(bc->get_shape(), bc_shape);
}

TEST(type_prop, broadcast_axes_oob)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto bc_shape = Shape{2, 3, 4};

    try
    {
        auto bc = make_shared<op::Broadcast>(param, bc_shape, AxisSet{1, 3});
        FAIL() << "Broadcast axis out of bounds not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Broadcast axis index (3) exceeds specified output shape rank");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, broadcast_shape_mismatch_wrong_rank)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto bc_shape = Shape{2, 3, 4, 5};

    try
    {
        auto bc = make_shared<op::Broadcast>(param, bc_shape, AxisSet{1});
        FAIL() << "Output shape mismatch (wrong rank) not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            "Broadcast argument shape, specified output shape, and axes are incompatible");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, broadcast_shape_mismatch_wrong_size)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto bc_shape = Shape{2, 3, 5};

    try
    {
        auto bc = make_shared<op::Broadcast>(param, bc_shape, AxisSet{1});
        FAIL() << "Output shape mismatch (wrong size) not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            "Broadcast argument shape, specified output shape, and axes are incompatible");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, broadcast_partial_rank_dynamic_ok)
{
    auto param = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    Shape bc_shape{2, 3, 4};
    auto bc = make_shared<op::Broadcast>(param, bc_shape, AxisSet{1});
    ASSERT_EQ(bc->get_element_type(), element::f32);
    ASSERT_EQ(bc->get_shape(), bc_shape);
}

TEST(type_prop, broadcast_partial_rank_dynamic_axes_oob)
{
    auto param = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto bc_shape = Shape{2, 3, 4};

    try
    {
        auto bc = make_shared<op::Broadcast>(param, bc_shape, AxisSet{1, 3});
        FAIL() << "Broadcast axis out of bounds not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Broadcast axis index (3) exceeds specified output shape rank");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, broadcast_partial_rank_static_dynamic_ok)
{
    auto param = make_shared<op::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 4});
    Shape bc_shape{2, 3, 4};
    auto bc = make_shared<op::Broadcast>(param, bc_shape, AxisSet{1});
    ASSERT_EQ(bc->get_element_type(), element::f32);
    ASSERT_EQ(bc->get_shape(), bc_shape);
}

TEST(type_prop, broadcast_partial_rank_static_dynamic_axes_oob)
{
    auto param = make_shared<op::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 4});
    auto bc_shape = Shape{2, 3, 4};

    try
    {
        auto bc = make_shared<op::Broadcast>(param, bc_shape, AxisSet{1, 3});
        FAIL() << "Broadcast axis out of bounds not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Broadcast axis index (3) exceeds specified output shape rank");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, broadcast_partial_rank_static_dynamic_shape_mismatch_wrong_rank)
{
    auto param = make_shared<op::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 4});
    auto bc_shape = Shape{2, 3, 4, 5};

    try
    {
        auto bc = make_shared<op::Broadcast>(param, bc_shape, AxisSet{1});
        FAIL() << "Output shape mismatch (wrong rank) not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            "Broadcast argument shape, specified output shape, and axes are incompatible");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, broadcast_partial_rank_static_dynamic_shape_mismatch_wrong_size)
{
    auto param = make_shared<op::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 4});
    auto bc_shape = Shape{2, 3, 5};

    try
    {
        auto bc = make_shared<op::Broadcast>(param, bc_shape, AxisSet{1});
        FAIL() << "Output shape mismatch (wrong size) not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            "Broadcast argument shape, specified output shape, and axes are incompatible");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, batchnorm_training_rank_less_than_2)
{
    auto dummy = make_shared<op::Parameter>(element::f32, Shape{1});
    try
    {
        auto bc = make_shared<op::BatchNormTraining>(0.001, dummy, dummy, dummy);
        FAIL() << "BatchNorm c-tor should throw for tensors whose rank is less than 2";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Input argument must have rank of at least 2"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, batchnorm_training_zero_channel_check)
{
    auto data_batch = make_shared<op::Parameter>(element::f32, Shape{1, 0, 2, 3});
    auto gamma = make_shared<op::Parameter>(element::f32, Shape{0});
    auto beta = make_shared<op::Parameter>(element::f32, Shape{0});
    try
    {
        auto bc = make_shared<op::BatchNormTraining>(0.001, gamma, beta, data_batch);
        FAIL() << "BatchNorm c-tor should throw for tensors w/ zero-dimension channels";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Channel count must be at least 1"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, batchnorm_training_et_check)
{
    auto data_batch = make_shared<op::Parameter>(element::f32, Shape{4, 3, 2, 2});
    auto gamma = make_shared<op::Parameter>(element::f64, Shape{3});
    auto beta = make_shared<op::Parameter>(element::f32, Shape{3});

    try
    {
        auto bc = make_shared<op::BatchNormTraining>(0.001, gamma, beta, data_batch);
        FAIL() << "BatchNorm c-tor should throw for different element types";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Input element types do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, batchnorm_training_shape_check)
{
    auto data_batch = make_shared<op::Parameter>(element::f32, Shape{4, 3, 2, 2});
    auto gamma = make_shared<op::Parameter>(element::f32, Shape{4});
    auto beta = make_shared<op::Parameter>(element::f32, Shape{3});

    try
    {
        auto bc = make_shared<op::BatchNormTraining>(0.001, gamma, beta, data_batch);
        FAIL() << "BatchNorm c-tor should throw if gamma and beta shapes don't match";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Shapes for gamma/beta do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, batchnorm_training_backprop_et_check)
{
    auto data_batch = make_shared<op::Parameter>(element::f32, Shape{4, 3, 2, 2});
    auto gamma = make_shared<op::Parameter>(element::f32, Shape{3});
    auto beta = make_shared<op::Parameter>(element::f64, Shape{3});
    auto mean = make_shared<op::Parameter>(element::f32, Shape{3});
    auto variance = make_shared<op::Parameter>(element::f32, Shape{3});
    auto delta = make_shared<op::Parameter>(element::f32, Shape{4, 3, 2, 2});

    try
    {
        auto bc = make_shared<op::BatchNormTrainingBackprop>(
            0.001, gamma, beta, data_batch, mean, variance, delta);
        FAIL() << "Deduced type should disagree with c-tor arguments";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Input element types do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, batchnorm_training_backprop_shape_check)
{
    auto data_batch = make_shared<op::Parameter>(element::f32, Shape{4, 3, 2, 2});
    auto gamma = make_shared<op::Parameter>(element::f32, Shape{3});
    auto beta = make_shared<op::Parameter>(element::f32, Shape{4});
    auto mean = make_shared<op::Parameter>(element::f32, Shape{3});
    auto variance = make_shared<op::Parameter>(element::f32, Shape{3});
    auto delta = make_shared<op::Parameter>(element::f32, Shape{4, 3, 2, 2});

    try
    {
        auto bc = make_shared<op::BatchNormTrainingBackprop>(
            0.001, gamma, beta, data_batch, mean, variance, delta);
        FAIL() << "Deduced type should disagree with c-tor arguments";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Shapes for gamma/beta/mean/variance do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, batchnorm_training_backprop_delta_check)
{
    auto dummy = make_shared<op::Parameter>(element::f32, Shape{3});
    auto dummy2 = make_shared<op::Parameter>(element::f32, Shape{4});
    auto param = make_shared<op::Parameter>(element::f32, Shape{4, 3, 2, 2});
    auto delta = make_shared<op::Parameter>(element::f32, Shape{4, 3, 2, 3});

    try
    {
        auto bc = make_shared<op::BatchNormTrainingBackprop>(
            0.001, dummy, dummy, param, dummy, dummy, delta);
        FAIL() << "Deduced type should disagree with c-tor arguments";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(), std::string("Shape of delta does not match the shape of the input data"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, batchnorm_inference_partial_all_rank_dynamic)
{
    PartialShape data_batch_shape{PartialShape::dynamic()};
    PartialShape gamma_shape{PartialShape::dynamic()};
    PartialShape beta_shape{PartialShape::dynamic()};
    PartialShape mean_shape{PartialShape::dynamic()};
    PartialShape variance_shape{PartialShape::dynamic()};
    double epsilon = 0.001;
    element::Type data_batch_et = element::f32;
    element::Type gamma_et = element::f32;
    element::Type beta_et = element::f32;
    element::Type mean_et = element::f32;
    element::Type variance_et = element::f32;

    auto data_batch = make_shared<op::Parameter>(data_batch_et, data_batch_shape);
    auto gamma = make_shared<op::Parameter>(gamma_et, gamma_shape);
    auto beta = make_shared<op::Parameter>(beta_et, beta_shape);
    auto mean = make_shared<op::Parameter>(mean_et, mean_shape);
    auto variance = make_shared<op::Parameter>(variance_et, variance_shape);

    auto bn = make_shared<op::BatchNormInference>(epsilon, gamma, beta, data_batch, mean, variance);

    ASSERT_EQ(bn->get_output_size(), 1);
    ASSERT_EQ(bn->get_output_element_type(0), data_batch_et);
    ASSERT_TRUE(bn->get_output_partial_shape(0).rank().is_dynamic());
}

TEST(type_prop, batchnorm_inference_partial_input_rank_static_dynamic_ok)
{
    PartialShape data_batch_shape{
        64, Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()};
    PartialShape gamma_shape{PartialShape::dynamic()};
    PartialShape beta_shape{PartialShape::dynamic()};
    PartialShape mean_shape{PartialShape::dynamic()};
    PartialShape variance_shape{PartialShape::dynamic()};
    double epsilon = 0.001;
    element::Type data_batch_et = element::f32;
    element::Type gamma_et = element::f32;
    element::Type beta_et = element::f32;
    element::Type mean_et = element::f32;
    element::Type variance_et = element::f32;

    auto data_batch = make_shared<op::Parameter>(data_batch_et, data_batch_shape);
    auto gamma = make_shared<op::Parameter>(gamma_et, gamma_shape);
    auto beta = make_shared<op::Parameter>(beta_et, beta_shape);
    auto mean = make_shared<op::Parameter>(mean_et, mean_shape);
    auto variance = make_shared<op::Parameter>(variance_et, variance_shape);

    auto bn = make_shared<op::BatchNormInference>(epsilon, gamma, beta, data_batch, mean, variance);

    ASSERT_EQ(bn->get_output_size(), 1);
    ASSERT_EQ(bn->get_output_element_type(0), data_batch_et);
    ASSERT_TRUE(bn->get_output_partial_shape(0).same_scheme(
        PartialShape{64, Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()}));
}

TEST(type_prop, batchnorm_inference_partial_input_rank_static_dynamic_zero_channels)
{
    PartialShape data_batch_shape{
        Dimension::dynamic(), 0, Dimension::dynamic(), Dimension::dynamic()};
    PartialShape gamma_shape{PartialShape::dynamic()};
    PartialShape beta_shape{PartialShape::dynamic()};
    PartialShape mean_shape{PartialShape::dynamic()};
    PartialShape variance_shape{PartialShape::dynamic()};
    double epsilon = 0.001;
    element::Type data_batch_et = element::f32;
    element::Type gamma_et = element::f32;
    element::Type beta_et = element::f32;
    element::Type mean_et = element::f32;
    element::Type variance_et = element::f32;

    auto data_batch = make_shared<op::Parameter>(data_batch_et, data_batch_shape);
    auto gamma = make_shared<op::Parameter>(gamma_et, gamma_shape);
    auto beta = make_shared<op::Parameter>(beta_et, beta_shape);
    auto mean = make_shared<op::Parameter>(mean_et, mean_shape);
    auto variance = make_shared<op::Parameter>(variance_et, variance_shape);

    try
    {
        auto bn =
            make_shared<op::BatchNormInference>(epsilon, gamma, beta, data_batch, mean, variance);
        FAIL() << "Zero channel count not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Channel count must be at least 1"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, batchnorm_inference_partial_input_rank_dynamic_some_rank_static_dynamic_ok)
{
    PartialShape data_batch_shape{PartialShape::dynamic()};
    PartialShape gamma_shape{Dimension::dynamic()};
    PartialShape beta_shape{PartialShape::dynamic()};
    PartialShape mean_shape{Dimension::dynamic()};
    PartialShape variance_shape{PartialShape::dynamic()};
    double epsilon = 0.001;
    element::Type data_batch_et = element::f32;
    element::Type gamma_et = element::f32;
    element::Type beta_et = element::f32;
    element::Type mean_et = element::f32;
    element::Type variance_et = element::f32;

    auto data_batch = make_shared<op::Parameter>(data_batch_et, data_batch_shape);
    auto gamma = make_shared<op::Parameter>(gamma_et, gamma_shape);
    auto beta = make_shared<op::Parameter>(beta_et, beta_shape);
    auto mean = make_shared<op::Parameter>(mean_et, mean_shape);
    auto variance = make_shared<op::Parameter>(variance_et, variance_shape);

    auto bn = make_shared<op::BatchNormInference>(epsilon, gamma, beta, data_batch, mean, variance);

    ASSERT_EQ(bn->get_output_size(), 1);
    ASSERT_EQ(bn->get_output_element_type(0), data_batch_et);
    ASSERT_TRUE(bn->get_output_partial_shape(0).rank().is_dynamic());
}

TEST(type_prop, batchnorm_inference_partial_input_rank_dynamic_some_rank_static_dynamic_wrong_rank)
{
    PartialShape data_batch_shape{PartialShape::dynamic()};
    PartialShape gamma_shape{Dimension::dynamic(), Dimension::dynamic()};
    PartialShape beta_shape{PartialShape::dynamic()};
    PartialShape mean_shape{Dimension::dynamic(), Dimension::dynamic()};
    PartialShape variance_shape{PartialShape::dynamic()};
    double epsilon = 0.001;
    element::Type data_batch_et = element::f32;
    element::Type gamma_et = element::f32;
    element::Type beta_et = element::f32;
    element::Type mean_et = element::f32;
    element::Type variance_et = element::f32;

    auto data_batch = make_shared<op::Parameter>(data_batch_et, data_batch_shape);
    auto gamma = make_shared<op::Parameter>(gamma_et, gamma_shape);
    auto beta = make_shared<op::Parameter>(beta_et, beta_shape);
    auto mean = make_shared<op::Parameter>(mean_et, mean_shape);
    auto variance = make_shared<op::Parameter>(variance_et, variance_shape);

    try
    {
        auto bn =
            make_shared<op::BatchNormInference>(epsilon, gamma, beta, data_batch, mean, variance);
        FAIL() << "Wrong gamma/beta/mean/variance shape not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Shape for gamma/beta/mean/variance ({?,?}) does not have rank 1"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop,
     batchnorm_inference_partial_input_rank_dynamic_some_rank_static_dynamic_inconsistent_rank)
{
    PartialShape data_batch_shape{PartialShape::dynamic()};
    PartialShape gamma_shape{3, Dimension::dynamic()};
    PartialShape beta_shape{PartialShape::dynamic()};
    PartialShape mean_shape{Dimension::dynamic()};
    PartialShape variance_shape{PartialShape::dynamic()};
    double epsilon = 0.001;
    element::Type data_batch_et = element::f32;
    element::Type gamma_et = element::f32;
    element::Type beta_et = element::f32;
    element::Type mean_et = element::f32;
    element::Type variance_et = element::f32;

    auto data_batch = make_shared<op::Parameter>(data_batch_et, data_batch_shape);
    auto gamma = make_shared<op::Parameter>(gamma_et, gamma_shape);
    auto beta = make_shared<op::Parameter>(beta_et, beta_shape);
    auto mean = make_shared<op::Parameter>(mean_et, mean_shape);
    auto variance = make_shared<op::Parameter>(variance_et, variance_shape);

    try
    {
        auto bn =
            make_shared<op::BatchNormInference>(epsilon, gamma, beta, data_batch, mean, variance);
        FAIL() << "Inconsistent gamma/beta/mean/variance shape not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Shapes for gamma/beta/mean/variance do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop,
     batchnorm_inference_partial_input_rank_dynamic_some_static_inconsistent_channel_count)
{
    PartialShape data_batch_shape{PartialShape::dynamic()};
    PartialShape gamma_shape{3};
    PartialShape beta_shape{PartialShape::dynamic()};
    PartialShape mean_shape{4};
    PartialShape variance_shape{PartialShape::dynamic()};
    double epsilon = 0.001;
    element::Type data_batch_et = element::f32;
    element::Type gamma_et = element::f32;
    element::Type beta_et = element::f32;
    element::Type mean_et = element::f32;
    element::Type variance_et = element::f32;

    auto data_batch = make_shared<op::Parameter>(data_batch_et, data_batch_shape);
    auto gamma = make_shared<op::Parameter>(gamma_et, gamma_shape);
    auto beta = make_shared<op::Parameter>(beta_et, beta_shape);
    auto mean = make_shared<op::Parameter>(mean_et, mean_shape);
    auto variance = make_shared<op::Parameter>(variance_et, variance_shape);

    try
    {
        auto bn =
            make_shared<op::BatchNormInference>(epsilon, gamma, beta, data_batch, mean, variance);
        FAIL() << "Inconsistent gamma/beta/mean/variance channel count not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Shapes for gamma/beta/mean/variance do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, batchnorm_inference_partial_input_rank_static_dynamic_some_static_ok)
{
    PartialShape data_batch_shape{64, Dimension::dynamic(), Dimension::dynamic(), 224};
    PartialShape gamma_shape{3};
    PartialShape beta_shape{PartialShape::dynamic()};
    PartialShape mean_shape{3};
    PartialShape variance_shape{PartialShape::dynamic()};
    double epsilon = 0.001;
    element::Type data_batch_et = element::f32;
    element::Type gamma_et = element::f32;
    element::Type beta_et = element::f32;
    element::Type mean_et = element::f32;
    element::Type variance_et = element::f32;

    auto data_batch = make_shared<op::Parameter>(data_batch_et, data_batch_shape);
    auto gamma = make_shared<op::Parameter>(gamma_et, gamma_shape);
    auto beta = make_shared<op::Parameter>(beta_et, beta_shape);
    auto mean = make_shared<op::Parameter>(mean_et, mean_shape);
    auto variance = make_shared<op::Parameter>(variance_et, variance_shape);

    auto bn = make_shared<op::BatchNormInference>(epsilon, gamma, beta, data_batch, mean, variance);

    ASSERT_EQ(bn->get_output_size(), 1);
    ASSERT_EQ(bn->get_output_element_type(0), data_batch_et);
    ASSERT_TRUE(bn->get_output_partial_shape(0).same_scheme(
        PartialShape{64, 3, Dimension::dynamic(), 224}));
}

TEST(type_prop,
     batchnorm_inference_partial_input_rank_static_dynamic_some_static_inconsistent_channel_count)
{
    PartialShape data_batch_shape{64, 4, Dimension::dynamic(), 224};
    PartialShape gamma_shape{3};
    PartialShape beta_shape{PartialShape::dynamic()};
    PartialShape mean_shape{3};
    PartialShape variance_shape{PartialShape::dynamic()};
    double epsilon = 0.001;
    element::Type data_batch_et = element::f32;
    element::Type gamma_et = element::f32;
    element::Type beta_et = element::f32;
    element::Type mean_et = element::f32;
    element::Type variance_et = element::f32;

    auto data_batch = make_shared<op::Parameter>(data_batch_et, data_batch_shape);
    auto gamma = make_shared<op::Parameter>(gamma_et, gamma_shape);
    auto beta = make_shared<op::Parameter>(beta_et, beta_shape);
    auto mean = make_shared<op::Parameter>(mean_et, mean_shape);
    auto variance = make_shared<op::Parameter>(variance_et, variance_shape);

    try
    {
        auto bn =
            make_shared<op::BatchNormInference>(epsilon, gamma, beta, data_batch, mean, variance);
        FAIL() << "Inconsistent input/gamma/beta/mean/variance channel count not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Input channel dimension (4) does not match "
                                         "shape for gamma/beta/mean/variance ({3})"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, batchnorm_training_partial_all_rank_dynamic)
{
    PartialShape data_batch_shape{PartialShape::dynamic()};
    PartialShape gamma_shape{PartialShape::dynamic()};
    PartialShape beta_shape{PartialShape::dynamic()};
    double epsilon = 0.001;
    element::Type data_batch_et = element::f32;
    element::Type gamma_et = element::f32;
    element::Type beta_et = element::f32;

    auto data_batch = make_shared<op::Parameter>(data_batch_et, data_batch_shape);
    auto gamma = make_shared<op::Parameter>(gamma_et, gamma_shape);
    auto beta = make_shared<op::Parameter>(beta_et, beta_shape);

    auto bn = make_shared<op::BatchNormTraining>(epsilon, gamma, beta, data_batch);

    ASSERT_EQ(bn->get_output_size(), 3);
    ASSERT_EQ(bn->get_output_element_type(0), data_batch_et);
    ASSERT_EQ(bn->get_output_element_type(1), data_batch_et);
    ASSERT_EQ(bn->get_output_element_type(2), data_batch_et);
    ASSERT_TRUE(bn->get_output_partial_shape(0).rank().is_dynamic());
    ASSERT_TRUE(bn->get_output_partial_shape(1).same_scheme(PartialShape::dynamic(1)));
    ASSERT_TRUE(bn->get_output_partial_shape(2).same_scheme(PartialShape::dynamic(1)));
}

TEST(type_prop, batchnorm_training_partial_input_rank_static_dynamic_batch_size_known_ok)
{
    PartialShape data_batch_shape{
        64, Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()};
    PartialShape gamma_shape{PartialShape::dynamic()};
    PartialShape beta_shape{PartialShape::dynamic()};
    double epsilon = 0.001;
    element::Type data_batch_et = element::f32;
    element::Type gamma_et = element::f32;
    element::Type beta_et = element::f32;

    auto data_batch = make_shared<op::Parameter>(data_batch_et, data_batch_shape);
    auto gamma = make_shared<op::Parameter>(gamma_et, gamma_shape);
    auto beta = make_shared<op::Parameter>(beta_et, beta_shape);

    auto bn = make_shared<op::BatchNormTraining>(epsilon, gamma, beta, data_batch);

    ASSERT_EQ(bn->get_output_size(), 3);
    ASSERT_EQ(bn->get_output_element_type(0), data_batch_et);
    ASSERT_EQ(bn->get_output_element_type(1), data_batch_et);
    ASSERT_EQ(bn->get_output_element_type(2), data_batch_et);
    ASSERT_TRUE(bn->get_output_partial_shape(0).same_scheme(
        PartialShape{64, Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()}));
    ASSERT_TRUE(bn->get_output_partial_shape(1).same_scheme(PartialShape::dynamic(1)));
    ASSERT_TRUE(bn->get_output_partial_shape(2).same_scheme(PartialShape::dynamic(1)));
}

TEST(type_prop, batchnorm_training_partial_input_rank_static_dynamic_channel_count_known_ok)
{
    PartialShape data_batch_shape{
        Dimension::dynamic(), 3, Dimension::dynamic(), Dimension::dynamic()};
    PartialShape gamma_shape{PartialShape::dynamic()};
    PartialShape beta_shape{PartialShape::dynamic()};
    double epsilon = 0.001;
    element::Type data_batch_et = element::f32;
    element::Type gamma_et = element::f32;
    element::Type beta_et = element::f32;

    auto data_batch = make_shared<op::Parameter>(data_batch_et, data_batch_shape);
    auto gamma = make_shared<op::Parameter>(gamma_et, gamma_shape);
    auto beta = make_shared<op::Parameter>(beta_et, beta_shape);

    auto bn = make_shared<op::BatchNormTraining>(epsilon, gamma, beta, data_batch);

    ASSERT_EQ(bn->get_output_size(), 3);
    ASSERT_EQ(bn->get_output_element_type(0), data_batch_et);
    ASSERT_EQ(bn->get_output_element_type(1), data_batch_et);
    ASSERT_EQ(bn->get_output_element_type(2), data_batch_et);
    ASSERT_TRUE(bn->get_output_partial_shape(0).same_scheme(
        PartialShape{Dimension::dynamic(), 3, Dimension::dynamic(), Dimension::dynamic()}));
    ASSERT_TRUE(bn->get_output_partial_shape(1).same_scheme(PartialShape{3}));
    ASSERT_TRUE(bn->get_output_partial_shape(2).same_scheme(PartialShape{3}));
}

TEST(type_prop, batchnorm_training_partial_input_rank_static_dynamic_zero_channels)
{
    PartialShape data_batch_shape{
        Dimension::dynamic(), 0, Dimension::dynamic(), Dimension::dynamic()};
    PartialShape gamma_shape{PartialShape::dynamic()};
    PartialShape beta_shape{PartialShape::dynamic()};
    double epsilon = 0.001;
    element::Type data_batch_et = element::f32;
    element::Type gamma_et = element::f32;
    element::Type beta_et = element::f32;

    auto data_batch = make_shared<op::Parameter>(data_batch_et, data_batch_shape);
    auto gamma = make_shared<op::Parameter>(gamma_et, gamma_shape);
    auto beta = make_shared<op::Parameter>(beta_et, beta_shape);
    try
    {
        auto bn = make_shared<op::BatchNormTraining>(epsilon, gamma, beta, data_batch);
        FAIL() << "Zero channel count not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Channel count must be at least 1"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, batchnorm_training_partial_input_rank_dynamic_some_rank_static_dynamic_ok)
{
    PartialShape data_batch_shape{PartialShape::dynamic()};
    PartialShape gamma_shape{Dimension::dynamic()};
    PartialShape beta_shape{PartialShape::dynamic()};
    double epsilon = 0.001;
    element::Type data_batch_et = element::f32;
    element::Type gamma_et = element::f32;
    element::Type beta_et = element::f32;

    auto data_batch = make_shared<op::Parameter>(data_batch_et, data_batch_shape);
    auto gamma = make_shared<op::Parameter>(gamma_et, gamma_shape);
    auto beta = make_shared<op::Parameter>(beta_et, beta_shape);

    auto bn = make_shared<op::BatchNormTraining>(epsilon, gamma, beta, data_batch);

    ASSERT_EQ(bn->get_output_size(), 3);
    ASSERT_EQ(bn->get_output_element_type(0), data_batch_et);
    ASSERT_EQ(bn->get_output_element_type(1), data_batch_et);
    ASSERT_EQ(bn->get_output_element_type(2), data_batch_et);
    ASSERT_TRUE(bn->get_output_partial_shape(0).rank().is_dynamic());
    ASSERT_TRUE(bn->get_output_partial_shape(1).same_scheme(PartialShape::dynamic(1)));
    ASSERT_TRUE(bn->get_output_partial_shape(2).same_scheme(PartialShape::dynamic(1)));
}

TEST(type_prop, batchnorm_training_partial_input_rank_dynamic_some_rank_static_dynamic_wrong_rank)
{
    PartialShape data_batch_shape{PartialShape::dynamic()};
    PartialShape gamma_shape{Dimension::dynamic(), Dimension::dynamic()};
    PartialShape beta_shape{PartialShape::dynamic()};
    double epsilon = 0.001;
    element::Type data_batch_et = element::f32;
    element::Type gamma_et = element::f32;
    element::Type beta_et = element::f32;

    auto data_batch = make_shared<op::Parameter>(data_batch_et, data_batch_shape);
    auto gamma = make_shared<op::Parameter>(gamma_et, gamma_shape);
    auto beta = make_shared<op::Parameter>(beta_et, beta_shape);

    try
    {
        auto bn = make_shared<op::BatchNormTraining>(epsilon, gamma, beta, data_batch);
        FAIL() << "Wrong gamma/beta shape not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Shape for gamma/beta ({?,?}) does not have rank 1"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop,
     batchnorm_training_partial_input_rank_dynamic_some_rank_static_dynamic_inconsistent_rank)
{
    PartialShape data_batch_shape{PartialShape::dynamic()};
    PartialShape gamma_shape{3, Dimension::dynamic()};
    PartialShape beta_shape{Dimension::dynamic()};
    double epsilon = 0.001;
    element::Type data_batch_et = element::f32;
    element::Type gamma_et = element::f32;
    element::Type beta_et = element::f32;

    auto data_batch = make_shared<op::Parameter>(data_batch_et, data_batch_shape);
    auto gamma = make_shared<op::Parameter>(gamma_et, gamma_shape);
    auto beta = make_shared<op::Parameter>(beta_et, beta_shape);

    try
    {
        auto bn = make_shared<op::BatchNormTraining>(epsilon, gamma, beta, data_batch);
        FAIL() << "Inconsistent gamma/beta shape not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Shapes for gamma/beta do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop,
     batchnorm_training_partial_input_rank_dynamic_some_static_inconsistent_channel_count)
{
    PartialShape data_batch_shape{PartialShape::dynamic()};
    PartialShape gamma_shape{3};
    PartialShape beta_shape{4};
    double epsilon = 0.001;
    element::Type data_batch_et = element::f32;
    element::Type gamma_et = element::f32;
    element::Type beta_et = element::f32;

    auto data_batch = make_shared<op::Parameter>(data_batch_et, data_batch_shape);
    auto gamma = make_shared<op::Parameter>(gamma_et, gamma_shape);
    auto beta = make_shared<op::Parameter>(beta_et, beta_shape);

    try
    {
        auto bn = make_shared<op::BatchNormTraining>(epsilon, gamma, beta, data_batch);
        FAIL() << "Inconsistent gamma/beta channel count not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Shapes for gamma/beta do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, batchnorm_training_partial_input_rank_static_dynamic_some_static_ok)
{
    PartialShape data_batch_shape{64, Dimension::dynamic(), Dimension::dynamic(), 224};
    PartialShape gamma_shape{3};
    PartialShape beta_shape{3};
    double epsilon = 0.001;
    element::Type data_batch_et = element::f32;
    element::Type gamma_et = element::f32;
    element::Type beta_et = element::f32;

    auto data_batch = make_shared<op::Parameter>(data_batch_et, data_batch_shape);
    auto gamma = make_shared<op::Parameter>(gamma_et, gamma_shape);
    auto beta = make_shared<op::Parameter>(beta_et, beta_shape);

    auto bn = make_shared<op::BatchNormTraining>(epsilon, gamma, beta, data_batch);

    ASSERT_EQ(bn->get_output_size(), 3);
    ASSERT_EQ(bn->get_output_element_type(0), data_batch_et);
    ASSERT_EQ(bn->get_output_element_type(1), data_batch_et);
    ASSERT_EQ(bn->get_output_element_type(2), data_batch_et);
    ASSERT_TRUE(bn->get_output_partial_shape(0).same_scheme(
        PartialShape{64, 3, Dimension::dynamic(), 224}));
    ASSERT_TRUE(bn->get_output_partial_shape(1).same_scheme(PartialShape{3}));
    ASSERT_TRUE(bn->get_output_partial_shape(2).same_scheme(PartialShape{3}));
}

TEST(type_prop,
     batchnorm_training_partial_input_rank_static_dynamic_some_static_inconsistent_channel_count)
{
    PartialShape data_batch_shape{64, 4, Dimension::dynamic(), 224};
    PartialShape gamma_shape{3};
    PartialShape beta_shape{PartialShape::dynamic()};
    double epsilon = 0.001;
    element::Type data_batch_et = element::f32;
    element::Type gamma_et = element::f32;
    element::Type beta_et = element::f32;

    auto data_batch = make_shared<op::Parameter>(data_batch_et, data_batch_shape);
    auto gamma = make_shared<op::Parameter>(gamma_et, gamma_shape);
    auto beta = make_shared<op::Parameter>(beta_et, beta_shape);

    try
    {
        auto bn = make_shared<op::BatchNormTraining>(epsilon, gamma, beta, data_batch);
        FAIL() << "Inconsistent input/gamma/beta channel count not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Input channel dimension (4) does not match shape for gamma/beta ({3})"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

////
////
////
////

TEST(type_prop, batchnorm_training_backprop_partial_all_rank_dynamic)
{
    PartialShape data_batch_shape{PartialShape::dynamic()};
    PartialShape gamma_shape{PartialShape::dynamic()};
    PartialShape beta_shape{PartialShape::dynamic()};
    PartialShape mean_shape{PartialShape::dynamic()};
    PartialShape variance_shape{PartialShape::dynamic()};
    PartialShape delta_shape{PartialShape::dynamic()};
    double epsilon = 0.001;
    element::Type data_batch_et = element::f32;
    element::Type gamma_et = element::f32;
    element::Type beta_et = element::f32;
    element::Type mean_et = element::f32;
    element::Type variance_et = element::f32;
    element::Type delta_et = element::f32;

    auto data_batch = make_shared<op::Parameter>(data_batch_et, data_batch_shape);
    auto gamma = make_shared<op::Parameter>(gamma_et, gamma_shape);
    auto beta = make_shared<op::Parameter>(beta_et, beta_shape);
    auto mean = make_shared<op::Parameter>(mean_et, mean_shape);
    auto variance = make_shared<op::Parameter>(variance_et, variance_shape);
    auto delta = make_shared<op::Parameter>(delta_et, delta_shape);

    auto bn = make_shared<op::BatchNormTrainingBackprop>(
        epsilon, gamma, beta, data_batch, mean, variance, delta);

    ASSERT_EQ(bn->get_output_size(), 3);
    ASSERT_EQ(bn->get_output_element_type(0), data_batch_et);
    ASSERT_EQ(bn->get_output_element_type(1), data_batch_et);
    ASSERT_EQ(bn->get_output_element_type(2), data_batch_et);
    ASSERT_TRUE(bn->get_output_partial_shape(0).rank().is_dynamic());
    ASSERT_TRUE(bn->get_output_partial_shape(1).same_scheme(PartialShape::dynamic(1)));
    ASSERT_TRUE(bn->get_output_partial_shape(2).same_scheme(PartialShape::dynamic(1)));
}

TEST(type_prop, batchnorm_training_backprop_partial_input_rank_static_dynamic_ok)
{
    PartialShape data_batch_shape{
        64, Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()};
    PartialShape gamma_shape{PartialShape::dynamic()};
    PartialShape beta_shape{PartialShape::dynamic()};
    PartialShape mean_shape{PartialShape::dynamic()};
    PartialShape variance_shape{PartialShape::dynamic()};
    PartialShape delta_shape{PartialShape::dynamic()};
    double epsilon = 0.001;
    element::Type data_batch_et = element::f32;
    element::Type gamma_et = element::f32;
    element::Type beta_et = element::f32;
    element::Type mean_et = element::f32;
    element::Type variance_et = element::f32;
    element::Type delta_et = element::f32;

    auto data_batch = make_shared<op::Parameter>(data_batch_et, data_batch_shape);
    auto gamma = make_shared<op::Parameter>(gamma_et, gamma_shape);
    auto beta = make_shared<op::Parameter>(beta_et, beta_shape);
    auto mean = make_shared<op::Parameter>(mean_et, mean_shape);
    auto variance = make_shared<op::Parameter>(variance_et, variance_shape);
    auto delta = make_shared<op::Parameter>(delta_et, delta_shape);

    auto bn = make_shared<op::BatchNormTrainingBackprop>(
        epsilon, gamma, beta, data_batch, mean, variance, delta);

    ASSERT_EQ(bn->get_output_size(), 3);
    ASSERT_EQ(bn->get_output_element_type(0), data_batch_et);
    ASSERT_EQ(bn->get_output_element_type(1), data_batch_et);
    ASSERT_EQ(bn->get_output_element_type(2), data_batch_et);
    ASSERT_TRUE(bn->get_output_partial_shape(0).same_scheme(
        PartialShape{64, Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()}));
    ASSERT_TRUE(bn->get_output_partial_shape(1).same_scheme(PartialShape::dynamic(1)));
    ASSERT_TRUE(bn->get_output_partial_shape(2).same_scheme(PartialShape::dynamic(1)));
}

TEST(type_prop, batchnorm_training_backprop_partial_input_rank_static_dynamic_zero_channels)
{
    PartialShape data_batch_shape{
        Dimension::dynamic(), 0, Dimension::dynamic(), Dimension::dynamic()};
    PartialShape gamma_shape{PartialShape::dynamic()};
    PartialShape beta_shape{PartialShape::dynamic()};
    PartialShape mean_shape{PartialShape::dynamic()};
    PartialShape variance_shape{PartialShape::dynamic()};
    PartialShape delta_shape{PartialShape::dynamic()};
    double epsilon = 0.001;
    element::Type data_batch_et = element::f32;
    element::Type gamma_et = element::f32;
    element::Type beta_et = element::f32;
    element::Type mean_et = element::f32;
    element::Type variance_et = element::f32;
    element::Type delta_et = element::f32;

    auto data_batch = make_shared<op::Parameter>(data_batch_et, data_batch_shape);
    auto gamma = make_shared<op::Parameter>(gamma_et, gamma_shape);
    auto beta = make_shared<op::Parameter>(beta_et, beta_shape);
    auto mean = make_shared<op::Parameter>(mean_et, mean_shape);
    auto variance = make_shared<op::Parameter>(variance_et, variance_shape);
    auto delta = make_shared<op::Parameter>(delta_et, delta_shape);

    try
    {
        auto bn = make_shared<op::BatchNormTrainingBackprop>(
            epsilon, gamma, beta, data_batch, mean, variance, delta);
        FAIL() << "Zero channel count not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Channel count must be at least 1"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, batchnorm_training_backprop_partial_delta_rank_static_dynamic_ok)
{
    PartialShape data_batch_shape{PartialShape::dynamic()};
    PartialShape gamma_shape{PartialShape::dynamic()};
    PartialShape beta_shape{PartialShape::dynamic()};
    PartialShape mean_shape{PartialShape::dynamic()};
    PartialShape variance_shape{PartialShape::dynamic()};
    PartialShape delta_shape{64, Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()};
    double epsilon = 0.001;
    element::Type data_batch_et = element::f32;
    element::Type gamma_et = element::f32;
    element::Type beta_et = element::f32;
    element::Type mean_et = element::f32;
    element::Type variance_et = element::f32;
    element::Type delta_et = element::f32;

    auto data_batch = make_shared<op::Parameter>(data_batch_et, data_batch_shape);
    auto gamma = make_shared<op::Parameter>(gamma_et, gamma_shape);
    auto beta = make_shared<op::Parameter>(beta_et, beta_shape);
    auto mean = make_shared<op::Parameter>(mean_et, mean_shape);
    auto variance = make_shared<op::Parameter>(variance_et, variance_shape);
    auto delta = make_shared<op::Parameter>(delta_et, delta_shape);

    auto bn = make_shared<op::BatchNormTrainingBackprop>(
        epsilon, gamma, beta, data_batch, mean, variance, delta);

    ASSERT_EQ(bn->get_output_size(), 3);
    ASSERT_EQ(bn->get_output_element_type(0), data_batch_et);
    ASSERT_EQ(bn->get_output_element_type(1), data_batch_et);
    ASSERT_EQ(bn->get_output_element_type(2), data_batch_et);
    ASSERT_TRUE(bn->get_output_partial_shape(0).same_scheme(
        PartialShape{64, Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()}));
    ASSERT_TRUE(bn->get_output_partial_shape(1).same_scheme(PartialShape::dynamic(1)));
    ASSERT_TRUE(bn->get_output_partial_shape(2).same_scheme(PartialShape::dynamic(1)));
}

TEST(type_prop, batchnorm_training_backprop_partial_delta_rank_static_dynamic_channels_known)
{
    PartialShape data_batch_shape{PartialShape::dynamic()};
    PartialShape gamma_shape{PartialShape::dynamic()};
    PartialShape beta_shape{PartialShape::dynamic()};
    PartialShape mean_shape{PartialShape::dynamic()};
    PartialShape variance_shape{PartialShape::dynamic()};
    PartialShape delta_shape{Dimension::dynamic(), 5, Dimension::dynamic(), Dimension::dynamic()};
    double epsilon = 0.001;
    element::Type data_batch_et = element::f32;
    element::Type gamma_et = element::f32;
    element::Type beta_et = element::f32;
    element::Type mean_et = element::f32;
    element::Type variance_et = element::f32;
    element::Type delta_et = element::f32;

    auto data_batch = make_shared<op::Parameter>(data_batch_et, data_batch_shape);
    auto gamma = make_shared<op::Parameter>(gamma_et, gamma_shape);
    auto beta = make_shared<op::Parameter>(beta_et, beta_shape);
    auto mean = make_shared<op::Parameter>(mean_et, mean_shape);
    auto variance = make_shared<op::Parameter>(variance_et, variance_shape);
    auto delta = make_shared<op::Parameter>(delta_et, delta_shape);

    auto bn = make_shared<op::BatchNormTrainingBackprop>(
        epsilon, gamma, beta, data_batch, mean, variance, delta);

    ASSERT_EQ(bn->get_output_size(), 3);
    ASSERT_EQ(bn->get_output_element_type(0), data_batch_et);
    ASSERT_EQ(bn->get_output_element_type(1), data_batch_et);
    ASSERT_EQ(bn->get_output_element_type(2), data_batch_et);
    ASSERT_TRUE(bn->get_output_partial_shape(0).same_scheme(
        PartialShape{Dimension::dynamic(), 5, Dimension::dynamic(), Dimension::dynamic()}));
    ASSERT_TRUE(bn->get_output_partial_shape(1).same_scheme(PartialShape{5}));
    ASSERT_TRUE(bn->get_output_partial_shape(2).same_scheme(PartialShape{5}));
}

TEST(type_prop, batchnorm_training_backprop_partial_delta_rank_static_dynamic_zero_channels)
{
    PartialShape data_batch_shape{PartialShape::dynamic()};
    PartialShape gamma_shape{PartialShape::dynamic()};
    PartialShape beta_shape{PartialShape::dynamic()};
    PartialShape mean_shape{PartialShape::dynamic()};
    PartialShape variance_shape{PartialShape::dynamic()};
    PartialShape delta_shape{Dimension::dynamic(), 0, Dimension::dynamic(), Dimension::dynamic()};
    double epsilon = 0.001;
    element::Type data_batch_et = element::f32;
    element::Type gamma_et = element::f32;
    element::Type beta_et = element::f32;
    element::Type mean_et = element::f32;
    element::Type variance_et = element::f32;
    element::Type delta_et = element::f32;

    auto data_batch = make_shared<op::Parameter>(data_batch_et, data_batch_shape);
    auto gamma = make_shared<op::Parameter>(gamma_et, gamma_shape);
    auto beta = make_shared<op::Parameter>(beta_et, beta_shape);
    auto mean = make_shared<op::Parameter>(mean_et, mean_shape);
    auto variance = make_shared<op::Parameter>(variance_et, variance_shape);
    auto delta = make_shared<op::Parameter>(delta_et, delta_shape);

    try
    {
        auto bn = make_shared<op::BatchNormTrainingBackprop>(
            epsilon, gamma, beta, data_batch, mean, variance, delta);
        FAIL() << "Zero channel count not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Channel count must be at least 1"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop,
     batchnorm_training_backprop_partial_input_and_delta_rank_dynamic_some_rank_static_dynamic_ok)
{
    PartialShape data_batch_shape{PartialShape::dynamic()};
    PartialShape gamma_shape{Dimension::dynamic()};
    PartialShape beta_shape{PartialShape::dynamic()};
    PartialShape mean_shape{Dimension::dynamic()};
    PartialShape variance_shape{PartialShape::dynamic()};
    PartialShape delta_shape{PartialShape::dynamic()};
    double epsilon = 0.001;
    element::Type data_batch_et = element::f32;
    element::Type gamma_et = element::f32;
    element::Type beta_et = element::f32;
    element::Type mean_et = element::f32;
    element::Type variance_et = element::f32;
    element::Type delta_et = element::f32;

    auto data_batch = make_shared<op::Parameter>(data_batch_et, data_batch_shape);
    auto gamma = make_shared<op::Parameter>(gamma_et, gamma_shape);
    auto beta = make_shared<op::Parameter>(beta_et, beta_shape);
    auto mean = make_shared<op::Parameter>(mean_et, mean_shape);
    auto variance = make_shared<op::Parameter>(variance_et, variance_shape);
    auto delta = make_shared<op::Parameter>(delta_et, delta_shape);

    auto bn = make_shared<op::BatchNormTrainingBackprop>(
        epsilon, gamma, beta, data_batch, mean, variance, delta);

    ASSERT_EQ(bn->get_output_size(), 3);
    ASSERT_EQ(bn->get_output_element_type(0), data_batch_et);
    ASSERT_EQ(bn->get_output_element_type(1), data_batch_et);
    ASSERT_EQ(bn->get_output_element_type(2), data_batch_et);
    ASSERT_TRUE(bn->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
    ASSERT_TRUE(bn->get_output_partial_shape(1).same_scheme(PartialShape::dynamic(1)));
    ASSERT_TRUE(bn->get_output_partial_shape(2).same_scheme(PartialShape::dynamic(1)));
}

TEST(
    type_prop,
    batchnorm_training_backprop_partial_input_and_delta_rank_dynamic_some_rank_static_dynamic_wrong_rank)
{
    PartialShape data_batch_shape{PartialShape::dynamic()};
    PartialShape gamma_shape{Dimension::dynamic(), Dimension::dynamic()};
    PartialShape beta_shape{PartialShape::dynamic()};
    PartialShape mean_shape{Dimension::dynamic(), Dimension::dynamic()};
    PartialShape variance_shape{PartialShape::dynamic()};
    PartialShape delta_shape{PartialShape::dynamic()};
    double epsilon = 0.001;
    element::Type data_batch_et = element::f32;
    element::Type gamma_et = element::f32;
    element::Type beta_et = element::f32;
    element::Type mean_et = element::f32;
    element::Type variance_et = element::f32;
    element::Type delta_et = element::f32;

    auto data_batch = make_shared<op::Parameter>(data_batch_et, data_batch_shape);
    auto gamma = make_shared<op::Parameter>(gamma_et, gamma_shape);
    auto beta = make_shared<op::Parameter>(beta_et, beta_shape);
    auto mean = make_shared<op::Parameter>(mean_et, mean_shape);
    auto variance = make_shared<op::Parameter>(variance_et, variance_shape);
    auto delta = make_shared<op::Parameter>(delta_et, delta_shape);

    try
    {
        auto bn = make_shared<op::BatchNormTrainingBackprop>(
            epsilon, gamma, beta, data_batch, mean, variance, delta);
        FAIL() << "Wrong gamma/beta/mean/variance shape not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Shape for gamma/beta/mean/variance ({?,?}) does not have rank 1"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(
    type_prop,
    batchnorm_training_backprop_partial_input_and_delta_rank_dynamic_some_rank_static_dynamic_inconsistent_rank)
{
    PartialShape data_batch_shape{PartialShape::dynamic()};
    PartialShape gamma_shape{3, Dimension::dynamic()};
    PartialShape beta_shape{PartialShape::dynamic()};
    PartialShape mean_shape{Dimension::dynamic()};
    PartialShape variance_shape{PartialShape::dynamic()};
    PartialShape delta_shape{PartialShape::dynamic()};
    double epsilon = 0.001;
    element::Type data_batch_et = element::f32;
    element::Type gamma_et = element::f32;
    element::Type beta_et = element::f32;
    element::Type mean_et = element::f32;
    element::Type variance_et = element::f32;
    element::Type delta_et = element::f32;

    auto data_batch = make_shared<op::Parameter>(data_batch_et, data_batch_shape);
    auto gamma = make_shared<op::Parameter>(gamma_et, gamma_shape);
    auto beta = make_shared<op::Parameter>(beta_et, beta_shape);
    auto mean = make_shared<op::Parameter>(mean_et, mean_shape);
    auto variance = make_shared<op::Parameter>(variance_et, variance_shape);
    auto delta = make_shared<op::Parameter>(delta_et, delta_shape);

    try
    {
        auto bn = make_shared<op::BatchNormTrainingBackprop>(
            epsilon, gamma, beta, data_batch, mean, variance, delta);
        FAIL() << "Wrong gamma/beta/mean/variance shape not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Shapes for gamma/beta/mean/variance do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(
    type_prop,
    batchnorm_training_backprop_partial_input_and_delta_rank_dynamic_some_static_inconsistent_channel_count)
{
    PartialShape data_batch_shape{PartialShape::dynamic()};
    PartialShape gamma_shape{3};
    PartialShape beta_shape{PartialShape::dynamic()};
    PartialShape mean_shape{4};
    PartialShape variance_shape{PartialShape::dynamic()};
    PartialShape delta_shape{PartialShape::dynamic()};
    double epsilon = 0.001;
    element::Type data_batch_et = element::f32;
    element::Type gamma_et = element::f32;
    element::Type beta_et = element::f32;
    element::Type mean_et = element::f32;
    element::Type variance_et = element::f32;
    element::Type delta_et = element::f32;

    auto data_batch = make_shared<op::Parameter>(data_batch_et, data_batch_shape);
    auto gamma = make_shared<op::Parameter>(gamma_et, gamma_shape);
    auto beta = make_shared<op::Parameter>(beta_et, beta_shape);
    auto mean = make_shared<op::Parameter>(mean_et, mean_shape);
    auto variance = make_shared<op::Parameter>(variance_et, variance_shape);
    auto delta = make_shared<op::Parameter>(delta_et, delta_shape);

    try
    {
        auto bn = make_shared<op::BatchNormTrainingBackprop>(
            epsilon, gamma, beta, data_batch, mean, variance, delta);
        FAIL() << "nconsistent gamma/beta/mean/variance channel count not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Shapes for gamma/beta/mean/variance do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop,
     batchnorm_training_backprop_partial_input_and_delta_rank_static_dynamic_some_static_ok)
{
    PartialShape data_batch_shape{64, Dimension::dynamic(), Dimension::dynamic(), 224};
    PartialShape gamma_shape{3};
    PartialShape beta_shape{PartialShape::dynamic()};
    PartialShape mean_shape{3};
    PartialShape variance_shape{PartialShape::dynamic()};
    PartialShape delta_shape{Dimension::dynamic(), 3, 448, 224};
    double epsilon = 0.001;
    element::Type data_batch_et = element::f32;
    element::Type gamma_et = element::f32;
    element::Type beta_et = element::f32;
    element::Type mean_et = element::f32;
    element::Type variance_et = element::f32;
    element::Type delta_et = element::f32;

    auto data_batch = make_shared<op::Parameter>(data_batch_et, data_batch_shape);
    auto gamma = make_shared<op::Parameter>(gamma_et, gamma_shape);
    auto beta = make_shared<op::Parameter>(beta_et, beta_shape);
    auto mean = make_shared<op::Parameter>(mean_et, mean_shape);
    auto variance = make_shared<op::Parameter>(variance_et, variance_shape);
    auto delta = make_shared<op::Parameter>(delta_et, delta_shape);

    auto bn = make_shared<op::BatchNormTrainingBackprop>(
        epsilon, gamma, beta, data_batch, mean, variance, delta);

    ASSERT_EQ(bn->get_output_size(), 3);
    ASSERT_EQ(bn->get_output_element_type(0), data_batch_et);
    ASSERT_EQ(bn->get_output_element_type(1), data_batch_et);
    ASSERT_EQ(bn->get_output_element_type(2), data_batch_et);
    ASSERT_TRUE(bn->get_output_partial_shape(0).same_scheme(PartialShape{64, 3, 448, 224}));
    ASSERT_TRUE(bn->get_output_partial_shape(1).same_scheme(PartialShape{3}));
    ASSERT_TRUE(bn->get_output_partial_shape(2).same_scheme(PartialShape{3}));
}

TEST(
    type_prop,
    batchnorm_training_backprop_partial_input_and_delta_rank_static_dynamic_some_static_inconsistent_channel_count)
{
    PartialShape data_batch_shape{64, Dimension::dynamic(), Dimension::dynamic(), 224};
    PartialShape gamma_shape{3};
    PartialShape beta_shape{PartialShape::dynamic()};
    PartialShape mean_shape{3};
    PartialShape variance_shape{PartialShape::dynamic()};
    PartialShape delta_shape{Dimension::dynamic(), 4, 448, 224};
    double epsilon = 0.001;
    element::Type data_batch_et = element::f32;
    element::Type gamma_et = element::f32;
    element::Type beta_et = element::f32;
    element::Type mean_et = element::f32;
    element::Type variance_et = element::f32;
    element::Type delta_et = element::f32;

    auto data_batch = make_shared<op::Parameter>(data_batch_et, data_batch_shape);
    auto gamma = make_shared<op::Parameter>(gamma_et, gamma_shape);
    auto beta = make_shared<op::Parameter>(beta_et, beta_shape);
    auto mean = make_shared<op::Parameter>(mean_et, mean_shape);
    auto variance = make_shared<op::Parameter>(variance_et, variance_shape);
    auto delta = make_shared<op::Parameter>(delta_et, delta_shape);

    try
    {
        auto bn = make_shared<op::BatchNormTrainingBackprop>(
            epsilon, gamma, beta, data_batch, mean, variance, delta);
        FAIL() << "Inconsistent delta/gamma/beta/mean/variance channel count not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Input channel dimension (4) does not match "
                                         "shape for gamma/beta/mean/variance ({3})"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(
    type_prop,
    batchnorm_training_backprop_partial_input_and_delta_rank_static_dynamic_some_static_inconsistent_batch_size)
{
    PartialShape data_batch_shape{64, 3, Dimension::dynamic(), 224};
    PartialShape gamma_shape{3};
    PartialShape beta_shape{PartialShape::dynamic()};
    PartialShape mean_shape{3};
    PartialShape variance_shape{PartialShape::dynamic()};
    PartialShape delta_shape{128, 4, Dimension::dynamic(), 224};
    double epsilon = 0.001;
    element::Type data_batch_et = element::f32;
    element::Type gamma_et = element::f32;
    element::Type beta_et = element::f32;
    element::Type mean_et = element::f32;
    element::Type variance_et = element::f32;
    element::Type delta_et = element::f32;

    auto data_batch = make_shared<op::Parameter>(data_batch_et, data_batch_shape);
    auto gamma = make_shared<op::Parameter>(gamma_et, gamma_shape);
    auto beta = make_shared<op::Parameter>(beta_et, beta_shape);
    auto mean = make_shared<op::Parameter>(mean_et, mean_shape);
    auto variance = make_shared<op::Parameter>(variance_et, variance_shape);
    auto delta = make_shared<op::Parameter>(delta_et, delta_shape);

    try
    {
        auto bn = make_shared<op::BatchNormTrainingBackprop>(
            epsilon, gamma, beta, data_batch, mean, variance, delta);
        FAIL() << "Inconsistent input/delta batch size not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Shape of delta does not match the shape of the input data (input data "
                        "shape: {64,3,?,224}, delta shape: {128,4,?,224})"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(
    type_prop,
    batchnorm_training_backprop_partial_input_and_delta_rank_static_dynamic_some_static_inconsistent_spatial_dims)
{
    PartialShape data_batch_shape{Dimension::dynamic(), 3, Dimension::dynamic(), 224};
    PartialShape gamma_shape{3};
    PartialShape beta_shape{PartialShape::dynamic()};
    PartialShape mean_shape{3};
    PartialShape variance_shape{PartialShape::dynamic()};
    PartialShape delta_shape{Dimension::dynamic(), 3, Dimension::dynamic(), 448};
    double epsilon = 0.001;
    element::Type data_batch_et = element::f32;
    element::Type gamma_et = element::f32;
    element::Type beta_et = element::f32;
    element::Type mean_et = element::f32;
    element::Type variance_et = element::f32;
    element::Type delta_et = element::f32;

    auto data_batch = make_shared<op::Parameter>(data_batch_et, data_batch_shape);
    auto gamma = make_shared<op::Parameter>(gamma_et, gamma_shape);
    auto beta = make_shared<op::Parameter>(beta_et, beta_shape);
    auto mean = make_shared<op::Parameter>(mean_et, mean_shape);
    auto variance = make_shared<op::Parameter>(variance_et, variance_shape);
    auto delta = make_shared<op::Parameter>(delta_et, delta_shape);

    try
    {
        auto bn = make_shared<op::BatchNormTrainingBackprop>(
            epsilon, gamma, beta, data_batch, mean, variance, delta);
        FAIL() << "Inconsistent input/delta spatial dimensions not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Shape of delta does not match the shape of the input data "
                        "(input data shape: {?,3,?,224}, delta shape: {?,3,?,448})"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, concat_deduce)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{2, 3, 4});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{2, 7, 4});
    auto param2 = make_shared<op::Parameter>(element::f32, Shape{2, 2, 4});
    auto c = make_shared<op::Concat>(NodeVector{param0, param1, param2}, 1);
    ASSERT_EQ(c->get_element_type(), element::f32);
    ASSERT_EQ(c->get_shape(), (Shape{2, 12, 4}));
}

TEST(type_prop, concat_deduce_wrong_rank)
{
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{2, 3, 4});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{2, 7, 4});
    auto param2 = make_shared<op::Parameter>(element::f32,
                                             Shape{
                                                 2, 2,
                                             });
    try
    {
        auto c = make_shared<op::Concat>(NodeVector{param0, param1, param2}, 1);
        // Should have thrown, so fail if it didn't
        FAIL() << "Deduced type should disagree with specified type";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Argument shapes are inconsistent; they must have the same rank, and must "
                        "have equal dimension everywhere except on the concatenation axis"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, concat_deduce_wrong_shape)
{
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{2, 3, 4});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{2, 7, 4});
    auto param2 = make_shared<op::Parameter>(element::f32, Shape{2, 2, 5});
    try
    {
        auto c = make_shared<op::Concat>(NodeVector{param0, param1, param2}, 1);
        // Should have thrown, so fail if it didn't
        FAIL() << "Deduced type should disagree with specified type";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Argument shapes are inconsistent; they must have the same rank, and must "
                        "have equal dimension everywhere except on the concatenation axis"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, concat_deduce_axis_oob)
{
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{2, 3, 4});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{2, 7, 4});
    auto param2 = make_shared<op::Parameter>(element::f32, Shape{2, 2, 5});
    try
    {
        auto c = make_shared<op::Concat>(NodeVector{param0, param1, param2}, 3);
        // Should have thrown, so fail if it didn't
        FAIL() << "Deduced type should disagree with specified type";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Concatenation axis (3) is out of bounds"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, concat_deduce_axis_barely_in_bounds)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{2, 3, 4});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{2, 3, 8});
    auto param2 = make_shared<op::Parameter>(element::f32, Shape{2, 3, 12});
    auto c = make_shared<op::Concat>(NodeVector{param0, param1, param2}, 2);
    ASSERT_EQ(c->get_element_type(), element::f32);
    ASSERT_EQ(c->get_shape(), (Shape{2, 3, 24}));
}

TEST(type_prop, concat_deduce_elem_type_mismatch)
{
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{2, 3, 4});
    auto param1 = make_shared<op::Parameter>(element::i32, Shape{2, 7, 4});
    auto param2 = make_shared<op::Parameter>(element::f32, Shape{2, 2, 4});
    try
    {
        auto c = make_shared<op::Concat>(NodeVector{param0, param1, param2}, 1);
        // Should have thrown, so fail if it didn't
        FAIL() << "Deduced type should disagree with specified type";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Argument element types are inconsistent"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, concat_partial_et_consistent)
{
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{2, 3, 4});
    auto param1 = make_shared<op::Parameter>(element::dynamic, Shape{2, 7, 4});
    auto param2 = make_shared<op::Parameter>(element::f32, Shape{2, 2, 4});
    auto c = make_shared<op::Concat>(NodeVector{param0, param1, param2}, 1);

    ASSERT_EQ(c->get_element_type(), element::f32);
    ASSERT_EQ(c->get_shape(), (Shape{2, 12, 4}));
}

TEST(type_prop, concat_partial_et_inconsistent)
{
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{2, 3, 4});
    auto param1 = make_shared<op::Parameter>(element::dynamic, Shape{2, 7, 4});
    auto param2 = make_shared<op::Parameter>(element::i32, Shape{2, 2, 4});
    try
    {
        auto c = make_shared<op::Concat>(NodeVector{param0, param1, param2}, 1);
        // Should have thrown, so fail if it didn't
        FAIL() << "Inconsistent element types not detected (some dynamic)";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Argument element types are inconsistent"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, concat_partial_all_rank_dynamic)
{
    auto param0 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto param1 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto param2 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto c = make_shared<op::Concat>(NodeVector{param0, param1, param2}, 1);

    ASSERT_TRUE(c->get_output_partial_shape(0).rank().is_dynamic());
}

TEST(type_prop, concat_partial_some_rank_dynamic_others_rank_static_dynamic_consistent)
{
    auto param0 =
        make_shared<op::Parameter>(element::f32, PartialShape{2, Dimension::dynamic(), 3});
    auto param1 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto param2 =
        make_shared<op::Parameter>(element::f32, PartialShape{2, 3, Dimension::dynamic()});
    auto c = make_shared<op::Concat>(NodeVector{param0, param1, param2}, 1);

    ASSERT_TRUE(
        c->get_output_partial_shape(0).same_scheme(PartialShape{2, Dimension::dynamic(), 3}));
}

TEST(type_prop, concat_partial_some_rank_dynamic_others_rank_static_dynamic_rank_inconsistent)
{
    auto param0 =
        make_shared<op::Parameter>(element::f32, PartialShape{2, Dimension::dynamic(), 3});
    auto param1 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto param2 =
        make_shared<op::Parameter>(element::f32, PartialShape{2, 3, Dimension::dynamic(), 4});
    try
    {
        auto c = make_shared<op::Concat>(NodeVector{param0, param1, param2}, 1);
        // Should have thrown, so fail if it didn't
        FAIL() << "Inconsistent ranks not detected (some args rank-dynamic, some args rank-static "
                  "dynamic)";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Argument shapes are inconsistent; they must have the same rank, and must "
                        "have equal dimension everywhere except on the concatenation axis"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, concat_partial_some_rank_dynamic_others_rank_static_dynamic_dims_inconsistent)
{
    auto param0 =
        make_shared<op::Parameter>(element::f32, PartialShape{2, Dimension::dynamic(), 3});
    auto param1 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto param2 =
        make_shared<op::Parameter>(element::f32, PartialShape{3, 3, Dimension::dynamic()});
    try
    {
        auto c = make_shared<op::Concat>(NodeVector{param0, param1, param2}, 1);
        // Should have thrown, so fail if it didn't
        FAIL() << "Inconsistent dimensions not detected (some args rank-dynamic, some args "
                  "rank-static dynamic)";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Argument shapes are inconsistent; they must have the same rank, and must "
                        "have equal dimension everywhere except on the concatenation axis"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop,
     concat_partial_some_rank_dynamic_others_rank_static_dynamic_dims_intransitively_inconsistent)
{
    auto param0 =
        make_shared<op::Parameter>(element::f32, PartialShape{2, Dimension::dynamic(), 3});
    auto param1 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto param2 = make_shared<op::Parameter>(
        element::f32, PartialShape{Dimension::dynamic(), 3, Dimension::dynamic()});
    auto param3 =
        make_shared<op::Parameter>(element::f32, PartialShape{3, 3, Dimension::dynamic()});
    try
    {
        auto c = make_shared<op::Concat>(NodeVector{param0, param1, param2, param3}, 1);
        // Should have thrown, so fail if it didn't
        FAIL() << "Inconsistent dimensions not detected (some args rank-dynamic, some args "
                  "rank-static dynamic)";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Argument shapes are inconsistent; they must have the same rank, and must "
                        "have equal dimension everywhere except on the concatenation axis"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, concat_partial_some_rank_dynamic_others_rank_static_with_concat_axis_static)
{
    auto param0 = make_shared<op::Parameter>(element::f32, PartialShape{2, 2, 3});
    auto param1 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto param2 =
        make_shared<op::Parameter>(element::f32, PartialShape{2, 3, Dimension::dynamic()});
    auto c = make_shared<op::Concat>(NodeVector{param0, param1, param2}, 1);

    ASSERT_TRUE(
        c->get_output_partial_shape(0).same_scheme(PartialShape{2, Dimension::dynamic(), 3}));
}

TEST(type_prop,
     concat_partial_some_rank_dynamic_others_rank_static_with_concat_axis_static_dims_inconsistent)
{
    auto param0 = make_shared<op::Parameter>(element::f32, PartialShape{2, 2, 3});
    auto param1 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto param2 =
        make_shared<op::Parameter>(element::f32, PartialShape{3, 3, Dimension::dynamic()});

    try
    {
        auto c = make_shared<op::Concat>(NodeVector{param0, param1, param2}, 1);
        // Should have thrown, so fail if it didn't
        FAIL() << "Inconsistent dimensions not detected (some args rank-dynamic, some args "
                  "rank-static dynamic)";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Argument shapes are inconsistent; they must have the same rank, and must "
                        "have equal dimension everywhere except on the concatenation axis"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, concat_partial_all_static_with_concat_axis_static_compatible_result_static)
{
    auto param0 = make_shared<op::Parameter>(element::f32, PartialShape{2, 2, 3});
    auto param1 =
        make_shared<op::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 4, 3});
    auto param2 =
        make_shared<op::Parameter>(element::f32, PartialShape{2, 3, Dimension::dynamic()});
    auto c = make_shared<op::Concat>(NodeVector{param0, param1, param2}, 1);

    ASSERT_EQ(c->get_shape(), (Shape{2, 9, 3}));
}

TEST(type_prop, concat_partial_all_static_with_concat_axis_static_compatible_result_dynamic)
{
    auto param0 =
        make_shared<op::Parameter>(element::f32, PartialShape{2, 2, Dimension::dynamic()});
    auto param1 = make_shared<op::Parameter>(
        element::f32, PartialShape{Dimension::dynamic(), 4, Dimension::dynamic()});
    auto param2 =
        make_shared<op::Parameter>(element::f32, PartialShape{2, 3, Dimension::dynamic()});
    auto c = make_shared<op::Concat>(NodeVector{param0, param1, param2}, 1);

    ASSERT_TRUE(
        c->get_output_partial_shape(0).same_scheme(PartialShape{2, 9, Dimension::dynamic()}));
}

TEST(type_prop, concat_partial_all_static_with_concat_axis_static_dims_incompatible)
{
    auto param0 = make_shared<op::Parameter>(element::f32, PartialShape{2, 2, 3});
    auto param1 =
        make_shared<op::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 4, 3});
    auto param2 =
        make_shared<op::Parameter>(element::f32, PartialShape{3, 3, Dimension::dynamic()});
    try
    {
        auto c = make_shared<op::Concat>(NodeVector{param0, param1, param2}, 1);
        // Should have thrown, so fail if it didn't
        FAIL() << "Inconsistent dimensions not detected (some args rank-dynamic, some args "
                  "rank-static dynamic)";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Argument shapes are inconsistent; they must have the same rank, and must "
                        "have equal dimension everywhere except on the concatenation axis"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, convert_deduce)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{2, 3, 4});
    auto c = make_shared<op::Convert>(param, element::i32);
    ASSERT_EQ(c->get_element_type(), element::i32);
    ASSERT_EQ(c->get_shape(), (Shape{2, 3, 4}));
}

TEST(type_prop, dot_deduce_scalar_2d)
{
    // Deduce type for scalar/matrix arguments
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto param2 = make_shared<op::Parameter>(element::f32, Shape{4, 5});
    auto bc = make_shared<op::Dot>(param1, param2);
    ASSERT_EQ(bc->get_element_type(), element::f32);
    ASSERT_EQ(bc->get_shape(), (Shape{4, 5}));
}

TEST(type_prop, dot_deduce_2d_scalar)
{
    // Deduce type for matrix/scalar arguments
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{4, 5});
    auto param2 = make_shared<op::Parameter>(element::f32, Shape{});
    auto bc = make_shared<op::Dot>(param1, param2);
    ASSERT_EQ(bc->get_element_type(), element::f32);
    ASSERT_EQ(bc->get_shape(), (Shape{4, 5}));
}

TEST(type_prop, dot_deduce_scalar_scalar)
{
    // Deduce type for scalar/scalar arguments
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto param2 = make_shared<op::Parameter>(element::f32, Shape{});
    auto bc = make_shared<op::Dot>(param1, param2);
    ASSERT_EQ(bc->get_element_type(), element::f32);
    ASSERT_EQ(bc->get_shape(), (Shape{}));
}

TEST(type_prop, dot_deduce_scalar_1d)
{
    // Deduce type for scalar/vector arguments
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto param2 = make_shared<op::Parameter>(element::f32, Shape{6});
    auto bc = make_shared<op::Dot>(param1, param2);
    ASSERT_EQ(bc->get_element_type(), element::f32);
    ASSERT_EQ(bc->get_shape(), (Shape{6}));
}

TEST(type_prop, dot_deduce_1d)
{
    // Deduce type for vector/vector arguments
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{4});
    auto param2 = make_shared<op::Parameter>(element::f32, Shape{4});
    auto bc = make_shared<op::Dot>(param1, param2);
    ASSERT_EQ(bc->get_element_type(), element::f32);
    ASSERT_EQ(bc->get_shape(), (Shape{}));
}

TEST(type_prop, dot_deduce_2d)
{
    // Deduce type for matrix/matrix arguments
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{4, 2});
    auto param2 = make_shared<op::Parameter>(element::f32, Shape{2, 3});
    auto bc = make_shared<op::Dot>(param1, param2);
    ASSERT_EQ(bc->get_element_type(), element::f32);
    ASSERT_EQ(bc->get_shape(), (Shape{4, 3}));
}

TEST(type_prop, dot_deduce_different_rank)
{
    // Deduce type for different-rank tensor arguments
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{2, 8, 4, 2});
    auto param2 = make_shared<op::Parameter>(element::f32, Shape{2, 1, 3});
    auto bc = make_shared<op::Dot>(param1, param2);
    ASSERT_EQ(bc->get_element_type(), element::f32);
    ASSERT_EQ(bc->get_shape(), (Shape{2, 8, 4, 1, 3}));
}

TEST(type_prop, dot_deduce_element_type_mismatch)
{
    // Type deduction fails due to element type mismatch
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{4, 2});
    auto param2 = make_shared<op::Parameter>(element::i32, Shape{2, 5});
    try
    {
        auto bc = make_shared<op::Dot>(param1, param2);
        // Should have thrown, so fail if it didn't
        FAIL() << "Element type mismatch not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Arguments do not have the same element type"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, dot_deduce_reduction_axes_size_mismatch)
{
    // Type deduction fails due to reduction axes size mismatch
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{4, 2});
    auto param2 = make_shared<op::Parameter>(element::f32, Shape{3, 5});
    try
    {
        auto bc = make_shared<op::Dot>(param1, param2);
        // Should have thrown, so fail if it didn't
        FAIL() << "Dot reduction axes size mismatch not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string(
                "Paired axes (axis 1 from arg0, axis 0 from arg1) do not have same length"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, dot_partial_both_rank_dynamic_axis_count_implicit)
{
    auto param0 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto param1 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto d = make_shared<op::Dot>(param0, param1);

    ASSERT_TRUE(d->get_output_partial_shape(0).rank().is_dynamic());
}

TEST(type_prop, dot_partial_both_rank_dynamic_axis_count_explicit)
{
    auto param0 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto param1 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto d = make_shared<op::Dot>(param0, param1, /*reduction axis count=*/1234);

    ASSERT_TRUE(d->get_output_partial_shape(0).rank().is_dynamic());
}

TEST(type_prop, dot_partial_left_rank_dynamic_right_rank_static_dynamic_axis_count_implicit)
{
    auto param0 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto param1 =
        make_shared<op::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 2, 3});
    auto d = make_shared<op::Dot>(param0, param1);

    ASSERT_TRUE(d->get_output_partial_shape(0).rank().is_dynamic());
}

TEST(type_prop, dot_partial_left_rank_dynamic_right_rank_static_dynamic_axis_count_explicit_ok)
{
    auto param0 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto param1 =
        make_shared<op::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 2, 3});
    auto d = make_shared<op::Dot>(param0, param1, /* reduction axis count=*/3);

    ASSERT_TRUE(d->get_output_partial_shape(0).rank().is_dynamic());
}

TEST(type_prop,
     dot_partial_left_rank_dynamic_right_rank_static_dynamic_axis_count_explicit_too_many)
{
    auto param0 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto param1 =
        make_shared<op::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 2, 3});
    try
    {
        auto d = make_shared<op::Dot>(param0, param1, /* reduction axis count=*/4);
        FAIL()
            << "Too many reduction axes not detected (rank-dynamic/rank-static dynamic operands)";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Reduction axes count (4) is too large");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, dot_partial_left_rank_static_dynamic_right_rank_dynamic_axis_count_implicit)
{
    auto param0 =
        make_shared<op::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 2, 3});
    auto param1 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto d = make_shared<op::Dot>(param0, param1);

    ASSERT_TRUE(d->get_output_partial_shape(0).rank().is_dynamic());
}

TEST(type_prop, dot_partial_left_rank_static_dynamic_right_rank_dynamic_axis_count_explicit_ok)
{
    auto param0 =
        make_shared<op::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 2, 3});
    auto param1 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto d = make_shared<op::Dot>(param0, param1, /* reduction axis count=*/3);

    ASSERT_TRUE(d->get_output_partial_shape(0).rank().is_dynamic());
}

TEST(type_prop,
     dot_partial_left_rank_static_dynamic_right_rank_dynamic_axis_count_explicit_too_many)
{
    auto param0 =
        make_shared<op::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 2, 3});
    auto param1 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    try
    {
        auto d = make_shared<op::Dot>(param0, param1, /* reduction axis count=*/4);
        FAIL()
            << "Too many reduction axes not detected (rank-dynamic/rank-static dynamic operands)";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Reduction axes count (4) is too large");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop,
     dot_partial_left_rank_static_dynamic_right_rank_static_dynamic_axis_count_implicit_1_ok)
{
    auto param0 =
        make_shared<op::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 2, 2});
    auto param1 = make_shared<op::Parameter>(
        element::f32, PartialShape{2, Dimension::dynamic(), 4, Dimension::dynamic(), 5});
    auto d = make_shared<op::Dot>(param0, param1);

    ASSERT_TRUE(d->get_output_partial_shape(0).same_scheme(
        PartialShape{Dimension::dynamic(), 2, Dimension::dynamic(), 4, Dimension::dynamic(), 5}));
}

TEST(type_prop,
     dot_partial_left_rank_static_dynamic_right_rank_static_dynamic_axis_count_implicit_0_ok)
{
    auto param0 = make_shared<op::Parameter>(element::f32, PartialShape{});
    auto param1 = make_shared<op::Parameter>(
        element::f32, PartialShape{2, Dimension::dynamic(), 4, Dimension::dynamic(), 5});
    auto d = make_shared<op::Dot>(param0, param1);

    ASSERT_TRUE(d->get_output_partial_shape(0).same_scheme(
        PartialShape{2, Dimension::dynamic(), 4, Dimension::dynamic(), 5}));
}

TEST(
    type_prop,
    dot_partial_left_rank_static_dynamic_right_rank_static_dynamic_axis_count_explicit_too_many_for_left)
{
    auto param0 =
        make_shared<op::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 2, 3});
    auto param1 =
        make_shared<op::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 2, 3, 5, 6});
    try
    {
        auto d = make_shared<op::Dot>(param0, param1, /* reduction axis count=*/4);
        FAIL() << "Too many reduction axes not detected (rank-static dynamic/rank-static dynamic "
                  "operands)";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Reduction axes count (4) is too large");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(
    type_prop,
    dot_partial_left_rank_static_dynamic_right_rank_static_dynamic_axis_count_explicit_too_many_for_right)
{
    auto param0 =
        make_shared<op::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 2, 3, 5, 6});
    auto param1 =
        make_shared<op::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 2, 3});
    try
    {
        auto d = make_shared<op::Dot>(param0, param1, /* reduction axis count=*/4);
        FAIL() << "Too many reduction axes not detected (rank-static dynamic/rank-static dynamic "
                  "operands)";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Reduction axes count (4) is too large");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(
    type_prop,
    dot_partial_left_rank_static_dynamic_right_rank_static_dynamic_axis_count_explicit_too_many_for_both)
{
    auto param0 =
        make_shared<op::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 2, 3});
    auto param1 =
        make_shared<op::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 2, 3});
    try
    {
        auto d = make_shared<op::Dot>(param0, param1, /* reduction axis count=*/4);
        FAIL() << "Too many reduction axes not detected (rank-static dynamic/rank-static dynamic "
                  "operands)";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Reduction axes count (4) is too large");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, dot_partial_left_et_dynamic)
{
    auto param0 = make_shared<op::Parameter>(element::dynamic, PartialShape::dynamic());
    auto param1 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto d = make_shared<op::Dot>(param0, param1, /* reduction axis count=*/3);

    ASSERT_EQ(d->get_output_element_type(0), element::f32);
}

TEST(type_prop, dot_partial_right_et_dynamic)
{
    auto param0 = make_shared<op::Parameter>(element::i32, PartialShape::dynamic());
    auto param1 = make_shared<op::Parameter>(element::dynamic, PartialShape::dynamic());
    auto d = make_shared<op::Dot>(param0, param1, /* reduction axis count=*/3);

    ASSERT_EQ(d->get_output_element_type(0), element::i32);
}

TEST(type_prop, dot_partial_both_et_dynamic)
{
    auto param0 = make_shared<op::Parameter>(element::dynamic, PartialShape::dynamic());
    auto param1 = make_shared<op::Parameter>(element::dynamic, PartialShape::dynamic());
    auto d = make_shared<op::Dot>(param0, param1, /* reduction axis count=*/3);

    ASSERT_EQ(d->get_output_element_type(0), element::dynamic);
}

//
// Tests for binary elementwise ops.
//
void test_binary(std::string node_type,
                 shared_ptr<Node>(f)(const shared_ptr<Node>& x, const shared_ptr<Node>& y))
{
    // Check for bad arguments
    auto tv0_2_4_param_0 = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto tv0_2_4_param_1 = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto tv0_2_4_param_2 = make_shared<op::Parameter>(element::i32, Shape{2, 4});
    auto tv0_4_2_param = make_shared<op::Parameter>(element::f32, Shape{4, 2});

    auto test_binary_bad_arguments_view_shapes = [&](const shared_ptr<Node>& x,
                                                     const shared_ptr<Node>& y) {
        try
        {
            auto node = f(x, y);
            // Should have thrown, so fail if it didn't
            FAIL() << "Incompatible view arguments not detected.";
        }
        catch (const NodeValidationError& error)
        {
            EXPECT_HAS_SUBSTRING(error.what(), std::string("Argument shapes are inconsistent"));
        }
        catch (...)
        {
            FAIL() << "Deduced type check failed for unexpected reason";
        }
    };
    test_binary_bad_arguments_view_shapes(tv0_2_4_param_0, tv0_4_2_param);

    auto test_binary_bad_arguments_view_element_types = [&](const shared_ptr<Node>& x,
                                                            const shared_ptr<Node>& y) {
        try
        {
            auto node = f(x, y);
            // Should have thrown, so fail if it didn't
            FAIL() << "Incompatible view arguments not detected.";
        }
        catch (const NodeValidationError& error)
        {
            EXPECT_HAS_SUBSTRING(error.what(),
                                 std::string("Argument element types are inconsistent"));
        }
        catch (...)
        {
            FAIL() << "Deduced type check failed for unexpected reason";
        }
    };

    test_binary_bad_arguments_view_element_types(tv0_2_4_param_0, tv0_2_4_param_2);

    auto test_binary_good_arguments = [&](const shared_ptr<Node>& x, const shared_ptr<Node>& y) {
        auto node = f(x, y);
        EXPECT_TRUE(node->has_same_type(node->get_arguments()[0]));
    };
    test_binary_good_arguments(tv0_2_4_param_0, tv0_2_4_param_1);
}

TEST(type_prop, add_bad_arguments)
{
    test_binary("Add",
                [](const shared_ptr<Node>& x, const shared_ptr<Node>& y) -> shared_ptr<Node> {
                    return make_shared<op::Add>(x, y);
                });
}

TEST(type_prop, divide_bad_arguments)
{
    test_binary("Divide",
                [](const shared_ptr<Node>& x, const shared_ptr<Node>& y) -> shared_ptr<Node> {
                    return make_shared<op::Divide>(x, y);
                });
}

TEST(type_prop, multiply_bad_arguments)
{
    test_binary("Multiply",
                [](const shared_ptr<Node>& x, const shared_ptr<Node>& y) -> shared_ptr<Node> {
                    return make_shared<op::Multiply>(x, y);
                });
}

TEST(type_prop, subtract_bad_arguments)
{
    test_binary("Subtract",
                [](const shared_ptr<Node>& x, const shared_ptr<Node>& y) -> shared_ptr<Node> {
                    return make_shared<op::Subtract>(x, y);
                });
}

//
// Tests for binary elementwise logical ops.
//
void test_binary_logical(std::string node_type,
                         shared_ptr<Node>(f)(const shared_ptr<Node>& x, const shared_ptr<Node>& y))
{
    // Check for bad arguments
    auto tv0_2_4_param_0 = make_shared<op::Parameter>(element::boolean, Shape{2, 4});
    auto tv0_2_4_param_1 = make_shared<op::Parameter>(element::boolean, Shape{2, 4});
    auto tv0_2_4_param_2 = make_shared<op::Parameter>(element::i32, Shape{2, 4});
    auto tv0_2_4_param_3 = make_shared<op::Parameter>(element::i32, Shape{2, 4});
    auto tv0_4_2_param = make_shared<op::Parameter>(element::boolean, Shape{4, 2});

    auto test_binary_bad_arguments_view_shapes = [&](const shared_ptr<Node>& x,
                                                     const shared_ptr<Node>& y) {
        try
        {
            auto node = f(x, y);
            // Should have thrown, so fail if it didn't
            FAIL() << "Incompatible view arguments not detected.";
        }
        catch (const NodeValidationError& error)
        {
            EXPECT_HAS_SUBSTRING(error.what(), std::string("Argument shapes are inconsistent"));
        }
        catch (...)
        {
            FAIL() << "Deduced type check failed for unexpected reason";
        }
    };
    test_binary_bad_arguments_view_shapes(tv0_2_4_param_0, tv0_4_2_param);

    auto test_binary_differ_arguments_view_element_types = [&](const shared_ptr<Node>& x,
                                                               const shared_ptr<Node>& y) {
        try
        {
            auto node = f(x, y);
            // Should have thrown, so fail if it didn't
            FAIL() << "Incompatible view arguments not detected.";
        }
        catch (const NodeValidationError& error)
        {
            EXPECT_HAS_SUBSTRING(error.what(),
                                 std::string("Argument element types are inconsistent"));
        }
        catch (...)
        {
            FAIL() << "Deduced type check failed for unexpected reason";
        }
    };

    auto test_binary_non_bool_arguments_view_element_types = [&](const shared_ptr<Node>& x,
                                                                 const shared_ptr<Node>& y) {
        try
        {
            auto node = f(x, y);
            // Should have thrown, so fail if it didn't
            FAIL() << "Incompatible view arguments not detected.";
        }
        catch (const ngraph_error& error)
        {
            EXPECT_HAS_SUBSTRING(error.what(), "must have boolean element type");
        }
        catch (...)
        {
            FAIL() << "Deduced type check failed for unexpected reason";
        }

    };

    test_binary_differ_arguments_view_element_types(tv0_2_4_param_0, tv0_2_4_param_2);
    test_binary_differ_arguments_view_element_types(tv0_2_4_param_2, tv0_2_4_param_0);
    test_binary_non_bool_arguments_view_element_types(tv0_2_4_param_2, tv0_2_4_param_3);

    auto test_binary_good_arguments = [&](const shared_ptr<Node>& x, const shared_ptr<Node>& y) {
        auto node = f(x, y);
        EXPECT_TRUE(node->has_same_type(node->get_arguments()[0]));
    };
    test_binary_good_arguments(tv0_2_4_param_0, tv0_2_4_param_1);
}

TEST(type_prop, and_bad_arguments)
{
    test_binary_logical(
        "And", [](const shared_ptr<Node>& x, const shared_ptr<Node>& y) -> shared_ptr<Node> {
            return make_shared<op::And>(x, y);
        });
}

TEST(type_prop, or_bad_arguments)
{
    test_binary_logical(
        "Or", [](const shared_ptr<Node>& x, const shared_ptr<Node>& y) -> shared_ptr<Node> {
            return make_shared<op::Or>(x, y);
        });
}

TEST(type_prop, comparison_good)
{
    auto tv0_2_4_param_0 = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto tv0_2_4_param_1 = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto eq = make_shared<op::Equal>(tv0_2_4_param_0, tv0_2_4_param_1);
    EXPECT_EQ(eq->get_element_type(), element::boolean);
    EXPECT_EQ(eq->get_shape(), (Shape{2, 4}));
}

TEST(type_prop, binary_arithmetic_bad_argument_element_types)
{
    auto tv0_2_4_param_0 = make_shared<op::Parameter>(element::boolean, Shape{2, 4});
    auto tv0_2_4_param_1 = make_shared<op::Parameter>(element::boolean, Shape{2, 4});
    try
    {
        auto bc = make_shared<op::Add>(tv0_2_4_param_0, tv0_2_4_param_1);
        // Should have thrown, so fail if it didn't
        FAIL() << "Did not detect incorrect element types for arithmetic operator";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Arguments cannot have boolean element type"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, unary_arithmetic_bad_argument_element_types)
{
    auto tv0_2_4_param = make_shared<op::Parameter>(element::boolean, Shape{2, 4});
    try
    {
        auto bc = make_shared<op::Negative>(tv0_2_4_param);
        // Should have thrown, so fail if it didn't
        FAIL() << "Did not detect incorrect element types for arithmetic operator";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Arguments cannot have boolean element type"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, select_deduce)
{
    auto tv0_2_4_param_0 = make_shared<op::Parameter>(element::boolean, Shape{2, 4});
    auto tv0_2_4_param_1 = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto tv0_2_4_param_2 = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto bc = make_shared<op::Select>(tv0_2_4_param_0, tv0_2_4_param_1, tv0_2_4_param_2);
    ASSERT_EQ(bc->get_element_type(), element::f32);
    ASSERT_EQ(bc->get_shape(), (Shape{2, 4}));
}

TEST(type_prop, select_shape_mismatch_a)
{
    auto tv0_2_4_param_0 = make_shared<op::Parameter>(element::boolean, Shape{3, 5});
    auto tv0_2_4_param_1 = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto tv0_2_4_param_2 = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    try
    {
        auto bc = make_shared<op::Select>(tv0_2_4_param_0, tv0_2_4_param_1, tv0_2_4_param_2);
        // Should have thrown, so fail if it didn't
        FAIL() << "Did not detect incorrect element types for arithmetic operator";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Argument shapes are inconsistent"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, select_shape_mismatch_b)
{
    auto tv0_2_4_param_0 = make_shared<op::Parameter>(element::boolean, Shape{2, 4});
    auto tv0_2_4_param_1 = make_shared<op::Parameter>(element::f32, Shape{3, 5});
    auto tv0_2_4_param_2 = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    try
    {
        auto bc = make_shared<op::Select>(tv0_2_4_param_0, tv0_2_4_param_1, tv0_2_4_param_2);
        // Should have thrown, so fail if it didn't
        FAIL() << "Did not detect incorrect element types for arithmetic operator";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Argument shapes are inconsistent"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, select_shape_mismatch_c)
{
    auto tv0_2_4_param_0 = make_shared<op::Parameter>(element::boolean, Shape{2, 4});
    auto tv0_2_4_param_1 = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto tv0_2_4_param_2 = make_shared<op::Parameter>(element::f32, Shape{3, 5});
    try
    {
        auto bc = make_shared<op::Select>(tv0_2_4_param_0, tv0_2_4_param_1, tv0_2_4_param_2);
        // Should have thrown, so fail if it didn't
        FAIL() << "Did not detect incorrect element types for arithmetic operator";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Argument shapes are inconsistent"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, select_elem_mismatch_a)
{
    auto tv0_2_4_param_0 = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto tv0_2_4_param_1 = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto tv0_2_4_param_2 = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    try
    {
        auto bc = make_shared<op::Select>(tv0_2_4_param_0, tv0_2_4_param_1, tv0_2_4_param_2);
        // Should have thrown, so fail if it didn't
        FAIL() << "Did not detect incorrect element types for arithmetic operator";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Argument 0 does not have boolean element type"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, select_elem_mismatch_bc)
{
    auto tv0_2_4_param_0 = make_shared<op::Parameter>(element::boolean, Shape{2, 4});
    auto tv0_2_4_param_1 = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto tv0_2_4_param_2 = make_shared<op::Parameter>(element::i32, Shape{2, 4});
    try
    {
        auto bc = make_shared<op::Select>(tv0_2_4_param_0, tv0_2_4_param_1, tv0_2_4_param_2);
        // Should have thrown, so fail if it didn't
        FAIL() << "Did not detect incorrect element types for arithmetic operator";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Argument 1 and 2 element types are inconsistent"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, select_partial_all_rank_dynamic)
{
    auto param0 = make_shared<op::Parameter>(element::boolean, PartialShape::dynamic());
    auto param1 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto param2 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());

    auto sel = make_shared<op::Select>(param0, param1, param2);

    ASSERT_EQ(sel->get_output_element_type(0), element::f32);
    ASSERT_TRUE(sel->get_output_partial_shape(0).rank().is_dynamic());
}

TEST(type_prop, select_partial_all_rank_dynamic_arg0_et_dynamic_arg1_arg2_et_mismatch)
{
    auto param0 = make_shared<op::Parameter>(element::dynamic, PartialShape::dynamic());
    auto param1 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto param2 = make_shared<op::Parameter>(element::i32, PartialShape::dynamic());

    try
    {
        auto sel = make_shared<op::Select>(param0, param1, param2);
        FAIL() << "Did not detect mismatched element types for args 1 and 2 (element type-dynamic "
                  "arg0)";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Argument 1 and 2 element types are inconsistent"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, select_partial_all_rank_dynamic_arg0_arg1_et_dynamic)
{
    auto param0 = make_shared<op::Parameter>(element::dynamic, PartialShape::dynamic());
    auto param1 = make_shared<op::Parameter>(element::dynamic, PartialShape::dynamic());
    auto param2 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());

    auto sel = make_shared<op::Select>(param0, param1, param2);

    ASSERT_EQ(sel->get_output_element_type(0), element::f32);
    ASSERT_TRUE(sel->get_output_partial_shape(0).rank().is_dynamic());
}

TEST(type_prop, select_partial_all_rank_dynamic_arg0_arg2_et_dynamic)
{
    auto param0 = make_shared<op::Parameter>(element::dynamic, PartialShape::dynamic());
    auto param1 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto param2 = make_shared<op::Parameter>(element::dynamic, PartialShape::dynamic());

    auto sel = make_shared<op::Select>(param0, param1, param2);

    ASSERT_EQ(sel->get_output_element_type(0), element::f32);
    ASSERT_TRUE(sel->get_output_partial_shape(0).rank().is_dynamic());
}

TEST(type_prop, select_partial_all_rank_dynamic_arg0_arg1_arg2_et_dynamic)
{
    auto param0 = make_shared<op::Parameter>(element::dynamic, PartialShape::dynamic());
    auto param1 = make_shared<op::Parameter>(element::dynamic, PartialShape::dynamic());
    auto param2 = make_shared<op::Parameter>(element::dynamic, PartialShape::dynamic());

    auto sel = make_shared<op::Select>(param0, param1, param2);

    ASSERT_EQ(sel->get_output_element_type(0), element::dynamic);
    ASSERT_TRUE(sel->get_output_partial_shape(0).rank().is_dynamic());
}

TEST(type_prop, select_partial_arg0_rank_dynamic_static_arg1_arg2_rank_dynamic_ok)
{
    auto param0 =
        make_shared<op::Parameter>(element::boolean, PartialShape{2, Dimension::dynamic(), 3});
    auto param1 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto param2 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());

    auto sel = make_shared<op::Select>(param0, param1, param2);

    ASSERT_EQ(sel->get_output_element_type(0), element::f32);
    ASSERT_TRUE(
        sel->get_output_partial_shape(0).same_scheme(PartialShape{2, Dimension::dynamic(), 3}));
}

TEST(type_prop, select_partial_arg1_rank_dynamic_static_arg0_arg2_rank_dynamic_ok)
{
    auto param0 = make_shared<op::Parameter>(element::boolean, PartialShape::dynamic());
    auto param1 =
        make_shared<op::Parameter>(element::f32, PartialShape{2, Dimension::dynamic(), 3});
    auto param2 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());

    auto sel = make_shared<op::Select>(param0, param1, param2);

    ASSERT_EQ(sel->get_output_element_type(0), element::f32);
    ASSERT_TRUE(
        sel->get_output_partial_shape(0).same_scheme(PartialShape{2, Dimension::dynamic(), 3}));
}

TEST(type_prop, select_partial_arg2_rank_dynamic_static_arg0_arg1_rank_dynamic_ok)
{
    auto param0 = make_shared<op::Parameter>(element::boolean, PartialShape::dynamic());
    auto param1 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto param2 =
        make_shared<op::Parameter>(element::f32, PartialShape{2, Dimension::dynamic(), 3});

    auto sel = make_shared<op::Select>(param0, param1, param2);

    ASSERT_EQ(sel->get_output_element_type(0), element::f32);
    ASSERT_TRUE(
        sel->get_output_partial_shape(0).same_scheme(PartialShape{2, Dimension::dynamic(), 3}));
}

TEST(type_prop, select_partial_all_rank_static_dynamic_ok)
{
    auto param0 = make_shared<op::Parameter>(
        element::boolean, PartialShape{2, Dimension::dynamic(), Dimension::dynamic()});
    auto param1 = make_shared<op::Parameter>(
        element::f32, PartialShape{Dimension::dynamic(), 8, Dimension::dynamic()});
    auto param2 = make_shared<op::Parameter>(
        element::f32, PartialShape{Dimension::dynamic(), Dimension::dynamic(), 3});

    auto sel = make_shared<op::Select>(param0, param1, param2);

    ASSERT_EQ(sel->get_output_element_type(0), element::f32);
    ASSERT_TRUE(sel->get_output_partial_shape(0).is_static());
    ASSERT_EQ(sel->get_output_shape(0), (Shape{2, 8, 3}));
}

TEST(type_prop, select_partial_all_rank_static_intransitive_incompatibility)
{
    auto param0 = make_shared<op::Parameter>(
        element::boolean, PartialShape{2, Dimension::dynamic(), Dimension::dynamic()});
    auto param1 = make_shared<op::Parameter>(
        element::f32, PartialShape{Dimension::dynamic(), 8, Dimension::dynamic()});
    auto param2 =
        make_shared<op::Parameter>(element::f32, PartialShape{3, Dimension::dynamic(), 3});

    try
    {
        auto sel = make_shared<op::Select>(param0, param1, param2);
        FAIL() << "Did not detect intransitive partial-shape incompatibility";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Argument shapes are inconsistent"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, reduce_deduce)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f =
        make_shared<Function>(f_param_0 + f_param_1, op::ParameterVector{f_param_0, f_param_1});

    auto r0 = make_shared<op::Reduce>(param_0, param_1, f, AxisSet{0});
    ASSERT_EQ(r0->get_element_type(), element::f32);
    ASSERT_EQ(r0->get_shape(), (Shape{4}));

    auto r1 = make_shared<op::Reduce>(param_0, param_1, f, AxisSet{1});
    ASSERT_EQ(r1->get_element_type(), element::f32);
    ASSERT_EQ(r1->get_shape(), (Shape{2}));

    auto r01 = make_shared<op::Reduce>(param_0, param_1, f, AxisSet{0, 1});
    ASSERT_EQ(r01->get_element_type(), element::f32);
    ASSERT_EQ(r01->get_shape(), (Shape{}));

    auto r_none = make_shared<op::Reduce>(param_0, param_1, f, AxisSet{});
    ASSERT_EQ(r_none->get_element_type(), element::f32);
    ASSERT_EQ(r_none->get_shape(), (Shape{2, 4}));
}

TEST(type_prop, reduce_nonscalar)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{2});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f =
        make_shared<Function>(f_param_0 + f_param_1, op::ParameterVector{f_param_0, f_param_1});

    try
    {
        auto r0 = make_shared<op::Reduce>(param_0, param_1, f, AxisSet{0});
        // Should have thrown, so fail if it didn't
        FAIL() << "Did not detect non-scalar initial value for reduce";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(), std::string("Argument for initial value is not a scalar"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, reduce_elem_type_mismatch)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto param_1 = make_shared<op::Parameter>(element::boolean, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f =
        make_shared<Function>(f_param_0 + f_param_1, op::ParameterVector{f_param_0, f_param_1});

    try
    {
        auto r0 = make_shared<op::Reduce>(param_0, param_1, f, AxisSet{0});
        // Should have thrown, so fail if it didn't
        FAIL() << "Did not detect element type mismatch for reduce";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(),
                  std::string("Element types for reductee and initial values do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, reduce_function_return_element_type_mismatch)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(make_shared<op::Equal>(f_param_0, f_param_1),
                                   op::ParameterVector{f_param_0, f_param_1});

    try
    {
        auto r0 = make_shared<op::Reduce>(param_0, param_1, f, AxisSet{0});
        // Should have thrown, so fail if it didn't
        FAIL() << "Did not detect incorrect element return type for reduction function";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(
            error.what(),
            std::string("Return element type from reduction function does not match expected"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, reduce_function_return_shape_mismatch)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(
        make_shared<op::Broadcast>(f_param_0 + f_param_1, Shape{1}, AxisSet{0}),
        op::ParameterVector{f_param_0, f_param_1});

    try
    {
        auto r0 = make_shared<op::Reduce>(param_0, param_1, f, AxisSet{0});
        // Should have thrown, so fail if it didn't
        FAIL() << "Did not detect incorrect return shape for reduction function";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(),
                  std::string("Return shape from reduction function is not a scalar"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, reduce_function_arg0_type_mismatch)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::boolean, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(f_param_1, op::ParameterVector{f_param_0, f_param_1});

    try
    {
        auto r0 = make_shared<op::Reduce>(param_0, param_1, f, AxisSet{0});
        // Should have thrown, so fail if it didn't
        FAIL() << "Did not detect incorrect element types for arithmetic operator";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(), std::string("Argument 0 of reduction function has wrong type"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, reduce_function_arg1_type_mismatch)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::boolean, Shape{});
    auto f = make_shared<Function>(f_param_0, op::ParameterVector{f_param_0, f_param_1});

    try
    {
        auto r0 = make_shared<op::Reduce>(param_0, param_1, f, AxisSet{0});
        // Should have thrown, so fail if it didn't
        FAIL() << "Did not detect incorrect element types for arithmetic operator";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(), std::string("Argument 1 of reduction function has wrong type"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, reduce_function_arg_count_mismatch)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_2 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(f_param_0 + f_param_1 + f_param_2,
                                   op::ParameterVector{f_param_0, f_param_1, f_param_2});

    try
    {
        auto r0 = make_shared<op::Reduce>(param_0, param_1, f, AxisSet{0});
        // Should have thrown, so fail if it didn't
        FAIL() << "Did not detect incorrect element types for arithmetic operator";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(),
                  std::string("Reduction function has wrong number of parameters (should be two)"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, reduce_axis_oob)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f =
        make_shared<Function>(f_param_0 + f_param_1, op::ParameterVector{f_param_0, f_param_1});

    try
    {
        auto r = make_shared<op::Reduce>(param_0, param_1, f, AxisSet{0, 2, 1});
        // Should have thrown, so fail if it didn't
        FAIL() << "Did not detect out-of-bound axis for reduce";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(), std::string("Reduction axis is out of bounds"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, function_call_deduce)
{
    // First create "f(A,B,C) = (A+B)*C".
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto C = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>((A + B * C), op::ParameterVector{A, B, C});

    // Now make "f(X,Y,Z) + f(X,Y,Z)"
    auto X = make_shared<op::Parameter>(element::f32, shape);
    auto Y = make_shared<op::Parameter>(element::f32, shape);
    auto Z = make_shared<op::Parameter>(element::f32, shape);
    auto r = make_shared<op::FunctionCall>(f, NodeVector{X, Y, Z});
    auto r_p_r = r + r;

    ASSERT_EQ(r_p_r->get_element_type(), element::f32);
    ASSERT_EQ(r_p_r->get_shape(), shape);
}

TEST(type_prop, reshape_deduce_s2v)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{});
    auto r = make_shared<op::Reshape>(param, AxisVector{}, Shape{1});
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_shape(), (Shape{1}));
}

TEST(type_prop, reshape_deduce_s2m)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{});
    auto r = make_shared<op::Reshape>(param, AxisVector{}, Shape{1, 1});
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_shape(), (Shape{1, 1}));
}

TEST(type_prop, reshape_deduce_s2t)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{});
    auto r = make_shared<op::Reshape>(param, AxisVector{}, Shape{1, 1, 1});
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_shape(), (Shape{1, 1, 1}));
}

TEST(type_prop, reshape_deduce_v2s)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{1});
    auto r = make_shared<op::Reshape>(param, AxisVector{0}, Shape{});
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_shape(), (Shape{}));
}

TEST(type_prop, reshape_deduce_m2s)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{1, 1});
    auto r = make_shared<op::Reshape>(param, AxisVector{0, 1}, Shape{});
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_shape(), (Shape{}));
}

TEST(type_prop, reshape_deduce_t2s)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{1, 1, 1});
    auto r = make_shared<op::Reshape>(param, AxisVector{0, 1, 2}, Shape{});
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_shape(), (Shape{}));
}

TEST(type_prop, reshape_deduce_m2v_01)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{3, 4});
    auto r = make_shared<op::Reshape>(param, AxisVector{0, 1}, Shape{12});
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_shape(), (Shape{12}));
}

TEST(type_prop, reshape_deduce_m2v_10)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{3, 4});
    auto r = make_shared<op::Reshape>(param, AxisVector{1, 0}, Shape{12});
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_shape(), (Shape{12}));
}

TEST(type_prop, reshape_deduce_t2v_012)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{3, 4, 5});
    auto r = make_shared<op::Reshape>(param, AxisVector{0, 1, 2}, Shape{60});
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_shape(), (Shape{60}));
}

TEST(type_prop, reshape_deduce_t2v_120)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{3, 4, 5});
    auto r = make_shared<op::Reshape>(param, AxisVector{1, 2, 0}, Shape{60});
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_shape(), (Shape{60}));
}

TEST(type_prop, reshape_deduce_not_enough_axes)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{3, 4, 5});
    try
    {
        auto r = make_shared<op::Reshape>(param, AxisVector{1, 0}, Shape{60});
        // Should have thrown, so fail if it didn't
        FAIL() << "Not enough axes not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Input axis order is not a permutation of argument's axis indices"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, reshape_deduce_too_many_axes)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{3, 4, 5});
    try
    {
        auto r = make_shared<op::Reshape>(param, AxisVector{1, 2, 0, 3}, Shape{60});
        // Should have thrown, so fail if it didn't
        FAIL() << "Too many axes not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Input axis order is not a permutation of argument's axis indices"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, reshape_deduce_duplicate_axes)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{3, 4, 5});
    try
    {
        auto r = make_shared<op::Reshape>(param, AxisVector{1, 1, 0}, Shape{60});
        // Should have thrown, so fail if it didn't
        FAIL() << "Too many axes not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Input axis order is not a permutation of argument's axis indices"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, reshape_deduce_wrong_output_shape)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{3, 4, 5});
    try
    {
        auto r = make_shared<op::Reshape>(param, AxisVector{1, 2, 0}, Shape{3, 3, 3});
        // Should have thrown, so fail if it didn't
        FAIL() << "Too many axes not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Product of output shape dimensions does not match "
                                         "product of argument shape dimensions"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

//
// Input shape rank dynamic, so we should set the desired output shape if the axis vector is not
// known invalid (invalid means it's not a permutation of {0,...,n-1} for any n).
//
TEST(type_prop, reshape_partial_rank_dynamic_axisvector_ok)
{
    auto param = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto r = make_shared<op::Reshape>(param, AxisVector{2, 1, 0, 3}, Shape{3, 1, 8, 2});
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_TRUE(r->get_output_partial_shape(0).is_static());
    ASSERT_EQ(r->get_shape(), (Shape{3, 1, 8, 2}));
}

TEST(type_prop, reshape_partial_rank_dynamic_axisvector_not_ok)
{
    auto param = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    try
    {
        auto r = make_shared<op::Reshape>(param, AxisVector{2, 1, 0, 4}, Shape{3, 1, 8, 2});
        // Should have thrown, so fail if it didn't
        FAIL() << "Did not detect malformed AxisVector (input shape rank dynamic)";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Input axis order is not a permutation of argument's axis indices"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

//
// Input shape rank static but input shape is dynamic, so should set desired output shape if the
// axis vector is consistent with the static rank.
//
TEST(type_prop, reshape_partial_rank_static_dynamic_axisvector_ok)
{
    auto param_shape =
        PartialShape{Dimension::dynamic(), 6, Dimension::dynamic(), Dimension::dynamic()};
    auto param = make_shared<op::Parameter>(element::f32, param_shape);
    auto r = make_shared<op::Reshape>(param, AxisVector{2, 1, 0, 3}, Shape{3, 1, 8, 2});
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_TRUE(r->get_output_partial_shape(0).is_static());
    ASSERT_EQ(r->get_shape(), (Shape{3, 1, 8, 2}));
}

TEST(type_prop, reshape_partial_rank_static_dynamic_axisvector_not_ok)
{
    auto param_shape =
        PartialShape{Dimension::dynamic(), 6, Dimension::dynamic(), Dimension::dynamic()};
    auto param = make_shared<op::Parameter>(element::f32, param_shape);
    try
    {
        auto r = make_shared<op::Reshape>(param, AxisVector{2, 1, 0}, Shape{3, 1, 8, 2});
        // Should have thrown, so fail if it didn't
        FAIL() << "Did not detect AxisVector inconsistent with rank (rank-static dynamic shape)";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Input axis order is not a permutation of argument's axis indices"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

//
// Input shape rank static but input shape is dynamic, _but_ one of its static dimensions is zero,
// so should set desired output shape only if it also has zero elements.
//
TEST(type_prop, reshape_partial_rank_static_dynamic_but_zero_ok)
{
    auto param_shape =
        PartialShape{Dimension::dynamic(), 0, Dimension::dynamic(), Dimension::dynamic()};
    auto param = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto r = make_shared<op::Reshape>(param, AxisVector{2, 1, 0, 3}, Shape{3, 1, 0, 2});
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_TRUE(r->get_output_partial_shape(0).is_static());
    ASSERT_EQ(r->get_shape(), (Shape{3, 1, 0, 2}));
}

TEST(type_prop, reshape_partial_rank_static_dynamic_but_zero_not_ok)
{
    auto param_shape =
        PartialShape{Dimension::dynamic(), 0, Dimension::dynamic(), Dimension::dynamic()};
    auto param = make_shared<op::Parameter>(element::f32, param_shape);
    try
    {
        auto r = make_shared<op::Reshape>(param, AxisVector{2, 1, 0}, Shape{3, 1, 8, 2});
        // Should have thrown, so fail if it didn't
        FAIL() << "Did not detect inconsistent output shape with static-zero-element rank-dynamic"
                  " static input shape";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Input axis order is not a permutation of argument's axis indices"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, slice_deduce_vector)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{6});
    auto sl = make_shared<op::Slice>(param, Coordinate{2}, Coordinate{5});
    ASSERT_EQ(sl->get_element_type(), element::f32);
    ASSERT_EQ(sl->get_shape(), (Shape{3}));
}

TEST(type_prop, slice_deduce_matrix)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{6, 8});
    auto sl = make_shared<op::Slice>(param, Coordinate{2, 1}, Coordinate{5, 7});
    ASSERT_EQ(sl->get_element_type(), element::f32);
    ASSERT_EQ(sl->get_shape(), (Shape{3, 6}));
}

TEST(type_prop, slice_deduce_matrix_strided)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{6, 8});
    auto sl = make_shared<op::Slice>(param, Coordinate{2, 1}, Coordinate{5, 7}, Strides{3, 2});
    ASSERT_EQ(sl->get_element_type(), element::f32);
    ASSERT_EQ(sl->get_shape(), (Shape{1, 3}));
}

TEST(type_prop, slice_deduce_matrix_strided_uneven)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{6, 8});
    auto sl = make_shared<op::Slice>(param, Coordinate{2, 1}, Coordinate{5, 7}, Strides{3, 4});
    ASSERT_EQ(sl->get_element_type(), element::f32);
    ASSERT_EQ(sl->get_shape(), (Shape{1, 2}));
}

TEST(type_prop, slice_deduce_vector_edge)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{6});
    auto sl = make_shared<op::Slice>(param, Coordinate{0}, Coordinate{6});
    ASSERT_EQ(sl->get_element_type(), element::f32);
    ASSERT_EQ(sl->get_shape(), (Shape{6}));
}

TEST(type_prop, slice_deduce_matrix_edge)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{6, 8});
    auto sl = make_shared<op::Slice>(param, Coordinate{0, 0}, Coordinate{6, 8});
    ASSERT_EQ(sl->get_element_type(), element::f32);
    ASSERT_EQ(sl->get_shape(), (Shape{6, 8}));
}

TEST(type_prop, slice_deduce_matrix_zero_cols)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{6, 8});
    auto sl = make_shared<op::Slice>(param, Coordinate{0, 0}, Coordinate{6, 0});
    ASSERT_EQ(sl->get_element_type(), element::f32);
    ASSERT_EQ(sl->get_shape(), (Shape{6, 0}));
}

TEST(type_prop, slice_deduce_matrix_zero_zero)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{6, 8});
    auto sl = make_shared<op::Slice>(param, Coordinate{0, 0}, Coordinate{0, 0});
    ASSERT_EQ(sl->get_element_type(), element::f32);
    ASSERT_EQ(sl->get_shape(), (Shape{0, 0}));
}

TEST(type_prop, slice_deduce_vector_invalid_strides)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{6});
    try
    {
        auto sl = make_shared<op::Slice>(param, Coordinate{0}, Coordinate{7}, Strides{1, 2});
        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid slice strides not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Ranks of lower bounds (Coordinate{0}), upper bounds "
                        "(Coordinate{7}) and strides (Strides{1, 2}) do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, slice_deduce_vector_edge_upper_oob)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{6});
    try
    {
        auto sl = make_shared<op::Slice>(param, Coordinate{0}, Coordinate{7});
        // Should have thrown, so fail if it didn't
        FAIL() << "Upper bound out of range not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Upper bound for slice at axis 0 is out of range"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, slice_deduce_matrix_edge_upper_oob)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{6, 8});
    try
    {
        auto sl = make_shared<op::Slice>(param, Coordinate{0, 0}, Coordinate{6, 9});
        // Should have thrown, so fail if it didn't
        FAIL() << "Upper bound out of range not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Upper bound for slice at axis 1 is out of range"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, slice_deduce_vector_lower_above_upper)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{6});
    try
    {
        auto sl = make_shared<op::Slice>(param, Coordinate{3}, Coordinate{2});
        // Should have thrown, so fail if it didn't
        FAIL() << "Lower bound above upper not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Lower bound for slice is greater than upper bound at axis 0"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, slice_deduce_matrix_lower_above_upper)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{6, 8});
    try
    {
        auto sl = make_shared<op::Slice>(param, Coordinate{0, 5}, Coordinate{6, 4});
        // Should have thrown, so fail if it didn't
        FAIL() << "Lower bound above upper not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Lower bound for slice is greater than upper bound at axis 1"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, slice_deduce_matrix_lower_missing)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{6, 8});
    try
    {
        auto sl = make_shared<op::Slice>(param, Coordinate{0}, Coordinate{5, 5});
        // Should have thrown, so fail if it didn't
        FAIL() << "Missing lower bound coordinate not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Ranks of lower bounds (Coordinate{0}), upper bounds "
                        "(Coordinate{5, 5}) and strides (Strides{1}) do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, slice_deduce_matrix_upper_missing)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{6, 8});
    try
    {
        auto sl = make_shared<op::Slice>(param, Coordinate{0, 0}, Coordinate{5});
        // Should have thrown, so fail if it didn't
        FAIL() << "Missing upper bound coordinate not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Ranks of lower bounds (Coordinate{0, 0}), upper bounds "
                        "(Coordinate{5}) and strides (Strides{1, 1}) do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, slice_deduce_matrix_lower_extra)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{6, 8});
    try
    {
        auto sl = make_shared<op::Slice>(param, Coordinate{0, 0, 0}, Coordinate{5, 5});
        // Should have thrown, so fail if it didn't
        FAIL() << "Extra lower bound coordinate not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Ranks of lower bounds (Coordinate{0, 0, "
                                         "0}), upper bounds (Coordinate{5, 5}) and "
                                         "strides (Strides{1, 1, 1}) do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, slice_deduce_matrix_upper_extra)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{6, 8});
    try
    {
        auto sl = make_shared<op::Slice>(param, Coordinate{0, 0}, Coordinate{5, 5, 5});
        // Should have thrown, so fail if it didn't
        FAIL() << "Extra upper bound coordinate not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Ranks of lower bounds (Coordinate{0, 0}), "
                                         "upper bounds (Coordinate{5, 5, 5}) and "
                                         "strides (Strides{1, 1}) do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, slice_partial_arg_input_rank_dynamic_attribs_ok)
{
    PartialShape input_shape{PartialShape::dynamic()};
    Coordinate lower_bounds{1, 2, 3, 4};
    Coordinate upper_bounds{1, 3, 5, 7};
    Strides strides{1, 1, 1, 2};

    auto param = make_shared<op::Parameter>(element::f32, input_shape);
    auto sl = make_shared<op::Slice>(param, lower_bounds, upper_bounds, strides);

    ASSERT_EQ(sl->get_element_type(), element::f32);
    ASSERT_EQ(sl->get_shape(), (Shape{0, 1, 2, 2}));
}

TEST(type_prop, slice_partial_arg_rank_dynamic_attribs_rank_mismatch)
{
    PartialShape input_shape{PartialShape::dynamic()};
    Coordinate lower_bounds{1, 2, 3, 4};
    Coordinate upper_bounds{1, 3, 5};
    Strides strides{1, 1, 1, 2};

    auto param = make_shared<op::Parameter>(element::f32, input_shape);
    try
    {
        auto sl = make_shared<op::Slice>(param, lower_bounds, upper_bounds, strides);
        // Should have thrown, so fail if it didn't
        FAIL() << "Mismatch of lower-bounds/upper-bounds/strides ranks not detected (argument "
                  "rank-dynamic)";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Ranks of lower bounds (Coordinate{1, 2, 3, 4}), upper bounds "
                        "(Coordinate{1, 3, 5}) and strides (Strides{1, 1, 1, 2}) do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, slice_partial_arg_rank_dynamic_attribs_bounds_crossing)
{
    PartialShape input_shape{PartialShape::dynamic()};
    Coordinate lower_bounds{1, 2, 3, 8};
    Coordinate upper_bounds{1, 3, 5, 7};
    Strides strides{1, 1, 1, 2};

    auto param = make_shared<op::Parameter>(element::f32, input_shape);
    try
    {
        auto sl = make_shared<op::Slice>(param, lower_bounds, upper_bounds, strides);
        // Should have thrown, so fail if it didn't
        FAIL() << "Crossing lower/upper bounds not detected (argument rank-dynamic)";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Lower bound for slice is greater than upper bound at axis 3 (lower "
                        "bounds: Coordinate{1, 2, 3, 8}, upper bounds: Coordinate{1, 3, 5, 7})"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, slice_partial_arg_rank_static_dynamic_ok)
{
    PartialShape input_shape{
        Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()};
    Coordinate lower_bounds{1, 2, 3, 4};
    Coordinate upper_bounds{1, 3, 5, 7};
    Strides strides{1, 1, 1, 2};

    auto param = make_shared<op::Parameter>(element::f32, input_shape);
    auto sl = make_shared<op::Slice>(param, lower_bounds, upper_bounds, strides);

    ASSERT_EQ(sl->get_element_type(), element::f32);
    ASSERT_EQ(sl->get_shape(), (Shape{0, 1, 2, 2}));
}

TEST(type_prop, slice_partial_arg_rank_static_dynamic_some_dims_known_ok)
{
    PartialShape input_shape{2, 4, 10, Dimension::dynamic()};
    Coordinate lower_bounds{1, 2, 3, 4};
    Coordinate upper_bounds{1, 3, 5, 7};
    Strides strides{1, 1, 1, 2};

    auto param = make_shared<op::Parameter>(element::f32, input_shape);
    auto sl = make_shared<op::Slice>(param, lower_bounds, upper_bounds, strides);

    ASSERT_EQ(sl->get_element_type(), element::f32);
    ASSERT_EQ(sl->get_shape(), (Shape{0, 1, 2, 2}));
}

TEST(type_prop, slice_partial_arg_rank_static_dynamic_attribs_rank_mismatches_arg)
{
    PartialShape input_shape{Dimension::dynamic(),
                             Dimension::dynamic(),
                             Dimension::dynamic(),
                             Dimension::dynamic(),
                             Dimension::dynamic()};
    Coordinate lower_bounds{1, 2, 3, 4};
    Coordinate upper_bounds{1, 3, 5, 7};
    Strides strides{1, 1, 1, 2};

    auto param = make_shared<op::Parameter>(element::f32, input_shape);
    try
    {
        auto sl = make_shared<op::Slice>(param, lower_bounds, upper_bounds, strides);
        // Should have thrown, so fail if it didn't
        FAIL() << "Mismatch of attrib ranks with arg ranks not detected (argument rank-static "
                  "dynamic)";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Input rank does not match the "
                                         "rank of the lower bounds (Coordinate{1, 2, "
                                         "3, 4}), upper bounds (Coordinate{1, 3, 5, "
                                         "7}), and strides (Strides{1, 1, 1, 2})"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, slice_partial_arg_rank_static_dynamic_some_dims_known_upper_bounds_oob)
{
    PartialShape input_shape{2, 2, 10, Dimension::dynamic()};
    Coordinate lower_bounds{1, 2, 3, 4};
    Coordinate upper_bounds{1, 3, 5, 7};
    Strides strides{1, 1, 1, 2};

    auto param = make_shared<op::Parameter>(element::f32, input_shape);
    try
    {
        auto sl = make_shared<op::Slice>(param, lower_bounds, upper_bounds, strides);
        // Should have thrown, so fail if it didn't
        FAIL() << "Upper bounds out of bounds not detected (argument rank-static dynamic)";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Upper bound for slice at axis 1 is out of "
                                         "range (upper bounds: Coordinate{1, 3, 5, "
                                         "7}, argument shape: {2,2,10,?})"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, scalar_constant_deduce_float32)
{
    auto c = op::Constant::create(element::f32, Shape{}, {208});
    ASSERT_EQ(c->get_element_type(), element::f32);
    ASSERT_EQ(c->get_shape(), (Shape{}));
}

TEST(type_prop, scalar_constant_deduce_bool)
{
    auto c = op::Constant::create(element::boolean, Shape{}, {1});
    ASSERT_EQ(c->get_element_type(), element::boolean);
    ASSERT_EQ(c->get_shape(), (Shape{}));
}

TEST(type_prop, tensor_constant_deduce_float32)
{
    auto c = op::Constant::create(element::f32, Shape{2, 2}, {208, 208, 208, 208});
    ASSERT_EQ(c->get_element_type(), element::f32);
    ASSERT_EQ(c->get_shape(), (Shape{2, 2}));
}

TEST(type_prop, tensor_constant_deduce_bool)
{
    auto c = op::Constant::create(element::boolean, Shape{2, 2}, {1, 1, 1, 1});
    ASSERT_EQ(c->get_element_type(), element::boolean);
    ASSERT_EQ(c->get_shape(), (Shape{2, 2}));
}

TEST(type_prop, tensor_constant_bad_count)
{
    try
    {
        auto c = op::Constant::create(element::boolean, Shape{2, 2}, {1, 1, 1});
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect number of literals not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Did not get the expected number of literals for a "
                                         "constant of shape Shape{2, 2} (got 3, expected 1 or 4)"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, replace_slice_deduce_vector)
{
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{3});
    auto rsl = make_shared<op::ReplaceSlice>(param0, param1, Coordinate{2}, Coordinate{5});
    ASSERT_EQ(rsl->get_element_type(), element::f32);
    ASSERT_EQ(rsl->get_shape(), (Shape{6}));
}

TEST(type_prop, replace_slice_deduce_matrix)
{
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6, 8});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{3, 6});
    auto rsl = make_shared<op::ReplaceSlice>(param0, param1, Coordinate{2, 1}, Coordinate{5, 7});
    ASSERT_EQ(rsl->get_element_type(), element::f32);
    ASSERT_EQ(rsl->get_shape(), (Shape{6, 8}));
}

TEST(type_prop, replace_slice_deduce_matrix_strided)
{
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6, 8});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{1, 3});
    auto rsl = make_shared<op::ReplaceSlice>(
        param0, param1, Coordinate{2, 1}, Coordinate{5, 7}, Strides{3, 2});
    ASSERT_EQ(rsl->get_element_type(), element::f32);
    ASSERT_EQ(rsl->get_shape(), (Shape{6, 8}));
}

TEST(type_prop, replace_slice_deduce_matrix_strided_uneven)
{
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6, 8});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{1, 2});
    auto rsl = make_shared<op::ReplaceSlice>(
        param0, param1, Coordinate{2, 1}, Coordinate{5, 7}, Strides{3, 4});
    ASSERT_EQ(rsl->get_element_type(), element::f32);
    ASSERT_EQ(rsl->get_shape(), (Shape{6, 8}));
}

TEST(type_prop, replace_slice_deduce_vector_edge)
{
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{6});
    auto rsl = make_shared<op::ReplaceSlice>(param0, param1, Coordinate{0}, Coordinate{6});
    ASSERT_EQ(rsl->get_element_type(), element::f32);
    ASSERT_EQ(rsl->get_shape(), (Shape{6}));
}

TEST(type_prop, replace_slice_deduce_matrix_edge)
{
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6, 8});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{6, 8});
    auto rsl = make_shared<op::ReplaceSlice>(param0, param1, Coordinate{0, 0}, Coordinate{6, 8});
    ASSERT_EQ(rsl->get_element_type(), element::f32);
    ASSERT_EQ(rsl->get_shape(), (Shape{6, 8}));
}

TEST(type_prop, replace_slice_deduce_matrix_zero_cols)
{
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6, 8});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{6, 0});
    auto rsl = make_shared<op::ReplaceSlice>(param0, param1, Coordinate{0, 0}, Coordinate{6, 0});
    ASSERT_EQ(rsl->get_element_type(), element::f32);
    ASSERT_EQ(rsl->get_shape(), (Shape{6, 8}));
}

TEST(type_prop, replace_slice_deduce_matrix_zero_zero)
{
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6, 8});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{0, 0});
    auto rsl = make_shared<op::ReplaceSlice>(param0, param1, Coordinate{0, 0}, Coordinate{0, 0});
    ASSERT_EQ(rsl->get_element_type(), element::f32);
    ASSERT_EQ(rsl->get_shape(), (Shape{6, 8}));
}

TEST(type_prop, replace_slice_deduce_vector_invalid_strides)
{
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{4});
    try
    {
        auto sl = make_shared<op::ReplaceSlice>(
            param0, param1, Coordinate{0}, Coordinate{7}, Strides{1, 2});
        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid slice strides not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Ranks of lower bounds (Coordinate{0}), upper bounds "
                        "(Coordinate{7}) and strides (Strides{1, 2}) do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, replace_slice_deduce_matrix_arg_rank_mismatch)
{
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6, 8});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{3, 6, 5});
    try
    {
        auto rsl =
            make_shared<op::ReplaceSlice>(param0, param1, Coordinate{2, 1}, Coordinate{5, 7});
        // Should have thrown, so fail if it didn't
        FAIL() << "Argument rank mismatch not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Argument ranks do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, replace_slice_deduce_matrix_arg_element_type_mismatch)
{
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6, 8});
    auto param1 = make_shared<op::Parameter>(element::i32, Shape{3, 6});
    try
    {
        auto rsl =
            make_shared<op::ReplaceSlice>(param0, param1, Coordinate{2, 1}, Coordinate{5, 7});
        // Should have thrown, so fail if it didn't
        FAIL() << "Argument element type mismatch not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Argument element types do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, replace_slice_deduce_matrix_slice_shape_mismatch)
{
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6, 8});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{3, 6});
    try
    {
        auto rsl =
            make_shared<op::ReplaceSlice>(param0, param1, Coordinate{1, 1}, Coordinate{5, 7});
        // Should have thrown, so fail if it didn't
        FAIL() << "Slice shape mismatch not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string(
                "Shape of replacement tensor ({3,6}) does not match the slice shape ({4,6})"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, replace_slice_deduce_matrix_slice_shape_mismatch_strided)
{
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6, 8});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{4, 6});
    try
    {
        auto rsl = make_shared<op::ReplaceSlice>(
            param0, param1, Coordinate{1, 1}, Coordinate{5, 7}, Strides{1, 2});
        // Should have thrown, so fail if it didn't
        FAIL() << "Slice shape mismatch not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string(
                "Shape of replacement tensor ({4,6}) does not match the slice shape ({4,3})"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, replace_slice_deduce_vector_edge_upper_oob)
{
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{7});
    try
    {
        auto rsl = make_shared<op::ReplaceSlice>(param0, param1, Coordinate{0}, Coordinate{7});
        // Should have thrown, so fail if it didn't
        FAIL() << "Upper bound out of range not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Upper bound for slice at axis 0 is out of range"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, replace_slice_deduce_matrix_edge_upper_oob)
{
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6, 8});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{6, 9});
    try
    {
        auto rsl =
            make_shared<op::ReplaceSlice>(param0, param1, Coordinate{0, 0}, Coordinate{6, 9});
        // Should have thrown, so fail if it didn't
        FAIL() << "Upper bound out of range not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Upper bound for slice at axis 1 is out of range"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, replace_slice_deduce_vector_lower_above_upper)
{
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{0});
    try
    {
        auto rsl = make_shared<op::ReplaceSlice>(param0, param1, Coordinate{3}, Coordinate{2});
        // Should have thrown, so fail if it didn't
        FAIL() << "Lower bound above upper not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Lower bound for slice is greater than upper bound at axis 0"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, replace_slice_deduce_matrix_lower_above_upper)
{
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6, 8});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{6, 0});
    try
    {
        auto rsl =
            make_shared<op::ReplaceSlice>(param0, param1, Coordinate{0, 5}, Coordinate{6, 4});
        // Should have thrown, so fail if it didn't
        FAIL() << "Lower bound above upper not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Lower bound for slice is greater than upper bound at axis 1"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, replace_slice_deduce_matrix_lower_missing)
{
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6, 8});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{6, 6});
    try
    {
        auto rsl = make_shared<op::ReplaceSlice>(param0, param1, Coordinate{0}, Coordinate{5, 5});
        // Should have thrown, so fail if it didn't
        FAIL() << "Missing lower bound coordinate not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Ranks of lower bounds (Coordinate{0}), upper bounds "
                        "(Coordinate{5, 5}) and strides (Strides{1}) do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, replace_slice_deduce_matrix_upper_missing)
{
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6, 8});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{6, 6});
    try
    {
        auto rsl = make_shared<op::ReplaceSlice>(param0, param1, Coordinate{0, 0}, Coordinate{5});
        // Should have thrown, so fail if it didn't
        FAIL() << "Missing upper bound coordinate not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Ranks of lower bounds (Coordinate{0, 0}), upper bounds "
                        "(Coordinate{5}) and strides (Strides{1, 1}) do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, replace_slice_deduce_matrix_lower_extra)
{
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6, 8});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{6, 6});
    try
    {
        auto rsl =
            make_shared<op::ReplaceSlice>(param0, param1, Coordinate{0, 0, 0}, Coordinate{5, 5});
        // Should have thrown, so fail if it didn't
        FAIL() << "Extra lower bound coordinate not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Ranks of lower bounds (Coordinate{0, 0, "
                                         "0}), upper bounds (Coordinate{5, 5}) and "
                                         "strides (Strides{1, 1, 1}) do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, replace_slice_deduce_matrix_upper_extra)
{
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6, 8});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{6, 6});
    try
    {
        auto rsl =
            make_shared<op::ReplaceSlice>(param0, param1, Coordinate{0, 0}, Coordinate{5, 5, 5});
        // Should have thrown, so fail if it didn't
        FAIL() << "Extra upper bound coordinate not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Ranks of lower bounds (Coordinate{0, 0}), "
                                         "upper bounds (Coordinate{5, 5, 5}) and "
                                         "strides (Strides{1, 1}) do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, replace_slice_partial_input_rank_dynamic_replacement_rank_dynamic_attribs_ok)
{
    PartialShape input_shape{PartialShape::dynamic()};
    PartialShape replacement_shape{PartialShape::dynamic()};
    Coordinate lower_bounds{1, 2, 3, 4};
    Coordinate upper_bounds{1, 3, 5, 7};
    Strides strides{1, 1, 1, 2};

    auto param0 = make_shared<op::Parameter>(element::f32, input_shape);
    auto param1 = make_shared<op::Parameter>(element::f32, replacement_shape);
    auto rsl = make_shared<op::ReplaceSlice>(param0, param1, lower_bounds, upper_bounds, strides);

    ASSERT_EQ(rsl->get_element_type(), element::f32);
    ASSERT_TRUE(rsl->get_output_partial_shape(0).same_scheme(PartialShape{
        Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()}));
}

TEST(type_prop,
     replace_slice_partial_input_rank_dynamic_replacement_rank_dynamic_attribs_rank_mismatch)
{
    PartialShape input_shape{PartialShape::dynamic()};
    PartialShape replacement_shape{PartialShape::dynamic()};
    Coordinate lower_bounds{1, 2, 3, 4};
    Coordinate upper_bounds{1, 3, 5};
    Strides strides{1, 1, 1, 2};

    auto param0 = make_shared<op::Parameter>(element::f32, input_shape);
    auto param1 = make_shared<op::Parameter>(element::f32, replacement_shape);
    try
    {
        auto rsl =
            make_shared<op::ReplaceSlice>(param0, param1, lower_bounds, upper_bounds, strides);
        // Should have thrown, so fail if it didn't
        FAIL() << "Mismatch of lower-bounds/upper-bounds/strides ranks not detected (argument "
                  "rank-dynamic)";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Ranks of lower bounds (Coordinate{1, 2, 3, 4}), upper bounds "
                        "(Coordinate{1, 3, 5}) and strides (Strides{1, 1, 1, 2}) do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop,
     replace_slice_partial_input_rank_dynamic_replacement_rank_dynamic_attribs_bounds_crossing)
{
    PartialShape input_shape{PartialShape::dynamic()};
    PartialShape replacement_shape{PartialShape::dynamic()};
    Coordinate lower_bounds{1, 2, 3, 8};
    Coordinate upper_bounds{1, 3, 5, 7};
    Strides strides{1, 1, 1, 2};

    auto param0 = make_shared<op::Parameter>(element::f32, input_shape);
    auto param1 = make_shared<op::Parameter>(element::f32, replacement_shape);
    try
    {
        auto rsl =
            make_shared<op::ReplaceSlice>(param0, param1, lower_bounds, upper_bounds, strides);
        // Should have thrown, so fail if it didn't
        FAIL() << "Crossing lower/upper bounds not detected (argument rank-dynamic)";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Lower bound for slice is greater than upper bound at axis 3 (lower "
                        "bounds: Coordinate{1, 2, 3, 8}, upper bounds: Coordinate{1, 3, 5, 7})"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, replace_slice_partial_input_rank_static_dynamic_replacement_rank_dynamic_ok)
{
    PartialShape input_shape{
        Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()};
    PartialShape replacement_shape{PartialShape::dynamic()};
    Coordinate lower_bounds{1, 2, 3, 4};
    Coordinate upper_bounds{1, 3, 5, 7};
    Strides strides{1, 1, 1, 2};

    auto param0 = make_shared<op::Parameter>(element::f32, input_shape);
    auto param1 = make_shared<op::Parameter>(element::f32, replacement_shape);
    auto rsl = make_shared<op::ReplaceSlice>(param0, param1, lower_bounds, upper_bounds, strides);

    ASSERT_EQ(rsl->get_element_type(), element::f32);
    ASSERT_TRUE(rsl->get_output_partial_shape(0).same_scheme(PartialShape{
        Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()}));
}

TEST(type_prop,
     replace_slice_partial_input_rank_static_dynamic_some_dims_known_replacement_rank_dynamic_ok)
{
    PartialShape input_shape{2, 4, 10, Dimension::dynamic()};
    PartialShape replacement_shape{PartialShape::dynamic()};
    Coordinate lower_bounds{1, 2, 3, 4};
    Coordinate upper_bounds{1, 3, 5, 7};
    Strides strides{1, 1, 1, 2};

    auto param0 = make_shared<op::Parameter>(element::f32, input_shape);
    auto param1 = make_shared<op::Parameter>(element::f32, replacement_shape);
    auto rsl = make_shared<op::ReplaceSlice>(param0, param1, lower_bounds, upper_bounds, strides);

    ASSERT_EQ(rsl->get_element_type(), element::f32);
    ASSERT_TRUE(
        rsl->get_output_partial_shape(0).same_scheme(PartialShape{2, 4, 10, Dimension::dynamic()}));
}

TEST(
    type_prop,
    replace_slice_partial_input_rank_static_dynamic_replacement_rank_dynamic_attribs_rank_mismatches_input)
{
    PartialShape input_shape{Dimension::dynamic(),
                             Dimension::dynamic(),
                             Dimension::dynamic(),
                             Dimension::dynamic(),
                             Dimension::dynamic()};
    PartialShape replacement_shape{PartialShape::dynamic()};
    Coordinate lower_bounds{1, 2, 3, 4};
    Coordinate upper_bounds{1, 3, 5, 7};
    Strides strides{1, 1, 1, 2};

    auto param0 = make_shared<op::Parameter>(element::f32, input_shape);
    auto param1 = make_shared<op::Parameter>(element::f32, replacement_shape);
    try
    {
        auto rsl =
            make_shared<op::ReplaceSlice>(param0, param1, lower_bounds, upper_bounds, strides);
        // Should have thrown, so fail if it didn't
        FAIL() << "Mismatch of attrib ranks with arg ranks not detected (argument rank-static "
                  "dynamic)";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Argument ranks do not match the rank of the lower bounds "
                                         "(Coordinate{1, 2, 3, 4}), upper bounds (Coordinate{1, 3, "
                                         "5, 7}), and strides (Strides{1, 1, 1, 2})"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(
    type_prop,
    replace_slice_partial_input_rank_static_dynamic_some_dims_known_replacement_rank_dynamic_upper_bounds_oob)
{
    PartialShape input_shape{2, 2, 10, Dimension::dynamic()};
    PartialShape replacement_shape{PartialShape::dynamic()};
    Coordinate lower_bounds{1, 2, 3, 4};
    Coordinate upper_bounds{1, 3, 5, 7};
    Strides strides{1, 1, 1, 2};

    auto param0 = make_shared<op::Parameter>(element::f32, input_shape);
    auto param1 = make_shared<op::Parameter>(element::f32, replacement_shape);
    try
    {
        auto rsl =
            make_shared<op::ReplaceSlice>(param0, param1, lower_bounds, upper_bounds, strides);
        // Should have thrown, so fail if it didn't
        FAIL() << "Upper bounds out of bounds not detected (argument rank-static dynamic)";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Upper bound for slice at axis 1 is out of "
                                         "range (upper bounds: Coordinate{1, 3, 5, "
                                         "7}, argument shape: {2,2,10,?})"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, replace_slice_partial_input_rank_dynamic_replacement_rank_static_dynamic_ok)
{
    PartialShape input_shape{PartialShape::dynamic()};
    PartialShape replacement_shape{
        Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()};
    Coordinate lower_bounds{1, 2, 3, 4};
    Coordinate upper_bounds{1, 3, 5, 7};
    Strides strides{1, 1, 1, 2};

    auto param0 = make_shared<op::Parameter>(element::f32, input_shape);
    auto param1 = make_shared<op::Parameter>(element::f32, replacement_shape);
    auto rsl = make_shared<op::ReplaceSlice>(param0, param1, lower_bounds, upper_bounds, strides);

    ASSERT_EQ(rsl->get_element_type(), element::f32);
    ASSERT_TRUE(rsl->get_output_partial_shape(0).same_scheme(PartialShape{
        Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()}));
}

TEST(type_prop,
     replace_slice_partial_input_rank_dynamic_replacement_rank_static_dynamic_some_dims_known_ok)
{
    PartialShape input_shape{PartialShape::dynamic()};
    PartialShape replacement_shape{0, Dimension::dynamic(), Dimension::dynamic(), 2};
    Coordinate lower_bounds{1, 2, 3, 4};
    Coordinate upper_bounds{1, 3, 5, 7};
    Strides strides{1, 1, 1, 2};

    auto param0 = make_shared<op::Parameter>(element::f32, input_shape);
    auto param1 = make_shared<op::Parameter>(element::f32, replacement_shape);
    auto rsl = make_shared<op::ReplaceSlice>(param0, param1, lower_bounds, upper_bounds, strides);

    ASSERT_EQ(rsl->get_element_type(), element::f32);
    ASSERT_TRUE(rsl->get_output_partial_shape(0).same_scheme(PartialShape{
        Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()}));
}

TEST(
    type_prop,
    replace_slice_partial_input_rank_dynamic_replacement_rank_static_dynamic_some_dims_known_attribs_mismatch_replacement_shape)
{
    PartialShape input_shape{PartialShape::dynamic()};
    PartialShape replacement_shape{1, Dimension::dynamic(), Dimension::dynamic(), 2};
    Coordinate lower_bounds{1, 2, 3, 4};
    Coordinate upper_bounds{1, 3, 5, 7};
    Strides strides{1, 1, 1, 2};

    auto param0 = make_shared<op::Parameter>(element::f32, input_shape);
    auto param1 = make_shared<op::Parameter>(element::f32, replacement_shape);
    try
    {
        auto rsl =
            make_shared<op::ReplaceSlice>(param0, param1, lower_bounds, upper_bounds, strides);
        // Should have thrown, so fail if it didn't
        FAIL() << "Mismatch of shape inferred from attributes with provided replacement shape not "
                  "detected (rank-dynamic/rank-static dynamic inputs)";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Shape of replacement tensor ({1,?,?,2}) does not match "
                                         "the slice shape ({0,1,2,2})"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(
    type_prop,
    replace_slice_partial_input_rank_dynamic_replacement_rank_static_dynamic_attribs_rank_mismatches_replacement)
{
    PartialShape input_shape{PartialShape::dynamic()};
    PartialShape replacement_shape{Dimension::dynamic(),
                                   Dimension::dynamic(),
                                   Dimension::dynamic(),
                                   Dimension::dynamic(),
                                   Dimension::dynamic()};
    Coordinate lower_bounds{1, 2, 3, 4};
    Coordinate upper_bounds{1, 3, 5, 7};
    Strides strides{1, 1, 1, 2};

    auto param0 = make_shared<op::Parameter>(element::f32, input_shape);
    auto param1 = make_shared<op::Parameter>(element::f32, replacement_shape);
    try
    {
        auto rsl =
            make_shared<op::ReplaceSlice>(param0, param1, lower_bounds, upper_bounds, strides);
        // Should have thrown, so fail if it didn't
        FAIL() << "Mismatch of attrib ranks with arg ranks not detected (arguments "
                  "rank-dynamic/rank-static "
                  "dynamic)";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Argument ranks do not match the rank of the lower bounds "
                                         "(Coordinate{1, 2, 3, 4}), upper bounds (Coordinate{1, 3, "
                                         "5, 7}), and strides (Strides{1, 1, 1, 2})"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(
    type_prop,
    replace_slice_partial_input_rank_static_dynamic_replacement_rank_static_dynamic_argument_ranks_mismatch)
{
    PartialShape input_shape{
        Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()};
    PartialShape replacement_shape{Dimension::dynamic(),
                                   Dimension::dynamic(),
                                   Dimension::dynamic(),
                                   Dimension::dynamic(),
                                   Dimension::dynamic()};
    Coordinate lower_bounds{1, 2, 3, 4};
    Coordinate upper_bounds{1, 3, 5, 7};
    Strides strides{1, 1, 1, 2};

    auto param0 = make_shared<op::Parameter>(element::f32, input_shape);
    auto param1 = make_shared<op::Parameter>(element::f32, replacement_shape);
    try
    {
        auto rsl =
            make_shared<op::ReplaceSlice>(param0, param1, lower_bounds, upper_bounds, strides);
        // Should have thrown, so fail if it didn't
        FAIL() << "Mismatching input/replacement ranks not detected (arguments both rank-static "
                  "dynamic)";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Argument ranks do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, one_hot_deduce_scalar)
{
    auto param = make_shared<op::Parameter>(element::i32, Shape{});
    auto oh = make_shared<op::OneHot>(param, Shape{9}, 0);
    ASSERT_EQ(oh->get_element_type(), element::i32);
    ASSERT_EQ(oh->get_shape(), (Shape{9}));
}

TEST(type_prop, one_hot_deduce_vector_0)
{
    auto param = make_shared<op::Parameter>(element::i32, Shape{8});
    auto oh = make_shared<op::OneHot>(param, Shape{9, 8}, 0);
    ASSERT_EQ(oh->get_element_type(), element::i32);
    ASSERT_EQ(oh->get_shape(), (Shape{9, 8}));
}

TEST(type_prop, one_hot_deduce_vector_1)
{
    auto param = make_shared<op::Parameter>(element::i32, Shape{8});
    auto oh = make_shared<op::OneHot>(param, Shape{8, 9}, 1);
    ASSERT_EQ(oh->get_element_type(), element::i32);
    ASSERT_EQ(oh->get_shape(), (Shape{8, 9}));
}

TEST(type_prop, one_hot_deduce_matrix_0)
{
    auto param = make_shared<op::Parameter>(element::i32, Shape{12, 24});
    auto oh = make_shared<op::OneHot>(param, Shape{2, 12, 24}, 0);
    ASSERT_EQ(oh->get_element_type(), element::i32);
    ASSERT_EQ(oh->get_shape(), (Shape{2, 12, 24}));
}

TEST(type_prop, one_hot_deduce_matrix_1)
{
    auto param = make_shared<op::Parameter>(element::i32, Shape{12, 24});
    auto oh = make_shared<op::OneHot>(param, Shape{12, 2, 24}, 1);
    ASSERT_EQ(oh->get_element_type(), element::i32);
    ASSERT_EQ(oh->get_shape(), (Shape{12, 2, 24}));
}

TEST(type_prop, one_hot_deduce_matrix_2)
{
    auto param = make_shared<op::Parameter>(element::i32, Shape{12, 24});
    auto oh = make_shared<op::OneHot>(param, Shape{12, 24, 2}, 2);
    ASSERT_EQ(oh->get_element_type(), element::i32);
    ASSERT_EQ(oh->get_shape(), (Shape{12, 24, 2}));
}

TEST(type_prop, one_hot_deduce_floating_point)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{12, 24});
    auto oh = make_shared<op::OneHot>(param, Shape{12, 24, 8}, 2);
    ASSERT_EQ(oh->get_element_type(), element::f32);
    ASSERT_EQ(oh->get_shape(), (Shape{12, 24, 8}));
}

TEST(type_prop, one_hot_deduce_axis_oob)
{
    auto param = make_shared<op::Parameter>(element::i32, Shape{12, 24});
    try
    {
        auto oh = make_shared<op::OneHot>(param, Shape{12, 24, 8}, 3);
        // Should have thrown, so fail if it didn't
        FAIL() << "One-hot axis out of bounds not detected.";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("One-hot axis (3) is out of bounds"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, one_hot_deduce_shape_incompatible)
{
    auto param = make_shared<op::Parameter>(element::i32, Shape{12, 24});
    try
    {
        auto oh = make_shared<op::OneHot>(param, Shape{12, 22, 8}, 2);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incompatible one-hot output shape not detected.";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(), std::string("Argument shape {12,24} does not match the expected shape"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, one_hot_partial_rank_dynamic_rank_dynamic)
{
    PartialShape input_shape{PartialShape::dynamic()};
    PartialShape requested_shape{PartialShape::dynamic()};
    size_t one_hot_axis{3000};

    auto param = make_shared<op::Parameter>(element::f32, input_shape);
    try
    {
        auto oh = make_shared<op::OneHot>(param, requested_shape, one_hot_axis);
        // Should have thrown, so fail if it didn't
        FAIL() << "Dynamic rank for requested result shape not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Requested result shape has dynamic rank"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, one_hot_partial_rank_dynamic_rank_static_dynamic_ok)
{
    PartialShape input_shape{PartialShape::dynamic()};
    PartialShape requested_shape{Dimension::dynamic(), 2, 3, Dimension::dynamic()};
    size_t one_hot_axis{2};

    auto param = make_shared<op::Parameter>(element::f32, input_shape);
    auto oh = make_shared<op::OneHot>(param, requested_shape, one_hot_axis);

    ASSERT_EQ(oh->get_output_element_type(0), element::f32);
    ASSERT_TRUE(oh->get_output_partial_shape(0).same_scheme(
        PartialShape{Dimension::dynamic(), 2, 3, Dimension::dynamic()}));
}

TEST(type_prop, one_hot_partial_rank_dynamic_rank_static_dynamic_one_hot_dim_dynamic)
{
    PartialShape input_shape{PartialShape::dynamic()};
    PartialShape requested_shape{Dimension::dynamic(), 2, 3, Dimension::dynamic()};
    size_t one_hot_axis{3};

    auto param = make_shared<op::Parameter>(element::f32, input_shape);
    try
    {
        auto oh = make_shared<op::OneHot>(param, requested_shape, one_hot_axis);
        // Should have thrown, so fail if it didn't
        FAIL() << "Dynamic one-hot dimension not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Requested result shape ({?,2,3,?}) has dynamic dimension "
                                         "at the one-hot axis (3)"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, one_hot_partial_rank_dynamic_rank_static_dynamic_one_hot_axis_oob)
{
    PartialShape input_shape{PartialShape::dynamic()};
    PartialShape requested_shape{Dimension::dynamic(), 2, 3, Dimension::dynamic()};
    size_t one_hot_axis{4};

    auto param = make_shared<op::Parameter>(element::f32, input_shape);
    try
    {
        auto oh = make_shared<op::OneHot>(param, requested_shape, one_hot_axis);
        // Should have thrown, so fail if it didn't
        FAIL() << "One-hot axis out of bounds not detected (rank-dynamic argument, rank-static "
                  "dynamic result shape)";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("One-hot axis (4) is out of bounds (requested result shape: {?,2,3,?})"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, one_hot_partial_rank_static_dynamic_rank_static_dynamic_ok)
{
    PartialShape input_shape{3, Dimension::dynamic(), Dimension::dynamic(), 4};
    PartialShape requested_shape{Dimension::dynamic(), 2, 3, Dimension::dynamic(), 4};
    size_t one_hot_axis{2};

    auto param = make_shared<op::Parameter>(element::f32, input_shape);
    auto oh = make_shared<op::OneHot>(param, requested_shape, one_hot_axis);

    ASSERT_EQ(oh->get_output_element_type(0), element::f32);
    ASSERT_TRUE(oh->get_output_partial_shape(0).same_scheme(
        PartialShape{3, 2, 3, Dimension::dynamic(), 4}));
}

TEST(type_prop,
     one_hot_partial_rank_static_dynamic_rank_static_dynamic_incompatible_rank_input_short)
{
    PartialShape input_shape{3, Dimension::dynamic(), Dimension::dynamic()};
    PartialShape requested_shape{Dimension::dynamic(), 2, 3, Dimension::dynamic(), 4};
    size_t one_hot_axis{2};

    auto param = make_shared<op::Parameter>(element::f32, input_shape);
    try
    {
        auto oh = make_shared<op::OneHot>(param, requested_shape, one_hot_axis);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incompatible input/output ranks not detected (rank-static dynamic argument, "
                  "rank-static dynamic result shape)";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Argument shape {3,?,?} does not match the expected shape of {?,2,?,4}"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop,
     one_hot_partial_rank_static_dynamic_rank_static_dynamic_incompatible_rank_input_long)
{
    PartialShape input_shape{3, Dimension::dynamic(), Dimension::dynamic(), 4, 5};
    PartialShape requested_shape{Dimension::dynamic(), 2, 3, Dimension::dynamic(), 4};
    size_t one_hot_axis{2};

    auto param = make_shared<op::Parameter>(element::f32, input_shape);
    try
    {
        auto oh = make_shared<op::OneHot>(param, requested_shape, one_hot_axis);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incompatible input/output ranks not detected (rank-static dynamic argument, "
                  "rank-static dynamic result shape)";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string(
                "Argument shape {3,?,?,4,5} does not match the expected shape of {?,2,?,4}"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, one_hot_partial_rank_static_dynamic_rank_static_dynamic_incompatible_dim)
{
    PartialShape input_shape{3, Dimension::dynamic(), Dimension::dynamic(), 5};
    PartialShape requested_shape{Dimension::dynamic(), 2, 3, Dimension::dynamic(), 4};
    size_t one_hot_axis{2};

    auto param = make_shared<op::Parameter>(element::f32, input_shape);
    try
    {
        auto oh = make_shared<op::OneHot>(param, requested_shape, one_hot_axis);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incompatible input/output dimensions not detected (rank-static dynamic "
                  "argument, rank-static dynamic result shape)";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Argument shape {3,?,?,5} does not match the expected shape of {?,2,?,4}"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, one_hot_partial_rank_static_dynamic_rank_static_dynamic_one_hot_dim_dynamic)
{
    PartialShape input_shape{3, Dimension::dynamic(), Dimension::dynamic(), 4};
    PartialShape requested_shape{
        Dimension::dynamic(), 2, Dimension::dynamic(), Dimension::dynamic(), 4};
    size_t one_hot_axis{2};

    auto param = make_shared<op::Parameter>(element::f32, input_shape);
    try
    {
        auto oh = make_shared<op::OneHot>(param, requested_shape, one_hot_axis);
        // Should have thrown, so fail if it didn't
        FAIL() << "Dynamic one-hot dimension not detected (rank-static dynamic argument, "
                  "rank-static dynamic result shape)";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Requested result shape ({?,2,?,?,4}) has dynamic "
                                         "dimension at the one-hot axis (2)"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, one_hot_partial_rank_static_dynamic_rank_static_dynamic_one_hot_axis_oob)
{
    PartialShape input_shape{3, Dimension::dynamic(), Dimension::dynamic(), 4};
    PartialShape requested_shape{
        Dimension::dynamic(), 2, Dimension::dynamic(), Dimension::dynamic(), 4};
    size_t one_hot_axis{2};

    auto param = make_shared<op::Parameter>(element::f32, input_shape);
    try
    {
        auto oh = make_shared<op::OneHot>(param, requested_shape, one_hot_axis);
        // Should have thrown, so fail if it didn't
        FAIL() << "One-hot axis out of bounds not detected (rank-static dynamic argument, "
                  "rank-static dynamic result shape)";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Requested result shape ({?,2,?,?,4}) has dynamic "
                                         "dimension at the one-hot axis (2)"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_1d_deduce)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{64, 3, 100});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 10});
    auto conv = make_shared<op::Convolution>(param0, param1);
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), (Shape{64, 128, 91}));

    EXPECT_EQ(conv->get_window_movement_strides(), Strides{1});
    EXPECT_EQ(conv->get_window_dilation_strides(), Strides{1});
    EXPECT_EQ(conv->get_data_dilation_strides(), Strides{1});

    EXPECT_EQ(conv->get_padding_below(), CoordinateDiff{0});
    EXPECT_EQ(conv->get_padding_above(), CoordinateDiff{0});
}

TEST(type_prop, conv_1d_back_data_batch_deduce)
{
    // Deduce type
    Shape data_batch_shape{64, 3, 100};
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 10});  // filters
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{64, 128, 91}); // output delta
    auto conv = make_shared<op::ConvolutionBackpropData>(data_batch_shape,
                                                         param0,
                                                         param1,
                                                         Strides{1},
                                                         Strides{1},
                                                         CoordinateDiff{0},
                                                         CoordinateDiff{0},
                                                         Strides{1});
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), data_batch_shape);

    EXPECT_EQ(conv->get_window_movement_strides_forward(), Strides{1});
    EXPECT_EQ(conv->get_window_dilation_strides_forward(), Strides{1});
    EXPECT_EQ(conv->get_data_dilation_strides_forward(), Strides{1});

    EXPECT_EQ(conv->get_padding_below_forward(), CoordinateDiff{0});
    EXPECT_EQ(conv->get_padding_above_forward(), CoordinateDiff{0});
}

TEST(type_prop, conv_1d_back_filters_deduce)
{
    // Deduce type
    // Shape data_batch_shape{64, 3, 100};
    Shape filters_shape{128, 3, 10};
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{64, 3, 100});  // data batch
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{64, 128, 91}); // output delta
    auto conv = make_shared<op::ConvolutionBackpropFilters>(param0,
                                                            filters_shape,
                                                            param1,
                                                            Strides{1},
                                                            Strides{1},
                                                            CoordinateDiff{0},
                                                            CoordinateDiff{0},
                                                            Strides{1});
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), filters_shape);

    EXPECT_EQ(conv->get_window_movement_strides_forward(), Strides{1});
    EXPECT_EQ(conv->get_window_dilation_strides_forward(), Strides{1});
    EXPECT_EQ(conv->get_data_dilation_strides_forward(), Strides{1});

    EXPECT_EQ(conv->get_padding_below_forward(), CoordinateDiff{0});
    EXPECT_EQ(conv->get_padding_above_forward(), CoordinateDiff{0});
}

TEST(type_prop, conv_1d_deduce_padded)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{64, 3, 100});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 10});
    auto move_strides = Strides{1};
    auto dilation_strides = Strides{1};
    auto padding_below = CoordinateDiff{2};
    auto padding_above = CoordinateDiff{3};
    auto conv = make_shared<op::Convolution>(
        param0, param1, move_strides, dilation_strides, padding_below, padding_above);
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), (Shape{64, 128, 96}));

    EXPECT_EQ(conv->get_window_movement_strides(), Strides{1});
    EXPECT_EQ(conv->get_window_dilation_strides(), Strides{1});
    EXPECT_EQ(conv->get_data_dilation_strides(), Strides{1});

    EXPECT_EQ(conv->get_padding_below(), CoordinateDiff{2});
    EXPECT_EQ(conv->get_padding_above(), CoordinateDiff{3});
}

TEST(type_prop, conv_1d_back_data_batch_deduce_padded)
{
    // Deduce type
    Shape data_batch_shape{64, 3, 100};
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 10});  // filters
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{64, 128, 96}); // output delta
    auto move_strides = Strides{1};
    auto dilation_strides = Strides{1};
    auto padding_below = CoordinateDiff{2};
    auto padding_above = CoordinateDiff{3};
    auto conv = make_shared<op::ConvolutionBackpropData>(data_batch_shape,
                                                         param0,
                                                         param1,
                                                         move_strides,
                                                         dilation_strides,
                                                         padding_below,
                                                         padding_above,
                                                         Strides{1});
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), data_batch_shape);

    EXPECT_EQ(conv->get_window_movement_strides_forward(), Strides{1});
    EXPECT_EQ(conv->get_window_dilation_strides_forward(), Strides{1});
    EXPECT_EQ(conv->get_data_dilation_strides_forward(), Strides{1});

    EXPECT_EQ(conv->get_padding_below_forward(), CoordinateDiff{2});
    EXPECT_EQ(conv->get_padding_above_forward(), CoordinateDiff{3});
}

TEST(type_prop, conv_1d_back_filters_deduce_padded)
{
    // Deduce type
    // Shape data_batch_shape{64, 3, 100};
    Shape filters_shape{128, 3, 10};
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{64, 3, 100});  // data batch
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{64, 128, 96}); // output delta
    auto move_strides = Strides{1};
    auto dilation_strides = Strides{1};
    auto padding_below = CoordinateDiff{2};
    auto padding_above = CoordinateDiff{3};
    auto conv = make_shared<op::ConvolutionBackpropFilters>(param0,
                                                            filters_shape,
                                                            param1,
                                                            move_strides,
                                                            dilation_strides,
                                                            padding_below,
                                                            padding_above,
                                                            Strides{1});
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), filters_shape);

    EXPECT_EQ(conv->get_window_movement_strides_forward(), Strides{1});
    EXPECT_EQ(conv->get_window_dilation_strides_forward(), Strides{1});
    EXPECT_EQ(conv->get_data_dilation_strides_forward(), Strides{1});

    EXPECT_EQ(conv->get_padding_below_forward(), CoordinateDiff{2});
    EXPECT_EQ(conv->get_padding_above_forward(), CoordinateDiff{3});
}

TEST(type_prop, conv_1d_deduce_strided)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{64, 3, 100});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 10});
    auto move_strides = Strides{2};
    auto conv = make_shared<op::Convolution>(param0, param1, move_strides);
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), (Shape{64, 128, 46}));

    EXPECT_EQ(conv->get_window_movement_strides(), Strides{2});
    EXPECT_EQ(conv->get_window_dilation_strides(), Strides{1});
    EXPECT_EQ(conv->get_data_dilation_strides(), Strides{1});

    EXPECT_EQ(conv->get_padding_below(), CoordinateDiff{0});
    EXPECT_EQ(conv->get_padding_above(), CoordinateDiff{0});
}

TEST(type_prop, conv_1d_back_data_batch_deduce_strided)
{
    // Deduce type
    Shape data_batch_shape{64, 3, 100};
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 10});  // filters
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{64, 128, 46}); // output delta
    auto move_strides = Strides{2};
    auto conv = make_shared<op::ConvolutionBackpropData>(data_batch_shape,
                                                         param0,
                                                         param1,
                                                         move_strides,
                                                         Strides{1},
                                                         CoordinateDiff{0},
                                                         CoordinateDiff{0},
                                                         Strides{1});
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), data_batch_shape);

    EXPECT_EQ(conv->get_window_movement_strides_forward(), Strides{2});
    EXPECT_EQ(conv->get_window_dilation_strides_forward(), Strides{1});
    EXPECT_EQ(conv->get_data_dilation_strides_forward(), Strides{1});

    EXPECT_EQ(conv->get_padding_below_forward(), CoordinateDiff{0});
    EXPECT_EQ(conv->get_padding_above_forward(), CoordinateDiff{0});
}

TEST(type_prop, conv_1d_back_filters_deduce_strided)
{
    // Deduce type
    // Shape data_batch_shape{64, 3, 100};
    Shape filters_shape{128, 3, 10};
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{64, 3, 100});  // data batch
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{64, 128, 46}); // output delta
    auto move_strides = Strides{2};
    auto conv = make_shared<op::ConvolutionBackpropFilters>(param0,
                                                            filters_shape,
                                                            param1,
                                                            move_strides,
                                                            Strides{1},
                                                            CoordinateDiff{0},
                                                            CoordinateDiff{0},
                                                            Strides{1});
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), filters_shape);

    EXPECT_EQ(conv->get_window_movement_strides_forward(), Strides{2});
    EXPECT_EQ(conv->get_window_dilation_strides_forward(), Strides{1});
    EXPECT_EQ(conv->get_data_dilation_strides_forward(), Strides{1});

    EXPECT_EQ(conv->get_padding_below_forward(), CoordinateDiff{0});
    EXPECT_EQ(conv->get_padding_above_forward(), CoordinateDiff{0});
}

TEST(type_prop, conv_1d_deduce_strided_padded)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{64, 3, 100});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 10});
    auto move_strides = Strides{2};
    auto dilation_strides = Strides{1};
    auto padding_below = CoordinateDiff{2};
    auto padding_above = CoordinateDiff{3};
    auto conv = make_shared<op::Convolution>(
        param0, param1, move_strides, dilation_strides, padding_below, padding_above);
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), (Shape{64, 128, 48}));

    EXPECT_EQ(conv->get_window_movement_strides(), Strides{2});
    EXPECT_EQ(conv->get_window_dilation_strides(), Strides{1});
    EXPECT_EQ(conv->get_data_dilation_strides(), Strides{1});

    EXPECT_EQ(conv->get_padding_below(), CoordinateDiff{2});
    EXPECT_EQ(conv->get_padding_above(), CoordinateDiff{3});
}

TEST(type_prop, conv_1d_back_data_batch_deduce_strided_padded)
{
    // Deduce type
    Shape data_batch_shape{64, 3, 100};
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 10});  // filters
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{64, 128, 48}); // output delta
    auto move_strides = Strides{2};
    auto dilation_strides = Strides{1};
    auto padding_below = CoordinateDiff{2};
    auto padding_above = CoordinateDiff{3};
    auto conv = make_shared<op::ConvolutionBackpropData>(data_batch_shape,
                                                         param0,
                                                         param1,
                                                         move_strides,
                                                         dilation_strides,
                                                         padding_below,
                                                         padding_above,
                                                         Strides{1});
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), data_batch_shape);

    EXPECT_EQ(conv->get_window_movement_strides_forward(), Strides{2});
    EXPECT_EQ(conv->get_window_dilation_strides_forward(), Strides{1});
    EXPECT_EQ(conv->get_data_dilation_strides_forward(), Strides{1});

    EXPECT_EQ(conv->get_padding_below_forward(), CoordinateDiff{2});
    EXPECT_EQ(conv->get_padding_above_forward(), CoordinateDiff{3});
}

TEST(type_prop, conv_1d_back_filters_deduce_strided_padded)
{
    // Deduce type
    // Shape data_batch_shape{64, 3, 100};
    Shape filters_shape{128, 3, 10};
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{64, 3, 100});  // data batch
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{64, 128, 48}); // output delta
    auto move_strides = Strides{2};
    auto dilation_strides = Strides{1};
    auto padding_below = CoordinateDiff{2};
    auto padding_above = CoordinateDiff{3};
    auto conv = make_shared<op::ConvolutionBackpropFilters>(param0,
                                                            filters_shape,
                                                            param1,
                                                            move_strides,
                                                            dilation_strides,
                                                            padding_below,
                                                            padding_above,
                                                            Strides{1});
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), filters_shape);

    EXPECT_EQ(conv->get_window_movement_strides_forward(), Strides{2});
    EXPECT_EQ(conv->get_window_dilation_strides_forward(), Strides{1});
    EXPECT_EQ(conv->get_data_dilation_strides_forward(), Strides{1});

    EXPECT_EQ(conv->get_padding_below_forward(), CoordinateDiff{2});
    EXPECT_EQ(conv->get_padding_above_forward(), CoordinateDiff{3});
}

TEST(type_prop, conv_1d_deduce_strided_small_uneven)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{64, 3, 5});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 2});
    auto move_strides = Strides{2};
    auto conv = make_shared<op::Convolution>(param0, param1, move_strides);
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), (Shape{64, 128, 2}));

    EXPECT_EQ(conv->get_window_movement_strides(), Strides{2});
    EXPECT_EQ(conv->get_window_dilation_strides(), Strides{1});
    EXPECT_EQ(conv->get_data_dilation_strides(), Strides{1});

    EXPECT_EQ(conv->get_padding_below(), CoordinateDiff{0});
    EXPECT_EQ(conv->get_padding_above(), CoordinateDiff{0});
}

TEST(type_prop, conv_1d_back_data_batch_deduce_strided_small_uneven)
{
    // Deduce type
    Shape data_batch_shape{64, 3, 5};
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 2});  // filters
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{64, 128, 2}); // output delta
    auto move_strides = Strides{2};
    auto conv = make_shared<op::ConvolutionBackpropData>(data_batch_shape,
                                                         param0,
                                                         param1,
                                                         move_strides,
                                                         Strides{1},
                                                         CoordinateDiff{0},
                                                         CoordinateDiff{0},
                                                         Strides{1});
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), data_batch_shape);

    EXPECT_EQ(conv->get_window_movement_strides_forward(), Strides{2});
    EXPECT_EQ(conv->get_window_dilation_strides_forward(), Strides{1});
    EXPECT_EQ(conv->get_data_dilation_strides_forward(), Strides{1});

    EXPECT_EQ(conv->get_padding_below_forward(), CoordinateDiff{0});
    EXPECT_EQ(conv->get_padding_above_forward(), CoordinateDiff{0});
}

TEST(type_prop, conv_1d_back_filters_deduce_strided_small_uneven)
{
    // Deduce type
    // Shape data_batch_shape{64, 3, 5};
    Shape filters_shape{128, 3, 2};
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{64, 3, 5});   // data batch
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{64, 128, 2}); // output delta
    auto move_strides = Strides{2};
    auto conv = make_shared<op::ConvolutionBackpropFilters>(param0,
                                                            filters_shape,
                                                            param1,
                                                            move_strides,
                                                            Strides{1},
                                                            CoordinateDiff{0},
                                                            CoordinateDiff{0},
                                                            Strides{1});
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), filters_shape);

    EXPECT_EQ(conv->get_window_movement_strides_forward(), Strides{2});
    EXPECT_EQ(conv->get_window_dilation_strides_forward(), Strides{1});
    EXPECT_EQ(conv->get_data_dilation_strides_forward(), Strides{1});

    EXPECT_EQ(conv->get_padding_below_forward(), CoordinateDiff{0});
    EXPECT_EQ(conv->get_padding_above_forward(), CoordinateDiff{0});
}

TEST(type_prop, conv_1d_deduce_strided_small_even)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{64, 3, 6});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 2});
    auto move_strides = Strides{2};
    auto conv = make_shared<op::Convolution>(param0, param1, move_strides);
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), (Shape{64, 128, 3}));

    EXPECT_EQ(conv->get_window_movement_strides(), Strides{2});
    EXPECT_EQ(conv->get_window_dilation_strides(), Strides{1});
    EXPECT_EQ(conv->get_data_dilation_strides(), Strides{1});

    EXPECT_EQ(conv->get_padding_below(), CoordinateDiff{0});
    EXPECT_EQ(conv->get_padding_above(), CoordinateDiff{0});
}

TEST(type_prop, conv_1d_back_data_batch_deduce_strided_small_even)
{
    // Deduce type
    Shape data_batch_shape{64, 3, 6};
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 2});  // filters
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{64, 128, 3}); // output delta
    auto move_strides = Strides{2};
    auto conv = make_shared<op::ConvolutionBackpropData>(data_batch_shape,
                                                         param0,
                                                         param1,
                                                         move_strides,
                                                         Strides{1},
                                                         CoordinateDiff{0},
                                                         CoordinateDiff{0},
                                                         Strides{1});
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), data_batch_shape);

    EXPECT_EQ(conv->get_window_movement_strides_forward(), Strides{2});
    EXPECT_EQ(conv->get_window_dilation_strides_forward(), Strides{1});
    EXPECT_EQ(conv->get_data_dilation_strides_forward(), Strides{1});

    EXPECT_EQ(conv->get_padding_below_forward(), CoordinateDiff{0});
    EXPECT_EQ(conv->get_padding_above_forward(), CoordinateDiff{0});
}

TEST(type_prop, conv_1d_back_filters_deduce_strided_small_even)
{
    // Deduce type
    // Shape data_batch_shape{64, 3, 6};
    Shape filters_shape{128, 3, 2};
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{64, 3, 6});   // data batch
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{64, 128, 3}); // output delta
    auto move_strides = Strides{2};
    auto conv = make_shared<op::ConvolutionBackpropFilters>(param0,
                                                            filters_shape,
                                                            param1,
                                                            move_strides,
                                                            Strides{1},
                                                            CoordinateDiff{0},
                                                            CoordinateDiff{0},
                                                            Strides{1});
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), filters_shape);

    EXPECT_EQ(conv->get_window_movement_strides_forward(), Strides{2});
    EXPECT_EQ(conv->get_window_dilation_strides_forward(), Strides{1});
    EXPECT_EQ(conv->get_data_dilation_strides_forward(), Strides{1});

    EXPECT_EQ(conv->get_padding_below_forward(), CoordinateDiff{0});
    EXPECT_EQ(conv->get_padding_above_forward(), CoordinateDiff{0});
}

TEST(type_prop, conv_1d_deduce_window_dilated)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{64, 3, 100});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 10});
    auto move_strides = Strides{1};
    auto dilate_strides = Strides{2};
    auto conv = make_shared<op::Convolution>(param0, param1, move_strides, dilate_strides);
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), (Shape{64, 128, 82}));

    EXPECT_EQ(conv->get_window_movement_strides(), Strides{1});
    EXPECT_EQ(conv->get_window_dilation_strides(), Strides{2});
    EXPECT_EQ(conv->get_data_dilation_strides(), Strides{1});

    EXPECT_EQ(conv->get_padding_below(), CoordinateDiff{0});
    EXPECT_EQ(conv->get_padding_above(), CoordinateDiff{0});
}

TEST(type_prop, conv_1d_back_data_batch_deduce_window_dilated)
{
    // Deduce type
    Shape data_batch_shape{64, 3, 100};
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 10});  // filters
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{64, 128, 82}); // output delta
    auto move_strides = Strides{1};
    auto dilate_strides = Strides{2};
    auto conv = make_shared<op::ConvolutionBackpropData>(data_batch_shape,
                                                         param0,
                                                         param1,
                                                         move_strides,
                                                         dilate_strides,
                                                         CoordinateDiff{0},
                                                         CoordinateDiff{0},
                                                         Strides{1});
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), data_batch_shape);

    EXPECT_EQ(conv->get_window_movement_strides_forward(), Strides{1});
    EXPECT_EQ(conv->get_window_dilation_strides_forward(), Strides{2});
    EXPECT_EQ(conv->get_data_dilation_strides_forward(), Strides{1});

    EXPECT_EQ(conv->get_padding_below_forward(), CoordinateDiff{0});
    EXPECT_EQ(conv->get_padding_above_forward(), CoordinateDiff{0});
}

TEST(type_prop, conv_1d_back_filters_deduce_window_dilated)
{
    // Deduce type
    // Shape data_batch_shape{64, 3, 100};
    Shape filters_shape{128, 3, 10};
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{64, 3, 100});  // data batch
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{64, 128, 82}); // output delta
    auto move_strides = Strides{1};
    auto dilate_strides = Strides{2};
    auto conv = make_shared<op::ConvolutionBackpropFilters>(param0,
                                                            filters_shape,
                                                            param1,
                                                            move_strides,
                                                            dilate_strides,
                                                            CoordinateDiff{0},
                                                            CoordinateDiff{0},
                                                            Strides{1});
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), filters_shape);

    EXPECT_EQ(conv->get_window_movement_strides_forward(), Strides{1});
    EXPECT_EQ(conv->get_window_dilation_strides_forward(), Strides{2});
    EXPECT_EQ(conv->get_data_dilation_strides_forward(), Strides{1});

    EXPECT_EQ(conv->get_padding_below_forward(), CoordinateDiff{0});
    EXPECT_EQ(conv->get_padding_above_forward(), CoordinateDiff{0});
}

TEST(type_prop, conv_1d_deduce_window_dilated_padded)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{64, 3, 100});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 10});
    auto move_strides = Strides{1};
    auto dilate_strides = Strides{2};
    auto padding_below = CoordinateDiff{2};
    auto padding_above = CoordinateDiff{3};
    auto conv = make_shared<op::Convolution>(
        param0, param1, move_strides, dilate_strides, padding_below, padding_above);
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), (Shape{64, 128, 87}));

    EXPECT_EQ(conv->get_window_movement_strides(), Strides{1});
    EXPECT_EQ(conv->get_window_dilation_strides(), Strides{2});
    EXPECT_EQ(conv->get_data_dilation_strides(), Strides{1});

    EXPECT_EQ(conv->get_padding_below(), CoordinateDiff{2});
    EXPECT_EQ(conv->get_padding_above(), CoordinateDiff{3});
}

TEST(type_prop, conv_1d_back_data_batch_deduce_window_dilated_padded)
{
    // Deduce type
    Shape data_batch_shape{64, 3, 100};
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 10});  // filters
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{64, 128, 87}); // output delta
    auto move_strides = Strides{1};
    auto dilate_strides = Strides{2};
    auto padding_below = CoordinateDiff{2};
    auto padding_above = CoordinateDiff{3};
    auto conv = make_shared<op::ConvolutionBackpropData>(data_batch_shape,
                                                         param0,
                                                         param1,
                                                         move_strides,
                                                         dilate_strides,
                                                         padding_below,
                                                         padding_above,
                                                         Strides{1});
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), data_batch_shape);

    EXPECT_EQ(conv->get_window_movement_strides_forward(), Strides{1});
    EXPECT_EQ(conv->get_window_dilation_strides_forward(), Strides{2});
    EXPECT_EQ(conv->get_data_dilation_strides_forward(), Strides{1});

    EXPECT_EQ(conv->get_padding_below_forward(), CoordinateDiff{2});
    EXPECT_EQ(conv->get_padding_above_forward(), CoordinateDiff{3});
}

TEST(type_prop, conv_1d_back_filters_deduce_window_dilated_padded)
{
    // Deduce type
    // Shape data_batch_shape{64, 3, 100};
    Shape filters_shape{128, 3, 10};
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{64, 3, 100});  // data batch
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{64, 128, 87}); // output delta
    auto move_strides = Strides{1};
    auto dilate_strides = Strides{2};
    auto padding_below = CoordinateDiff{2};
    auto padding_above = CoordinateDiff{3};
    auto conv = make_shared<op::ConvolutionBackpropFilters>(param0,
                                                            filters_shape,
                                                            param1,
                                                            move_strides,
                                                            dilate_strides,
                                                            padding_below,
                                                            padding_above,
                                                            Strides{1});
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), filters_shape);

    EXPECT_EQ(conv->get_window_movement_strides_forward(), Strides{1});
    EXPECT_EQ(conv->get_window_dilation_strides_forward(), Strides{2});
    EXPECT_EQ(conv->get_data_dilation_strides_forward(), Strides{1});

    EXPECT_EQ(conv->get_padding_below_forward(), CoordinateDiff{2});
    EXPECT_EQ(conv->get_padding_above_forward(), CoordinateDiff{3});
}

TEST(type_prop, conv_1d_deduce_window_dilated_data_dilated_padded)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{64, 3, 100});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 10});
    auto move_strides = Strides{1};
    auto dilate_strides = Strides{2};
    auto padding_below = CoordinateDiff{2};
    auto padding_above = CoordinateDiff{3};
    auto data_dilate_strides = Strides{3};
    auto conv = make_shared<op::Convolution>(param0,
                                             param1,
                                             move_strides,
                                             dilate_strides,
                                             padding_below,
                                             padding_above,
                                             data_dilate_strides);
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), (Shape{64, 128, 285}));

    EXPECT_EQ(conv->get_window_movement_strides(), Strides{1});
    EXPECT_EQ(conv->get_window_dilation_strides(), Strides{2});
    EXPECT_EQ(conv->get_data_dilation_strides(), Strides{3});

    EXPECT_EQ(conv->get_padding_below(), CoordinateDiff{2});
    EXPECT_EQ(conv->get_padding_above(), CoordinateDiff{3});
}

TEST(type_prop, conv_1d_back_data_batch_deduce_window_dilated_data_dilated_padded)
{
    // Deduce type
    Shape data_batch_shape{64, 3, 100};
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 10});   // filters
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{64, 128, 285}); // output delta
    auto move_strides = Strides{1};
    auto dilate_strides = Strides{2};
    auto padding_below = CoordinateDiff{2};
    auto padding_above = CoordinateDiff{3};
    auto data_dilate_strides = Strides{3};
    auto conv = make_shared<op::ConvolutionBackpropData>(data_batch_shape,
                                                         param0,
                                                         param1,
                                                         move_strides,
                                                         dilate_strides,
                                                         padding_below,
                                                         padding_above,
                                                         data_dilate_strides);
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), data_batch_shape);

    EXPECT_EQ(conv->get_window_movement_strides_forward(), Strides{1});
    EXPECT_EQ(conv->get_window_dilation_strides_forward(), Strides{2});
    EXPECT_EQ(conv->get_data_dilation_strides_forward(), Strides{3});

    EXPECT_EQ(conv->get_padding_below_forward(), CoordinateDiff{2});
    EXPECT_EQ(conv->get_padding_above_forward(), CoordinateDiff{3});
}

TEST(type_prop, conv_1d_back_filters_deduce_window_dilated_data_dilated_padded)
{
    // Deduce type
    // Shape data_batch_shape{64, 3, 100};
    Shape filters_shape{128, 3, 10};
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{64, 3, 100});   // data batch
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{64, 128, 285}); // output delta
    auto move_strides = Strides{1};
    auto dilate_strides = Strides{2};
    auto padding_below = CoordinateDiff{2};
    auto padding_above = CoordinateDiff{3};
    auto data_dilate_strides = Strides{3};
    auto conv = make_shared<op::ConvolutionBackpropFilters>(param0,
                                                            filters_shape,
                                                            param1,
                                                            move_strides,
                                                            dilate_strides,
                                                            padding_below,
                                                            padding_above,
                                                            data_dilate_strides);
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), filters_shape);

    EXPECT_EQ(conv->get_window_movement_strides_forward(), Strides{1});
    EXPECT_EQ(conv->get_window_dilation_strides_forward(), Strides{2});
    EXPECT_EQ(conv->get_data_dilation_strides_forward(), Strides{3});

    EXPECT_EQ(conv->get_padding_below_forward(), CoordinateDiff{2});
    EXPECT_EQ(conv->get_padding_above_forward(), CoordinateDiff{3});
}

TEST(type_prop, conv_2d_deduce)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{64, 3, 100, 150});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 10, 20});
    auto conv = make_shared<op::Convolution>(param0, param1);
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), (Shape{64, 128, 91, 131}));

    EXPECT_EQ(conv->get_window_movement_strides(), (Strides{1, 1}));
    EXPECT_EQ(conv->get_window_dilation_strides(), (Strides{1, 1}));
    EXPECT_EQ(conv->get_data_dilation_strides(), (Strides{1, 1}));

    EXPECT_EQ(conv->get_padding_below(), (CoordinateDiff{0, 0}));
    EXPECT_EQ(conv->get_padding_above(), (CoordinateDiff{0, 0}));
}

TEST(type_prop, conv_2d_deduce_padded)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{64, 3, 100, 150});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 10, 20});
    auto move_strides = Strides{1, 1};
    auto dilate_strides = Strides{1, 1};
    auto padding_below = CoordinateDiff{2, 3};
    auto padding_above = CoordinateDiff{3, 4};
    auto conv = make_shared<op::Convolution>(
        param0, param1, move_strides, dilate_strides, padding_below, padding_above);
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), (Shape{64, 128, 96, 138}));

    EXPECT_EQ(conv->get_window_movement_strides(), (Strides{1, 1}));
    EXPECT_EQ(conv->get_window_dilation_strides(), (Strides{1, 1}));
    EXPECT_EQ(conv->get_data_dilation_strides(), (Strides{1, 1}));

    EXPECT_EQ(conv->get_padding_below(), (CoordinateDiff{2, 3}));
    EXPECT_EQ(conv->get_padding_above(), (CoordinateDiff{3, 4}));
}

TEST(type_prop, conv_2d_deduce_padded_neg)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{64, 3, 100, 150});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 10, 20});
    auto move_strides = Strides{1, 1};
    auto dilate_strides = Strides{1, 1};
    auto padding_below = CoordinateDiff{2, -3};
    auto padding_above = CoordinateDiff{3, -4};
    auto conv = make_shared<op::Convolution>(
        param0, param1, move_strides, dilate_strides, padding_below, padding_above);
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), (Shape{64, 128, 96, 124}));

    EXPECT_EQ(conv->get_window_movement_strides(), (Strides{1, 1}));
    EXPECT_EQ(conv->get_window_dilation_strides(), (Strides{1, 1}));
    EXPECT_EQ(conv->get_data_dilation_strides(), (Strides{1, 1}));

    EXPECT_EQ(conv->get_padding_below(), (CoordinateDiff{2, -3}));
    EXPECT_EQ(conv->get_padding_above(), (CoordinateDiff{3, -4}));
}

TEST(type_prop, conv_2d_deduce_strided)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{64, 3, 100, 150});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 10, 20});
    auto move_strides = Strides{2, 3};
    auto conv = make_shared<op::Convolution>(param0, param1, move_strides);
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), (Shape{64, 128, 46, 44}));

    EXPECT_EQ(conv->get_window_movement_strides(), (Strides{2, 3}));
    EXPECT_EQ(conv->get_window_dilation_strides(), (Strides{1, 1}));
    EXPECT_EQ(conv->get_data_dilation_strides(), (Strides{1, 1}));

    EXPECT_EQ(conv->get_padding_below(), (CoordinateDiff{0, 0}));
    EXPECT_EQ(conv->get_padding_above(), (CoordinateDiff{0, 0}));
}

TEST(type_prop, conv_2d_deduce_strided_window_dilated)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{64, 3, 100, 150});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 10, 20});
    auto move_strides = Strides{2, 3};
    auto dilate_strides = Strides{3, 2};
    auto conv = make_shared<op::Convolution>(param0, param1, move_strides, dilate_strides);
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), (Shape{64, 128, 37, 38}));

    EXPECT_EQ(conv->get_window_movement_strides(), (Strides{2, 3}));
    EXPECT_EQ(conv->get_window_dilation_strides(), (Strides{3, 2}));
    EXPECT_EQ(conv->get_data_dilation_strides(), (Strides{1, 1}));

    EXPECT_EQ(conv->get_padding_below(), (CoordinateDiff{0, 0}));
    EXPECT_EQ(conv->get_padding_above(), (CoordinateDiff{0, 0}));
}

TEST(type_prop, conv_2d_deduce_strided_window_dilated_data_dilated)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{64, 3, 100, 150});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 10, 20});
    auto move_strides = Strides{2, 3};
    auto dilate_strides = Strides{3, 2};
    auto padding_below = CoordinateDiff{0, 0};
    auto padding_above = CoordinateDiff{0, 0};
    auto data_dilate_strides = Strides{2, 3};
    auto conv = make_shared<op::Convolution>(param0,
                                             param1,
                                             move_strides,
                                             dilate_strides,
                                             padding_below,
                                             padding_above,
                                             data_dilate_strides);
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), (Shape{64, 128, 86, 137}));

    EXPECT_EQ(conv->get_window_movement_strides(), (Strides{2, 3}));
    EXPECT_EQ(conv->get_window_dilation_strides(), (Strides{3, 2}));
    EXPECT_EQ(conv->get_data_dilation_strides(), (Strides{2, 3}));

    EXPECT_EQ(conv->get_padding_below(), (CoordinateDiff{0, 0}));
    EXPECT_EQ(conv->get_padding_above(), (CoordinateDiff{0, 0}));
}

TEST(type_prop, conv_2d_deduce_strided_window_dilated_small)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{64, 3, 7, 8});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 2, 3});
    auto move_strides = Strides{2, 3};
    auto dilate_strides = Strides{3, 2};
    auto conv = make_shared<op::Convolution>(param0, param1, move_strides, dilate_strides);
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), (Shape{64, 128, 2, 2}));

    EXPECT_EQ(conv->get_window_movement_strides(), (Strides{2, 3}));
    EXPECT_EQ(conv->get_window_dilation_strides(), (Strides{3, 2}));
    EXPECT_EQ(conv->get_data_dilation_strides(), (Strides{1, 1}));

    EXPECT_EQ(conv->get_padding_below(), (CoordinateDiff{0, 0}));
    EXPECT_EQ(conv->get_padding_above(), (CoordinateDiff{0, 0}));
}

TEST(type_prop, conv_3d_deduce_strided_window_dilated_small)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{64, 3, 7, 8, 10});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 2, 3, 2});
    auto move_strides = Strides{2, 3, 4};
    auto dilate_strides = Strides{3, 2, 2};
    auto conv = make_shared<op::Convolution>(param0, param1, move_strides, dilate_strides);
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), (Shape{64, 128, 2, 2, 2}));

    EXPECT_EQ(conv->get_window_movement_strides(), (Strides{2, 3, 4}));
    EXPECT_EQ(conv->get_window_dilation_strides(), (Strides{3, 2, 2}));
    EXPECT_EQ(conv->get_data_dilation_strides(), (Strides{1, 1, 1}));

    EXPECT_EQ(conv->get_padding_below(), (CoordinateDiff{0, 0, 0}));
    EXPECT_EQ(conv->get_padding_above(), (CoordinateDiff{0, 0, 0}));
}

TEST(type_prop, conv_3d_deduce_strided_window_dilated_data_dilated_small)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{64, 3, 7, 8, 10});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 2, 3, 2});
    auto move_strides = Strides{2, 3, 4};
    auto dilate_strides = Strides{3, 2, 2};
    auto padding_below = CoordinateDiff{0, 0, 0};
    auto padding_above = CoordinateDiff{0, 0, 0};
    auto data_dilate_strides = Strides{2, 3, 2};
    auto conv = make_shared<op::Convolution>(param0,
                                             param1,
                                             move_strides,
                                             dilate_strides,
                                             padding_below,
                                             padding_above,
                                             data_dilate_strides);
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), (Shape{64, 128, 5, 6, 5}));

    EXPECT_EQ(conv->get_window_movement_strides(), (Strides{2, 3, 4}));
    EXPECT_EQ(conv->get_window_dilation_strides(), (Strides{3, 2, 2}));
    EXPECT_EQ(conv->get_data_dilation_strides(), (Strides{2, 3, 2}));

    EXPECT_EQ(conv->get_padding_below(), (CoordinateDiff{0, 0, 0}));
    EXPECT_EQ(conv->get_padding_above(), (CoordinateDiff{0, 0, 0}));
}

TEST(type_prop, conv_invalid_element_type_mismatch)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{3, 3, 3, 3});
    auto param1 = make_shared<op::Parameter>(element::i32, Shape{3, 3, 2, 2});
    try
    {
        auto conv = make_shared<op::Convolution>(param0, param1);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with element type mismatch not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Element types for data batch and filters do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_invalid_0d_input)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{});
    try
    {
        auto conv = make_shared<op::Convolution>(param0, param1);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid 0D input not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Data batch and filters must have rank of at least 3 "
                                         "(one batch axis, one input-channel axis, "
                                         "and at least one spatial dimension)"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_invalid_1d_input)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{2});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{2});
    try
    {
        auto conv = make_shared<op::Convolution>(param0, param1);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid 1D input not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Data batch and filters must have rank of at least 3 "
                                         "(one batch axis, one input-channel axis, "
                                         "and at least one spatial dimension)"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_invalid_2d_input)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{2, 6});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{2, 6});
    try
    {
        auto conv = make_shared<op::Convolution>(param0, param1);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid 2D input not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Data batch and filters must have rank of at least 3 "
                                         "(one batch axis, one input-channel axis, "
                                         "and at least one spatial dimension)"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_invalid_0_batch_size)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{0, 6, 1});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{0, 6, 1});
    try
    {
        auto conv = make_shared<op::Convolution>(param0, param1);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with 0 batch size not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Batch size is zero"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_invalid_0_input_channels)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6, 0, 1});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{5, 0, 1});
    try
    {
        auto conv = make_shared<op::Convolution>(param0, param1);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with 0 input channels not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Data batch channel count and/or filter input channel count is zero"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_invalid_wrong_number_of_filter_dimensions_too_many)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6, 2, 10, 10});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{5, 2, 3, 3, 3});
    try
    {
        auto conv = make_shared<op::Convolution>(param0, param1);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with too many filter dimensions not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Data batch and filters rank do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_invalid_wrong_number_of_filter_dimensions_too_few)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6, 2, 10, 10});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{5, 2, 3});
    try
    {
        auto conv = make_shared<op::Convolution>(param0, param1);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with too few filter dimensions not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Data batch and filters rank do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_invalid_0_output_channels)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6, 2, 10, 10});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{0, 2, 3, 3});
    try
    {
        auto conv = make_shared<op::Convolution>(param0, param1);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with 0 output channels not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Filter output channel count is zero"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_invalid_input_channel_mismatch)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6, 2, 10, 10});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{6, 3, 3, 3});
    try
    {
        auto conv = make_shared<op::Convolution>(param0, param1);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with channel count mismatch not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string(
                "Data batch channel count (2) does not match filter input channel count (3)"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_invalid_movement_stride_rank)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6, 2, 10, 10});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{6, 2, 3, 3});
    try
    {
        auto conv = make_shared<op::Convolution>(param0, param1, Strides{2, 3, 8});

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with wrong movement stride rank not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Ranks for data item shape/filters shape (data batch has shape "
                        "{6,2,10,10}, so data item rank is 2 and filters have shape {6,2,3,3}, so "
                        "filters spatial rank is 2), data dilation (Strides{1, 1}), padding below "
                        "(CoordinateDiff{0, 0}), padding above (CoordinateDiff{0, 0}), filter "
                        "strides (Strides{2, 3, 8}), and filter dilation (Strides{1, 1}) do not "
                        "match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_invalid_window_dilation_stride_rank)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6, 2, 10, 10});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{6, 2, 3, 3});
    try
    {
        auto conv = make_shared<op::Convolution>(param0, param1, Strides{2, 3}, Strides{2, 3, 8});

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with wrong window dilation stride rank not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Ranks for data item shape/filters shape (data batch has shape "
                        "{6,2,10,10}, so data item rank is 2 and filters have shape {6,2,3,3}, so "
                        "filters spatial rank is 2), data dilation (Strides{1, 1}), padding below "
                        "(CoordinateDiff{0, 0}), padding above (CoordinateDiff{0, 0}), filter "
                        "strides (Strides{2, 3}), and filter dilation (Strides{2, 3, 8}) do not "
                        "match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_invalid_data_dilation_stride_rank)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6, 2, 10, 10});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{6, 2, 3, 3});
    try
    {
        auto conv = make_shared<op::Convolution>(param0,
                                                 param1,
                                                 Strides{2, 3},
                                                 Strides{2, 3},
                                                 CoordinateDiff{0, 0},
                                                 CoordinateDiff{0, 0},
                                                 Strides{2, 3, 8});

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with wrong data dilation stride rank not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Ranks for data item shape/filters shape (data batch has shape "
                        "{6,2,10,10}, so data item rank is 2 and filters have shape {6,2,3,3}, so "
                        "filters spatial rank is 2), data dilation (Strides{2, 3, 8}), padding "
                        "below (CoordinateDiff{0, 0}), padding above (CoordinateDiff{0, 0}), "
                        "filter strides (Strides{2, 3}), and filter dilation (Strides{2, 3}) do "
                        "not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_invalid_padding_below_rank)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6, 2, 10, 10});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{6, 2, 3, 3});
    try
    {
        auto conv = make_shared<op::Convolution>(param0,
                                                 param1,
                                                 Strides{2, 3},
                                                 Strides{1, 1},
                                                 CoordinateDiff{0, 0, 0},
                                                 CoordinateDiff{0, 0});

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with wrong padding-below rank not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string(
                "Ranks for data item shape/filters shape (data batch has shape "
                "{6,2,10,10}, so data item rank is 2 and filters have shape {6,2,3,3}, so "
                "filters spatial rank is 2), data dilation (Strides{1, 1}), padding below "
                "(CoordinateDiff{0, 0, 0}), padding above (CoordinateDiff{0, 0}), filter "
                "strides (Strides{2, 3}), and filter dilation (Strides{1, 1}) do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_invalid_padding_above_rank)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6, 2, 10, 10});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{6, 2, 3, 3});
    try
    {
        auto conv = make_shared<op::Convolution>(param0,
                                                 param1,
                                                 Strides{2, 3},
                                                 Strides{2, 3},
                                                 CoordinateDiff{0, 0},
                                                 CoordinateDiff{0, 0, 0});

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with wrong padding-above rank not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string(
                "Ranks for data item shape/filters shape (data batch has shape "
                "{6,2,10,10}, so data item rank is 2 and filters have shape {6,2,3,3}, so "
                "filters spatial rank is 2), data dilation (Strides{1, 1}), padding below "
                "(CoordinateDiff{0, 0}), padding above (CoordinateDiff{0, 0, 0}), filter "
                "strides (Strides{2, 3}), and filter dilation (Strides{2, 3}) do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_invalid_input_spatial_size_negative_after_padding)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6, 2, 10, 10});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{6, 2, 3, 3});
    try
    {
        auto conv = make_shared<op::Convolution>(param0,
                                                 param1,
                                                 Strides{1, 1},
                                                 Strides{1, 1},
                                                 CoordinateDiff{-4, 0},
                                                 CoordinateDiff{-7, 0});

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with negative-length post-padding spatial axis not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Data shape after padding and dilation has dimension less "
                                         "than 1 (dim: -1) at axis 0"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_invalid_input_spatial_size_zero_after_padding)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6, 2, 10, 10});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{6, 2, 3, 3});
    try
    {
        auto conv = make_shared<op::Convolution>(param0,
                                                 param1,
                                                 Strides{1, 1},
                                                 Strides{1, 1},
                                                 CoordinateDiff{-4, 0},
                                                 CoordinateDiff{-6, 0});

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with zero-length post-padding spatial axis not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Data shape after padding and dilation has dimension less "
                                         "than 1 (dim: 0) at axis 0"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_invalid_input_spatial_size_0)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6, 2, 0, 10});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{6, 2, 3, 3});
    try
    {
        auto conv = make_shared<op::Convolution>(param0, param1);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with zero-length spatial axis not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Data shape after padding and dilation has "
                                         "dimension less than 1 (dim: 0) at axis 0"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_invalid_window_size_0)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6, 2, 10, 10});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{6, 2, 3, 0});
    try
    {
        auto conv = make_shared<op::Convolution>(param0, param1);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with zero-length window axis not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Window after dilation has dimension less than 1 (dim: 0) at axis 1"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_invalid_window_dilation_stride_0)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6, 2, 10, 10});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{6, 2, 3, 3});
    try
    {
        auto conv = make_shared<op::Convolution>(param0, param1, Strides{2, 3}, Strides{2, 0});

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with wrong 0-length window dilation stride axis not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Window dilation (Strides{2, 0}) has zero dimension at axis 1"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_invalid_data_dilation_stride_0)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6, 2, 10, 10});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{6, 2, 3, 3});
    try
    {
        auto conv = make_shared<op::Convolution>(param0,
                                                 param1,
                                                 Strides{2, 3},
                                                 Strides{2, 3},
                                                 CoordinateDiff{0, 0},
                                                 CoordinateDiff{0, 0},
                                                 Strides{2, 0});

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with wrong 0-length data dilation stride axis not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Data dilation (Strides{2, 0}) has zero dimension at axis 1"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_invalid_dilated_window_too_large)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6, 2, 8, 8});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{6, 2, 3, 3});
    try
    {
        auto conv = make_shared<op::Convolution>(param0, param1, Strides{1, 1}, Strides{4, 4});

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with oversized dilated window not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Window after dilation has dimension (dim: 9) larger than "
                                         "the data shape after padding (dim: 8) at axis 0"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_invalid_movement_stride_0)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6, 2, 10, 10});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{6, 2, 3, 3});
    try
    {
        auto conv = make_shared<op::Convolution>(param0, param1, Strides{0, 1});

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with wrong 0-length movement stride axis not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Window strides (Strides{0, 1}) has zero dimension at axis 0"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_partial_rank_dynamic_rank_dynamic_ok)
{
    PartialShape data_batch_shape{PartialShape::dynamic()};
    PartialShape filters_shape{PartialShape::dynamic()};
    Strides window_movement_strides{1, 1};
    Strides window_dilation_strides{1, 1};
    CoordinateDiff padding_below{0, 0};
    CoordinateDiff padding_above{0, 0};
    Strides data_dilation_strides{1, 1};

    auto param0 = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto param1 = make_shared<op::Parameter>(element::f32, filters_shape);

    auto conv = make_shared<op::Convolution>(param0,
                                             param1,
                                             window_movement_strides,
                                             window_dilation_strides,
                                             padding_below,
                                             padding_above,
                                             data_dilation_strides);

    ASSERT_EQ(conv->get_output_element_type(0), element::f32);
    ASSERT_TRUE(conv->get_output_partial_shape(0).same_scheme(PartialShape::dynamic(4)));
}

TEST(type_prop, conv_partial_rank_dynamic_rank_dynamic_window_strides_rank_wrong)
{
    PartialShape data_batch_shape{PartialShape::dynamic()};
    PartialShape filters_shape{PartialShape::dynamic()};
    Strides window_movement_strides{1, 1, 1};
    Strides window_dilation_strides{1, 1};
    CoordinateDiff padding_below{0, 0};
    CoordinateDiff padding_above{0, 0};
    Strides data_dilation_strides{1, 1};

    auto param0 = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto param1 = make_shared<op::Parameter>(element::f32, filters_shape);

    try
    {
        auto conv = make_shared<op::Convolution>(param0,
                                                 param1,
                                                 window_movement_strides,
                                                 window_dilation_strides,
                                                 padding_below,
                                                 padding_above,
                                                 data_dilation_strides);

        FAIL() << "Window stride rank mismatch not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Ranks for data item shape/filters shape (data batch has shape ?, so data "
                        "item rank is ? and filters have shape ?, so filters spatial rank is ?), "
                        "data dilation (Strides{1, 1}), padding below (CoordinateDiff{0, 0}), "
                        "padding above (CoordinateDiff{0, 0}), filter strides (Strides{1, 1, 1}), "
                        "and filter dilation (Strides{1, 1}) do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_partial_rank_dynamic_rank_dynamic_window_strides_dim_zero)
{
    PartialShape data_batch_shape{PartialShape::dynamic()};
    PartialShape filters_shape{PartialShape::dynamic()};
    Strides window_movement_strides{1, 0};
    Strides window_dilation_strides{1, 1};
    CoordinateDiff padding_below{0, 0};
    CoordinateDiff padding_above{0, 0};
    Strides data_dilation_strides{1, 1};

    auto param0 = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto param1 = make_shared<op::Parameter>(element::f32, filters_shape);

    try
    {
        auto conv = make_shared<op::Convolution>(param0,
                                                 param1,
                                                 window_movement_strides,
                                                 window_dilation_strides,
                                                 padding_below,
                                                 padding_above,
                                                 data_dilation_strides);

        FAIL() << "Window stride with dimension zero not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Window strides (Strides{1, 0}) has zero dimension at axis 1"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_partial_rank_dynamic_rank_dynamic_window_dilation_rank_wrong)
{
    PartialShape data_batch_shape{PartialShape::dynamic()};
    PartialShape filters_shape{PartialShape::dynamic()};
    Strides window_movement_strides{1, 1};
    Strides window_dilation_strides{1, 1, 1};
    CoordinateDiff padding_below{0, 0};
    CoordinateDiff padding_above{0, 0};
    Strides data_dilation_strides{1, 1};

    auto param0 = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto param1 = make_shared<op::Parameter>(element::f32, filters_shape);

    try
    {
        auto conv = make_shared<op::Convolution>(param0,
                                                 param1,
                                                 window_movement_strides,
                                                 window_dilation_strides,
                                                 padding_below,
                                                 padding_above,
                                                 data_dilation_strides);

        FAIL() << "Window dilation rank mismatch not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Ranks for data item shape/filters shape (data batch has shape ?, so data "
                        "item rank is ? and filters have shape ?, so filters spatial rank is ?), "
                        "data dilation (Strides{1, 1}), padding below (CoordinateDiff{0, 0}), "
                        "padding above (CoordinateDiff{0, 0}), filter strides (Strides{1, 1}), and "
                        "filter dilation (Strides{1, 1, 1}) do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_partial_rank_dynamic_rank_dynamic_window_dilation_dim_zero)
{
    PartialShape data_batch_shape{PartialShape::dynamic()};
    PartialShape filters_shape{PartialShape::dynamic()};
    Strides window_movement_strides{1, 1};
    Strides window_dilation_strides{1, 0};
    CoordinateDiff padding_below{0, 0};
    CoordinateDiff padding_above{0, 0};
    Strides data_dilation_strides{1, 1};

    auto param0 = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto param1 = make_shared<op::Parameter>(element::f32, filters_shape);

    try
    {
        auto conv = make_shared<op::Convolution>(param0,
                                                 param1,
                                                 window_movement_strides,
                                                 window_dilation_strides,
                                                 padding_below,
                                                 padding_above,
                                                 data_dilation_strides);

        FAIL() << "Window dilation with dimension zero not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Window dilation (Strides{1, 0}) has zero dimension at axis 1"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_partial_rank_dynamic_rank_dynamic_padding_below_rank_wrong)
{
    PartialShape data_batch_shape{PartialShape::dynamic()};
    PartialShape filters_shape{PartialShape::dynamic()};
    Strides window_movement_strides{1, 1};
    Strides window_dilation_strides{1, 1};
    CoordinateDiff padding_below{0, 0, 0};
    CoordinateDiff padding_above{0, 0};
    Strides data_dilation_strides{1, 1};

    auto param0 = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto param1 = make_shared<op::Parameter>(element::f32, filters_shape);

    try
    {
        auto conv = make_shared<op::Convolution>(param0,
                                                 param1,
                                                 window_movement_strides,
                                                 window_dilation_strides,
                                                 padding_below,
                                                 padding_above,
                                                 data_dilation_strides);

        FAIL() << "Padding below rank mismatch not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Ranks for data item shape/filters shape (data batch has shape ?, so data "
                        "item rank is ? and filters have shape ?, so filters spatial rank is ?), "
                        "data dilation (Strides{1, 1}), padding below (CoordinateDiff{0, 0, 0}), "
                        "padding above (CoordinateDiff{0, 0}), filter strides (Strides{1, 1}), and "
                        "filter dilation (Strides{1, 1}) do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_partial_rank_dynamic_rank_dynamic_padding_above_rank_wrong)
{
    PartialShape data_batch_shape{PartialShape::dynamic()};
    PartialShape filters_shape{PartialShape::dynamic()};
    Strides window_movement_strides{1, 1};
    Strides window_dilation_strides{1, 1};
    CoordinateDiff padding_below{0, 0};
    CoordinateDiff padding_above{0, 0, 0};
    Strides data_dilation_strides{1, 1};

    auto param0 = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto param1 = make_shared<op::Parameter>(element::f32, filters_shape);

    try
    {
        auto conv = make_shared<op::Convolution>(param0,
                                                 param1,
                                                 window_movement_strides,
                                                 window_dilation_strides,
                                                 padding_below,
                                                 padding_above,
                                                 data_dilation_strides);

        FAIL() << "Padding above rank mismatch not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Ranks for data item shape/filters shape (data batch has shape ?, so data "
                        "item rank is ? and filters have shape ?, so filters spatial rank is ?), "
                        "data dilation (Strides{1, 1}), padding below (CoordinateDiff{0, 0}), "
                        "padding above (CoordinateDiff{0, 0, 0}), filter strides (Strides{1, 1}), "
                        "and filter dilation (Strides{1, 1}) do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_partial_rank_dynamic_rank_dynamic_data_dilation_rank_wrong)
{
    PartialShape data_batch_shape{PartialShape::dynamic()};
    PartialShape filters_shape{PartialShape::dynamic()};
    Strides window_movement_strides{1, 1};
    Strides window_dilation_strides{1, 1};
    CoordinateDiff padding_below{0, 0};
    CoordinateDiff padding_above{0, 0};
    Strides data_dilation_strides{1, 1, 1};

    auto param0 = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto param1 = make_shared<op::Parameter>(element::f32, filters_shape);

    try
    {
        auto conv = make_shared<op::Convolution>(param0,
                                                 param1,
                                                 window_movement_strides,
                                                 window_dilation_strides,
                                                 padding_below,
                                                 padding_above,
                                                 data_dilation_strides);

        FAIL() << "Data dilation rank mismatch not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Ranks for data item shape/filters shape (data batch has shape ?, so data "
                        "item rank is ? and filters have shape ?, so filters spatial rank is ?), "
                        "data dilation (Strides{1, 1, 1}), padding below (CoordinateDiff{0, 0}), "
                        "padding above (CoordinateDiff{0, 0}), filter strides (Strides{1, 1}), and "
                        "filter dilation (Strides{1, 1}) do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_partial_rank_dynamic_rank_dynamic_data_dilation_dim_zero)
{
    PartialShape data_batch_shape{PartialShape::dynamic()};
    PartialShape filters_shape{PartialShape::dynamic()};
    Strides window_movement_strides{1, 1};
    Strides window_dilation_strides{1, 1};
    CoordinateDiff padding_below{0, 0};
    CoordinateDiff padding_above{0, 0};
    Strides data_dilation_strides{1, 0};

    auto param0 = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto param1 = make_shared<op::Parameter>(element::f32, filters_shape);

    try
    {
        auto conv = make_shared<op::Convolution>(param0,
                                                 param1,
                                                 window_movement_strides,
                                                 window_dilation_strides,
                                                 padding_below,
                                                 padding_above,
                                                 data_dilation_strides);

        FAIL() << "Data dilation with dimension zero not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Data dilation (Strides{1, 0}) has zero dimension at axis 1"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_partial_rank_static_dynamic_rank_dynamic_ok)
{
    PartialShape data_batch_shape{PartialShape::dynamic(4)};
    PartialShape filters_shape{PartialShape::dynamic()};
    Strides window_movement_strides{1, 1};
    Strides window_dilation_strides{1, 1};
    CoordinateDiff padding_below{0, 0};
    CoordinateDiff padding_above{0, 0};
    Strides data_dilation_strides{1, 1};

    auto param0 = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto param1 = make_shared<op::Parameter>(element::f32, filters_shape);

    auto conv = make_shared<op::Convolution>(param0,
                                             param1,
                                             window_movement_strides,
                                             window_dilation_strides,
                                             padding_below,
                                             padding_above,
                                             data_dilation_strides);

    ASSERT_EQ(conv->get_output_element_type(0), element::f32);
    ASSERT_TRUE(conv->get_output_partial_shape(0).same_scheme(PartialShape::dynamic(4)));
}

TEST(type_prop, conv_partial_rank_static_dynamic_rank_dynamic_data_batch_rank_wrong)
{
    PartialShape data_batch_shape{PartialShape::dynamic(5)};
    PartialShape filters_shape{PartialShape::dynamic()};
    Strides window_movement_strides{1, 1};
    Strides window_dilation_strides{1, 1};
    CoordinateDiff padding_below{0, 0};
    CoordinateDiff padding_above{0, 0};
    Strides data_dilation_strides{1, 1};

    auto param0 = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto param1 = make_shared<op::Parameter>(element::f32, filters_shape);

    try
    {
        auto conv = make_shared<op::Convolution>(param0,
                                                 param1,
                                                 window_movement_strides,
                                                 window_dilation_strides,
                                                 padding_below,
                                                 padding_above,
                                                 data_dilation_strides);

        FAIL() << "Data batch rank mismatch not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Ranks for data item shape/filters shape (data batch has shape "
                        "{?,?,?,?,?}, so data item rank is 3 and filters have shape ?, so filters "
                        "spatial rank is ?), data dilation (Strides{1, 1}), padding below "
                        "(CoordinateDiff{0, 0}), padding above (CoordinateDiff{0, 0}), filter "
                        "strides (Strides{1, 1}), and filter dilation (Strides{1, 1}) do not "
                        "match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_partial_rank_static_dynamic_rank_dynamic_batch_size_known_ok)
{
    PartialShape data_batch_shape{
        64, Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()};
    PartialShape filters_shape{PartialShape::dynamic()};
    Strides window_movement_strides{1, 1};
    Strides window_dilation_strides{1, 1};
    CoordinateDiff padding_below{0, 0};
    CoordinateDiff padding_above{0, 0};
    Strides data_dilation_strides{1, 1};

    auto param0 = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto param1 = make_shared<op::Parameter>(element::f32, filters_shape);

    auto conv = make_shared<op::Convolution>(param0,
                                             param1,
                                             window_movement_strides,
                                             window_dilation_strides,
                                             padding_below,
                                             padding_above,
                                             data_dilation_strides);

    ASSERT_EQ(conv->get_output_element_type(0), element::f32);
    ASSERT_TRUE(conv->get_output_partial_shape(0).same_scheme(
        PartialShape{64, Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()}));
}

TEST(type_prop, conv_partial_rank_static_dynamic_rank_dynamic_batch_size_known_zero)
{
    PartialShape data_batch_shape{
        0, Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()};
    PartialShape filters_shape{PartialShape::dynamic()};
    Strides window_movement_strides{1, 1};
    Strides window_dilation_strides{1, 1};
    CoordinateDiff padding_below{0, 0};
    CoordinateDiff padding_above{0, 0};
    Strides data_dilation_strides{1, 1};

    auto param0 = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto param1 = make_shared<op::Parameter>(element::f32, filters_shape);

    try
    {
        auto conv = make_shared<op::Convolution>(param0,
                                                 param1,
                                                 window_movement_strides,
                                                 window_dilation_strides,
                                                 padding_below,
                                                 padding_above,
                                                 data_dilation_strides);

        FAIL() << "Zero batch size not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Batch size is zero"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_partial_rank_static_dynamic_rank_dynamic_input_channel_count_known_ok)
{
    PartialShape data_batch_shape{
        Dimension::dynamic(), 3, Dimension::dynamic(), Dimension::dynamic()};
    PartialShape filters_shape{PartialShape::dynamic()};
    Strides window_movement_strides{1, 1};
    Strides window_dilation_strides{1, 1};
    CoordinateDiff padding_below{0, 0};
    CoordinateDiff padding_above{0, 0};
    Strides data_dilation_strides{1, 1};

    auto param0 = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto param1 = make_shared<op::Parameter>(element::f32, filters_shape);

    auto conv = make_shared<op::Convolution>(param0,
                                             param1,
                                             window_movement_strides,
                                             window_dilation_strides,
                                             padding_below,
                                             padding_above,
                                             data_dilation_strides);

    ASSERT_EQ(conv->get_output_element_type(0), element::f32);
    ASSERT_TRUE(conv->get_output_partial_shape(0).same_scheme(PartialShape::dynamic(4)));
}

TEST(type_prop, conv_partial_rank_static_dynamic_rank_dynamic_input_channel_count_known_zero)
{
    PartialShape data_batch_shape{
        Dimension::dynamic(), 0, Dimension::dynamic(), Dimension::dynamic()};
    PartialShape filters_shape{PartialShape::dynamic()};
    Strides window_movement_strides{1, 1};
    Strides window_dilation_strides{1, 1};
    CoordinateDiff padding_below{0, 0};
    CoordinateDiff padding_above{0, 0};
    Strides data_dilation_strides{1, 1};

    auto param0 = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto param1 = make_shared<op::Parameter>(element::f32, filters_shape);

    try
    {
        auto conv = make_shared<op::Convolution>(param0,
                                                 param1,
                                                 window_movement_strides,
                                                 window_dilation_strides,
                                                 padding_below,
                                                 padding_above,
                                                 data_dilation_strides);

        FAIL() << "Zero input channel count not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Data batch channel count and/or filter input channel count is zero"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_partial_rank_dynamic_rank_static_dynamic_output_channel_count_known_ok)
{
    PartialShape data_batch_shape{PartialShape::dynamic(4)};
    PartialShape filters_shape{
        32, Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()};
    Strides window_movement_strides{1, 1};
    Strides window_dilation_strides{1, 1};
    CoordinateDiff padding_below{0, 0};
    CoordinateDiff padding_above{0, 0};
    Strides data_dilation_strides{1, 1};

    auto param0 = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto param1 = make_shared<op::Parameter>(element::f32, filters_shape);

    auto conv = make_shared<op::Convolution>(param0,
                                             param1,
                                             window_movement_strides,
                                             window_dilation_strides,
                                             padding_below,
                                             padding_above,
                                             data_dilation_strides);

    ASSERT_EQ(conv->get_output_element_type(0), element::f32);
    ASSERT_TRUE(conv->get_output_partial_shape(0).same_scheme(
        PartialShape{Dimension::dynamic(), 32, Dimension::dynamic(), Dimension::dynamic()}));
}

TEST(type_prop, conv_partial_rank_dynamic_rank_static_dynamic_output_channel_count_known_zero)
{
    PartialShape data_batch_shape{PartialShape::dynamic(4)};
    PartialShape filters_shape{0, Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()};
    Strides window_movement_strides{1, 1};
    Strides window_dilation_strides{1, 1};
    CoordinateDiff padding_below{0, 0};
    CoordinateDiff padding_above{0, 0};
    Strides data_dilation_strides{1, 1};

    auto param0 = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto param1 = make_shared<op::Parameter>(element::f32, filters_shape);

    try
    {
        auto conv = make_shared<op::Convolution>(param0,
                                                 param1,
                                                 window_movement_strides,
                                                 window_dilation_strides,
                                                 padding_below,
                                                 padding_above,
                                                 data_dilation_strides);

        FAIL() << "Zero output channel count not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Filter output channel count is zero"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_partial_rank_dynamic_rank_static_dynamic_input_channel_count_known_ok)
{
    PartialShape data_batch_shape{PartialShape::dynamic(4)};
    PartialShape filters_shape{Dimension::dynamic(), 4, Dimension::dynamic(), Dimension::dynamic()};
    Strides window_movement_strides{1, 1};
    Strides window_dilation_strides{1, 1};
    CoordinateDiff padding_below{0, 0};
    CoordinateDiff padding_above{0, 0};
    Strides data_dilation_strides{1, 1};

    auto param0 = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto param1 = make_shared<op::Parameter>(element::f32, filters_shape);

    auto conv = make_shared<op::Convolution>(param0,
                                             param1,
                                             window_movement_strides,
                                             window_dilation_strides,
                                             padding_below,
                                             padding_above,
                                             data_dilation_strides);

    ASSERT_EQ(conv->get_output_element_type(0), element::f32);
    ASSERT_TRUE(conv->get_output_partial_shape(0).same_scheme(PartialShape::dynamic(4)));
}

TEST(type_prop, conv_partial_rank_dynamic_rank_static_dynamic_input_channel_count_known_zero)
{
    PartialShape data_batch_shape{PartialShape::dynamic(4)};
    PartialShape filters_shape{Dimension::dynamic(), 0, Dimension::dynamic(), Dimension::dynamic()};
    Strides window_movement_strides{1, 1};
    Strides window_dilation_strides{1, 1};
    CoordinateDiff padding_below{0, 0};
    CoordinateDiff padding_above{0, 0};
    Strides data_dilation_strides{1, 1};

    auto param0 = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto param1 = make_shared<op::Parameter>(element::f32, filters_shape);

    try
    {
        auto conv = make_shared<op::Convolution>(param0,
                                                 param1,
                                                 window_movement_strides,
                                                 window_dilation_strides,
                                                 padding_below,
                                                 padding_above,
                                                 data_dilation_strides);

        FAIL() << "Zero input channel count not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Data batch channel count and/or filter input channel count is zero"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_partial_rank_static_dynamic_rank_static_dynamic_ok)
{
    PartialShape data_batch_shape{PartialShape::dynamic(4)};
    PartialShape filters_shape{PartialShape::dynamic(4)};
    Strides window_movement_strides{1, 1};
    Strides window_dilation_strides{1, 1};
    CoordinateDiff padding_below{0, 0};
    CoordinateDiff padding_above{0, 0};
    Strides data_dilation_strides{1, 1};

    auto param0 = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto param1 = make_shared<op::Parameter>(element::f32, filters_shape);

    auto conv = make_shared<op::Convolution>(param0,
                                             param1,
                                             window_movement_strides,
                                             window_dilation_strides,
                                             padding_below,
                                             padding_above,
                                             data_dilation_strides);

    ASSERT_EQ(conv->get_output_element_type(0), element::f32);
    ASSERT_TRUE(conv->get_output_partial_shape(0).same_scheme(PartialShape::dynamic(4)));
}

TEST(type_prop, conv_partial_rank_static_dynamic_rank_static_dynamic_arg_ranks_mismatch)
{
    PartialShape data_batch_shape{PartialShape::dynamic(5)};
    PartialShape filters_shape{PartialShape::dynamic(4)};
    Strides window_movement_strides{1, 1};
    Strides window_dilation_strides{1, 1};
    CoordinateDiff padding_below{0, 0};
    CoordinateDiff padding_above{0, 0};
    Strides data_dilation_strides{1, 1};

    auto param0 = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto param1 = make_shared<op::Parameter>(element::f32, filters_shape);

    try
    {
        auto conv = make_shared<op::Convolution>(param0,
                                                 param1,
                                                 window_movement_strides,
                                                 window_dilation_strides,
                                                 padding_below,
                                                 padding_above,
                                                 data_dilation_strides);

        FAIL() << "Argument rank mismatch not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Data batch and filters rank do not match (data batch "
                                         "shape: {?,?,?,?,?}, filters shape: {?,?,?,?})"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_partial_rank_static_dynamic_rank_static_dynamic_input_channel_counts_known_ok)
{
    PartialShape data_batch_shape{
        Dimension::dynamic(), 3, Dimension::dynamic(), Dimension::dynamic()};
    PartialShape filters_shape{Dimension::dynamic(), 3, Dimension::dynamic(), Dimension::dynamic()};
    Strides window_movement_strides{1, 1};
    Strides window_dilation_strides{1, 1};
    CoordinateDiff padding_below{0, 0};
    CoordinateDiff padding_above{0, 0};
    Strides data_dilation_strides{1, 1};

    auto param0 = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto param1 = make_shared<op::Parameter>(element::f32, filters_shape);

    auto conv = make_shared<op::Convolution>(param0,
                                             param1,
                                             window_movement_strides,
                                             window_dilation_strides,
                                             padding_below,
                                             padding_above,
                                             data_dilation_strides);

    ASSERT_EQ(conv->get_output_element_type(0), element::f32);
    ASSERT_TRUE(conv->get_output_partial_shape(0).same_scheme(PartialShape::dynamic(4)));
}

TEST(type_prop, conv_partial_rank_static_dynamic_rank_static_dynamic_input_channel_counts_mismatch)
{
    PartialShape data_batch_shape{
        Dimension::dynamic(), 3, Dimension::dynamic(), Dimension::dynamic()};
    PartialShape filters_shape{
        Dimension::dynamic(), 22, Dimension::dynamic(), Dimension::dynamic()};
    Strides window_movement_strides{1, 1};
    Strides window_dilation_strides{1, 1};
    CoordinateDiff padding_below{0, 0};
    CoordinateDiff padding_above{0, 0};
    Strides data_dilation_strides{1, 1};

    auto param0 = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto param1 = make_shared<op::Parameter>(element::f32, filters_shape);

    try
    {
        auto conv = make_shared<op::Convolution>(param0,
                                                 param1,
                                                 window_movement_strides,
                                                 window_dilation_strides,
                                                 padding_below,
                                                 padding_above,
                                                 data_dilation_strides);

        FAIL() << "Input channel count mismatch not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string(
                "Data batch channel count (3) does not match filter input channel count (22)"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_partial_rank_static_dynamic_rank_static_dynamic_all_nonspatial_known_ok)
{
    PartialShape data_batch_shape{64, 3, Dimension::dynamic(), Dimension::dynamic()};
    PartialShape filters_shape{100, 3, Dimension::dynamic(), Dimension::dynamic()};
    Strides window_movement_strides{1, 1};
    Strides window_dilation_strides{1, 1};
    CoordinateDiff padding_below{0, 0};
    CoordinateDiff padding_above{0, 0};
    Strides data_dilation_strides{1, 1};

    auto param0 = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto param1 = make_shared<op::Parameter>(element::f32, filters_shape);

    auto conv = make_shared<op::Convolution>(param0,
                                             param1,
                                             window_movement_strides,
                                             window_dilation_strides,
                                             padding_below,
                                             padding_above,
                                             data_dilation_strides);

    ASSERT_EQ(conv->get_output_element_type(0), element::f32);
    ASSERT_TRUE(conv->get_output_partial_shape(0).same_scheme(
        PartialShape{64, 100, Dimension::dynamic(), Dimension::dynamic()}));
}

TEST(type_prop,
     conv_partial_rank_static_dynamic_rank_static_dynamic_all_nonspatial_some_spatial_known_ok)
{
    PartialShape data_batch_shape{64, 3, 200, Dimension::dynamic()};
    PartialShape filters_shape{100, 3, 5, Dimension::dynamic()};
    Strides window_movement_strides{1, 1};
    Strides window_dilation_strides{1, 1};
    CoordinateDiff padding_below{0, 0};
    CoordinateDiff padding_above{0, 0};
    Strides data_dilation_strides{1, 1};

    auto param0 = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto param1 = make_shared<op::Parameter>(element::f32, filters_shape);

    auto conv = make_shared<op::Convolution>(param0,
                                             param1,
                                             window_movement_strides,
                                             window_dilation_strides,
                                             padding_below,
                                             padding_above,
                                             data_dilation_strides);

    ASSERT_EQ(conv->get_output_element_type(0), element::f32);
    ASSERT_TRUE(conv->get_output_partial_shape(0).same_scheme(
        PartialShape{64, 100, 196, Dimension::dynamic()}));
}

TEST(
    type_prop,
    conv_partial_rank_static_dynamic_rank_static_dynamic_all_nonspatial_some_spatial_known_filters_too_big)
{
    PartialShape data_batch_shape{64, 3, 200, Dimension::dynamic()};
    PartialShape filters_shape{100, 3, 201, Dimension::dynamic()};
    Strides window_movement_strides{1, 1};
    Strides window_dilation_strides{1, 1};
    CoordinateDiff padding_below{0, 0};
    CoordinateDiff padding_above{0, 0};
    Strides data_dilation_strides{1, 1};

    auto param0 = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto param1 = make_shared<op::Parameter>(element::f32, filters_shape);

    try
    {
        auto conv = make_shared<op::Convolution>(param0,
                                                 param1,
                                                 window_movement_strides,
                                                 window_dilation_strides,
                                                 padding_below,
                                                 padding_above,
                                                 data_dilation_strides);

        FAIL() << "Oversize filter not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Window after dilation has dimension (dim: 201) larger "
                                         "than the data shape after padding (dim: 200) at axis 0"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(
    type_prop,
    conv_partial_rank_static_dynamic_rank_static_dynamic_all_nonspatial_some_spatial_known_filters_not_too_big_after_padding)
{
    PartialShape data_batch_shape{64, 3, 200, Dimension::dynamic()};
    PartialShape filters_shape{100, 3, 201, Dimension::dynamic()};
    Strides window_movement_strides{1, 1};
    Strides window_dilation_strides{1, 1};
    CoordinateDiff padding_below{2, 0};
    CoordinateDiff padding_above{-1, 0};
    Strides data_dilation_strides{1, 1};

    auto param0 = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto param1 = make_shared<op::Parameter>(element::f32, filters_shape);

    auto conv = make_shared<op::Convolution>(param0,
                                             param1,
                                             window_movement_strides,
                                             window_dilation_strides,
                                             padding_below,
                                             padding_above,
                                             data_dilation_strides);

    ASSERT_EQ(conv->get_output_element_type(0), element::f32);
    ASSERT_TRUE(conv->get_output_partial_shape(0).same_scheme(
        PartialShape{64, 100, 1, Dimension::dynamic()}));
}

TEST(
    type_prop,
    conv_partial_rank_static_dynamic_rank_static_dynamic_all_nonspatial_some_spatial_known_filters_not_too_big_after_data_dilation)
{
    PartialShape data_batch_shape{64, 3, 200, Dimension::dynamic()};
    PartialShape filters_shape{100, 3, 201, Dimension::dynamic()};
    Strides window_movement_strides{1, 1};
    Strides window_dilation_strides{1, 1};
    CoordinateDiff padding_below{0, 0};
    CoordinateDiff padding_above{0, 0};
    Strides data_dilation_strides{2, 1};

    auto param0 = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto param1 = make_shared<op::Parameter>(element::f32, filters_shape);

    auto conv = make_shared<op::Convolution>(param0,
                                             param1,
                                             window_movement_strides,
                                             window_dilation_strides,
                                             padding_below,
                                             padding_above,
                                             data_dilation_strides);

    ASSERT_EQ(conv->get_output_element_type(0), element::f32);
    ASSERT_TRUE(conv->get_output_partial_shape(0).same_scheme(
        PartialShape{64, 100, 199, Dimension::dynamic()}));
}

TEST(
    type_prop,
    conv_partial_rank_static_dynamic_rank_static_dynamic_all_nonspatial_some_spatial_known_filters_not_too_big_after_data_dilation_strided)
{
    PartialShape data_batch_shape{64, 3, 200, Dimension::dynamic()};
    PartialShape filters_shape{100, 3, 201, Dimension::dynamic()};
    Strides window_movement_strides{3, 1};
    Strides window_dilation_strides{1, 1};
    CoordinateDiff padding_below{0, 0};
    CoordinateDiff padding_above{0, 0};
    Strides data_dilation_strides{2, 1};

    auto param0 = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto param1 = make_shared<op::Parameter>(element::f32, filters_shape);

    auto conv = make_shared<op::Convolution>(param0,
                                             param1,
                                             window_movement_strides,
                                             window_dilation_strides,
                                             padding_below,
                                             padding_above,
                                             data_dilation_strides);

    ASSERT_EQ(conv->get_output_element_type(0), element::f32);
    ASSERT_TRUE(conv->get_output_partial_shape(0).same_scheme(
        PartialShape{64, 100, 67, Dimension::dynamic()}));
}

TEST(
    type_prop,
    conv_partial_rank_static_dynamic_rank_static_dynamic_all_nonspatial_some_spatial_known_filters_too_big_after_filter_dilation)
{
    PartialShape data_batch_shape{64, 3, 200, Dimension::dynamic()};
    PartialShape filters_shape{100, 3, 101, Dimension::dynamic()};
    Strides window_movement_strides{1, 1};
    Strides window_dilation_strides{2, 1};
    CoordinateDiff padding_below{0, 0};
    CoordinateDiff padding_above{0, 0};
    Strides data_dilation_strides{1, 1};

    auto param0 = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto param1 = make_shared<op::Parameter>(element::f32, filters_shape);

    try
    {
        auto conv = make_shared<op::Convolution>(param0,
                                                 param1,
                                                 window_movement_strides,
                                                 window_dilation_strides,
                                                 padding_below,
                                                 padding_above,
                                                 data_dilation_strides);

        FAIL() << "Oversize filter after window dilation not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Window after dilation has dimension (dim: 201) larger "
                                         "than the data shape after padding (dim: 200) at axis 0"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(
    type_prop,
    conv_partial_rank_static_dynamic_rank_static_dynamic_all_nonspatial_some_spatial_zero_data_batch_dim)
{
    PartialShape data_batch_shape{64, 3, 200, 0};
    PartialShape filters_shape{100, 3, 5, Dimension::dynamic()};
    Strides window_movement_strides{1, 1};
    Strides window_dilation_strides{1, 1};
    CoordinateDiff padding_below{0, 0};
    CoordinateDiff padding_above{0, 0};
    Strides data_dilation_strides{1, 1};

    auto param0 = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto param1 = make_shared<op::Parameter>(element::f32, filters_shape);

    try
    {
        auto conv = make_shared<op::Convolution>(param0,
                                                 param1,
                                                 window_movement_strides,
                                                 window_dilation_strides,
                                                 padding_below,
                                                 padding_above,
                                                 data_dilation_strides);

        FAIL() << "Zero dimension in data batch not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Data shape after padding and dilation has "
                                         "dimension less than 1 (dim: 0) at axis 1"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(
    type_prop,
    conv_partial_rank_static_dynamic_rank_static_dynamic_all_nonspatial_some_spatial_positive_data_batch_dim_after_padding)
{
    PartialShape data_batch_shape{64, 3, 200, 0};
    PartialShape filters_shape{100, 3, 5, Dimension::dynamic()};
    Strides window_movement_strides{1, 1};
    Strides window_dilation_strides{1, 1};
    CoordinateDiff padding_below{0, 2};
    CoordinateDiff padding_above{0, -1};
    Strides data_dilation_strides{1, 1};

    auto param0 = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto param1 = make_shared<op::Parameter>(element::f32, filters_shape);

    auto conv = make_shared<op::Convolution>(param0,
                                             param1,
                                             window_movement_strides,
                                             window_dilation_strides,
                                             padding_below,
                                             padding_above,
                                             data_dilation_strides);

    ASSERT_EQ(conv->get_output_element_type(0), element::f32);
    ASSERT_TRUE(conv->get_output_partial_shape(0).same_scheme(
        PartialShape{64, 100, 196, Dimension::dynamic()}));
}

TEST(
    type_prop,
    conv_partial_rank_static_dynamic_rank_static_dynamic_all_nonspatial_some_spatial_zero_data_batch_dim_after_padding)
{
    PartialShape data_batch_shape{64, 3, 200, 20};
    PartialShape filters_shape{100, 3, 5, Dimension::dynamic()};
    Strides window_movement_strides{1, 1};
    Strides window_dilation_strides{1, 1};
    CoordinateDiff padding_below{0, 0};
    CoordinateDiff padding_above{0, -20};
    Strides data_dilation_strides{1, 1};

    auto param0 = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto param1 = make_shared<op::Parameter>(element::f32, filters_shape);

    try
    {
        auto conv = make_shared<op::Convolution>(param0,
                                                 param1,
                                                 window_movement_strides,
                                                 window_dilation_strides,
                                                 padding_below,
                                                 padding_above,
                                                 data_dilation_strides);

        FAIL() << "Zero padded dimension in data batch not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Data shape after padding and dilation has "
                                         "dimension less than 1 (dim: 0) at axis 1"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(
    type_prop,
    conv_partial_rank_static_dynamic_rank_static_dynamic_all_nonspatial_some_spatial_negative_data_batch_dim_after_padding)
{
    PartialShape data_batch_shape{64, 3, 200, 20};
    PartialShape filters_shape{100, 3, 5, Dimension::dynamic()};
    Strides window_movement_strides{1, 1};
    Strides window_dilation_strides{1, 1};
    CoordinateDiff padding_below{0, -1};
    CoordinateDiff padding_above{0, -20};
    Strides data_dilation_strides{1, 1};

    auto param0 = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto param1 = make_shared<op::Parameter>(element::f32, filters_shape);

    try
    {
        auto conv = make_shared<op::Convolution>(param0,
                                                 param1,
                                                 window_movement_strides,
                                                 window_dilation_strides,
                                                 padding_below,
                                                 padding_above,
                                                 data_dilation_strides);

        FAIL() << "Negative padded dimension in data batch not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Data shape after padding and dilation has dimension less "
                                         "than 1 (dim: -1) at axis 1"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_partial_dynamic_et)
{
    // For this test the exact shape parameters are kind of arbitrary---just copied and pasted
    // from some known-"OK" test above. We're only concerned about the element types.
    PartialShape data_batch_shape{64, 3, 200, Dimension::dynamic()};
    PartialShape filters_shape{100, 3, 201, Dimension::dynamic()};
    Strides window_movement_strides{1, 1};
    Strides window_dilation_strides{1, 1};
    CoordinateDiff padding_below{2, 0};
    CoordinateDiff padding_above{-1, 0};
    Strides data_dilation_strides{1, 1};

    auto param0 = make_shared<op::Parameter>(element::dynamic, data_batch_shape);
    auto param1 = make_shared<op::Parameter>(element::dynamic, filters_shape);

    auto conv = make_shared<op::Convolution>(param0,
                                             param1,
                                             window_movement_strides,
                                             window_dilation_strides,
                                             padding_below,
                                             padding_above,
                                             data_dilation_strides);

    ASSERT_TRUE(conv->get_output_element_type(0).is_dynamic());
    ASSERT_TRUE(conv->get_output_partial_shape(0).same_scheme(
        PartialShape{64, 100, 1, Dimension::dynamic()}));
}

TEST(type_prop, max_pool_1d_deduce)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{64, 3, 100});
    Shape window_shape{10};
    auto max_pool = make_shared<op::MaxPool>(param, window_shape);

    EXPECT_EQ(max_pool->get_element_type(), element::f32);
    EXPECT_EQ(max_pool->get_shape(), (Shape{64, 3, 91}));

    EXPECT_EQ(max_pool->get_window_movement_strides(), Strides{1});
    EXPECT_EQ(max_pool->get_window_shape(), Shape{10});
}

TEST(type_prop, max_pool_1d_deduce_strided)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{64, 3, 100});
    Shape window_shape{10};
    auto move_strides = Strides{2};
    auto max_pool = make_shared<op::MaxPool>(param, window_shape, move_strides);

    EXPECT_EQ(max_pool->get_element_type(), element::f32);
    EXPECT_EQ(max_pool->get_shape(), (Shape{64, 3, 46}));

    EXPECT_EQ(max_pool->get_window_movement_strides(), Strides{2});
    EXPECT_EQ(max_pool->get_window_shape(), Shape{10});
}

TEST(type_prop, max_pool_1d_deduce_strided_small_uneven)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{64, 3, 5});
    Shape window_shape{2};
    auto move_strides = Strides{2};
    auto max_pool = make_shared<op::MaxPool>(param, window_shape, move_strides);

    EXPECT_EQ(max_pool->get_element_type(), element::f32);
    EXPECT_EQ(max_pool->get_shape(), (Shape{64, 3, 2}));

    EXPECT_EQ(max_pool->get_window_movement_strides(), Strides{2});
    EXPECT_EQ(max_pool->get_window_shape(), Shape{2});
}

TEST(type_prop, max_pool_1d_deduce_strided_small_even)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{64, 3, 6});
    Shape window_shape{2};
    auto move_strides = Strides{2};
    auto max_pool = make_shared<op::MaxPool>(param, window_shape, move_strides);

    EXPECT_EQ(max_pool->get_element_type(), element::f32);
    EXPECT_EQ(max_pool->get_shape(), (Shape{64, 3, 3}));

    EXPECT_EQ(max_pool->get_window_movement_strides(), Strides{2});
    EXPECT_EQ(max_pool->get_window_shape(), Shape{2});
}

TEST(type_prop, max_pool_2d_deduce)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{64, 3, 100, 150});
    Shape window_shape{10, 20};
    auto max_pool = make_shared<op::MaxPool>(param, window_shape);

    EXPECT_EQ(max_pool->get_element_type(), element::f32);
    EXPECT_EQ(max_pool->get_shape(), (Shape{64, 3, 91, 131}));

    EXPECT_EQ(max_pool->get_window_movement_strides(), (Strides{1, 1}));
    EXPECT_EQ(max_pool->get_window_shape(), (Shape{10, 20}));
}

TEST(type_prop, max_pool_2d_deduce_strided)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{64, 3, 100, 150});
    Shape window_shape{10, 20};
    auto move_strides = Strides{2, 3};
    auto max_pool = make_shared<op::MaxPool>(param, window_shape, move_strides);

    EXPECT_EQ(max_pool->get_element_type(), element::f32);
    EXPECT_EQ(max_pool->get_shape(), (Shape{64, 3, 46, 44}));

    EXPECT_EQ(max_pool->get_window_movement_strides(), (Strides{2, 3}));
    EXPECT_EQ(max_pool->get_window_shape(), (Shape{10, 20}));
}

TEST(type_prop, max_pool_3d_deduce_strided_small)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{64, 3, 7, 8, 10});
    Shape window_shape{2, 3, 2};
    auto move_strides = Strides{2, 3, 4};
    auto max_pool = make_shared<op::MaxPool>(param, window_shape, move_strides);

    EXPECT_EQ(max_pool->get_element_type(), element::f32);
    EXPECT_EQ(max_pool->get_shape(), (Shape{64, 3, 3, 2, 3}));

    EXPECT_EQ(max_pool->get_window_movement_strides(), (Strides{2, 3, 4}));
    EXPECT_EQ(max_pool->get_window_shape(), (Shape{2, 3, 2}));
}

TEST(type_prop, max_pool_invalid_0d_input)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{});
    Shape window_shape{};
    try
    {
        auto max_pool = make_shared<op::MaxPool>(param, window_shape);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid 0D input not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Data batch must have rank of at least 3"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, max_pool_invalid_1d_input)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{2});
    Shape window_shape{};
    try
    {
        auto max_pool = make_shared<op::MaxPool>(param, window_shape);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid 1D input not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Data batch must have rank of at least 3"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, max_pool_invalid_2d_input)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{2, 6});
    Shape window_shape{};
    try
    {
        auto max_pool = make_shared<op::MaxPool>(param, window_shape);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid 2D input not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Data batch must have rank of at least 3"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, max_pool_invalid_0_batch_size)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{0, 6, 1});
    Shape window_shape{1};
    try
    {
        auto max_pool = make_shared<op::MaxPool>(param, window_shape);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with 0 batch size not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Batch size is zero"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, max_pool_invalid_0_channels)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{6, 0, 1});
    Shape window_shape{1};
    try
    {
        auto max_pool = make_shared<op::MaxPool>(param, window_shape);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with 0 channels not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Channel count is zero"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, max_pool_invalid_wrong_number_of_window_dimensions_too_many)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{6, 2, 10, 10});
    Shape window_shape{3, 3, 3};
    try
    {
        auto max_pool = make_shared<op::MaxPool>(param, window_shape);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with too many window dimensions not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Ranks for data item shape (data batch has shape {6,2,10,10}, so data item "
                        "rank is 2), padding below (CoordinateDiff{0, 0, 0}), padding above "
                        "(CoordinateDiff{0, 0, 0}), window shape ({3,3,3}), and window strides "
                        "(Strides{1, 1, 1}) do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, max_pool_invalid_wrong_number_of_window_dimensions_too_few)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{6, 2, 10, 10});
    Shape window_shape{3};
    try
    {
        auto max_pool = make_shared<op::MaxPool>(param, window_shape);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with too few window dimensions not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Ranks for data item shape (data batch has shape {6,2,10,10}, so data item "
                        "rank is 2), padding below (CoordinateDiff{0}), padding above "
                        "(CoordinateDiff{0}), window shape ({3}), and window strides (Strides{1}) "
                        "do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, max_pool_invalid_movement_stride_rank)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{6, 2, 10, 10});
    Shape window_shape{3, 3};
    auto move_strides = Strides{2, 3, 8};
    try
    {
        auto max_pool = make_shared<op::MaxPool>(param, window_shape, move_strides);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with wrong movement stride rank not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Ranks for data item shape (data batch has shape {6,2,10,10}, so data item "
                        "rank is 2), padding below (CoordinateDiff{0, 0}), padding above "
                        "(CoordinateDiff{0, 0}), window shape ({3,3}), and window strides "
                        "(Strides{2, 3, 8}) do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, max_pool_invalid_input_data_size_0)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{6, 2, 0, 10});
    Shape window_shape{3, 3};
    try
    {
        auto max_pool = make_shared<op::MaxPool>(param, window_shape);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with zero-length spatial axis not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Data shape after padding and dilation has "
                                         "dimension less than 1 (dim: 0) at axis 0"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, max_pool_invalid_window_size_0)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{6, 2, 10, 10});
    Shape window_shape{3, 0};
    try
    {
        auto max_pool = make_shared<op::MaxPool>(param, window_shape);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with zero-length window axis not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Window after dilation has dimension less than 1 (dim: 0) at axis 1"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, max_pool_invalid_dilated_too_large)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{6, 2, 8, 8});
    Shape window_shape{9, 9};
    try
    {
        auto max_pool = make_shared<op::MaxPool>(param, window_shape);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with oversized window not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Window after dilation has dimension (dim: 9) larger than "
                                         "the data shape after padding (dim: 8) at axis 0"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, max_pool_invalid_movement_stride_0)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{6, 2, 10, 10});
    Shape window_shape{3, 3};
    auto move_strides = Strides{0, 1};
    try
    {
        auto max_pool = make_shared<op::MaxPool>(param, window_shape, move_strides);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with 0-length movement stride axis not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Window strides (Strides{0, 1}) has zero dimension at axis 0"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, max_pool_partial_rank_dynamic_ok)
{
    PartialShape arg_shape{PartialShape::dynamic()};
    Shape window_shape{2, 3, 4, 5};
    Strides window_movement_strides{1, 1, 1, 1};
    Shape padding_below{0, 0, 0, 0};
    Shape padding_above{0, 0, 0, 0};

    auto param = make_shared<op::Parameter>(element::f32, arg_shape);
    auto mp = make_shared<op::MaxPool>(
        param, window_shape, window_movement_strides, padding_below, padding_above);

    ASSERT_EQ(mp->get_output_element_type(0), element::f32);
    ASSERT_TRUE(mp->get_output_partial_shape(0).same_scheme(PartialShape::dynamic(6)));
}

TEST(type_prop, max_pool_partial_rank_dynamic_attrib_rank_mismatch)
{
    PartialShape arg_shape{PartialShape::dynamic()};
    Shape window_shape{2, 3, 4, 5};
    Strides window_movement_strides{1, 1, 1, 1, 1};
    Shape padding_below{0, 0, 0, 0};
    Shape padding_above{0, 0, 0, 0};

    auto param = make_shared<op::Parameter>(element::f32, arg_shape);

    try
    {
        auto mp = make_shared<op::MaxPool>(
            param, window_shape, window_movement_strides, padding_below, padding_above);
        FAIL() << "Mismatch of attribute ranks not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Ranks for data item shape (data batch has shape ?, so data item rank is "
                        "?), padding below (CoordinateDiff{0, 0, 0, 0}), padding above "
                        "(CoordinateDiff{0, 0, 0, 0}), window shape ({2,3,4,5}), and window "
                        "strides (Strides{1, 1, 1, 1, 1}) do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, max_pool_partial_rank_static_dynamic_ok)
{
    PartialShape arg_shape{PartialShape::dynamic(6)};
    Shape window_shape{2, 3, 4, 5};
    Strides window_movement_strides{1, 1, 1, 1};
    Shape padding_below{0, 0, 0, 0};
    Shape padding_above{0, 0, 0, 0};

    auto param = make_shared<op::Parameter>(element::f32, arg_shape);
    auto mp = make_shared<op::MaxPool>(
        param, window_shape, window_movement_strides, padding_below, padding_above);

    ASSERT_EQ(mp->get_output_element_type(0), element::f32);
    ASSERT_TRUE(mp->get_output_partial_shape(0).same_scheme(PartialShape::dynamic(6)));
}

TEST(type_prop, max_pool_partial_rank_static_dynamic_some_dims_known_ok)
{
    PartialShape arg_shape{5, Dimension::dynamic(), 8, Dimension::dynamic(), 4, 7};
    Shape window_shape{2, 3, 4, 5};
    Strides window_movement_strides{1, 1, 1, 1};
    Shape padding_below{0, 0, 0, 0};
    Shape padding_above{0, 0, 0, 0};

    auto param = make_shared<op::Parameter>(element::f32, arg_shape);
    auto mp = make_shared<op::MaxPool>(
        param, window_shape, window_movement_strides, padding_below, padding_above);

    ASSERT_EQ(mp->get_output_element_type(0), element::f32);
    ASSERT_TRUE(mp->get_output_partial_shape(0).same_scheme(
        PartialShape{5, Dimension::dynamic(), 7, Dimension::dynamic(), 1, 3}));
}

TEST(type_prop, max_pool_partial_rank_static_dynamic_attrib_rank_mismatch)
{
    PartialShape arg_shape{5, Dimension::dynamic(), 8, Dimension::dynamic(), 4, 7};
    Shape window_shape{2, 3, 4, 5, 6};
    Strides window_movement_strides{1, 1, 1, 1};
    Shape padding_below{0, 0, 0, 0};
    Shape padding_above{0, 0, 0, 0};

    auto param = make_shared<op::Parameter>(element::f32, arg_shape);

    try
    {
        auto mp = make_shared<op::MaxPool>(
            param, window_shape, window_movement_strides, padding_below, padding_above);
        FAIL() << "Mismatch of attribute ranks not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Ranks for data item shape (data batch has shape {5,?,8,?,4,7}, so data "
                        "item rank is 4), padding below (CoordinateDiff{0, 0, 0, 0}), padding "
                        "above (CoordinateDiff{0, 0, 0, 0}), window shape ({2,3,4,5,6}), and "
                        "window strides (Strides{1, 1, 1, 1}) do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, max_pool_partial_rank_static_dynamic_window_not_too_big)
{
    PartialShape arg_shape{5, Dimension::dynamic(), 8, Dimension::dynamic(), 4, 7};
    Shape window_shape{9, 3, 4, 5};
    Strides window_movement_strides{1, 1, 1, 1};
    Shape padding_below{0, 0, 0, 0};
    Shape padding_above{0, 0, 0, 0};

    auto param = make_shared<op::Parameter>(element::f32, arg_shape);

    try
    {
        auto mp = make_shared<op::MaxPool>(
            param, window_shape, window_movement_strides, padding_below, padding_above);
        FAIL() << "Oversized window not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Window after dilation has dimension (dim: 9) larger than "
                                         "the data shape after padding (dim: 8) at axis 0"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, max_pool_partial_rank_static_dynamic_padded_window_not_too_big)
{
    PartialShape arg_shape{5, Dimension::dynamic(), 8, Dimension::dynamic(), 4, 7};
    Shape window_shape{9, 3, 4, 5};
    Strides window_movement_strides{1, 1, 1, 1};
    Shape padding_below{0, 0, 0, 0};
    Shape padding_above{1, 0, 0, 0};

    auto param = make_shared<op::Parameter>(element::f32, arg_shape);
    auto mp = make_shared<op::MaxPool>(
        param, window_shape, window_movement_strides, padding_below, padding_above);

    ASSERT_EQ(mp->get_output_element_type(0), element::f32);
    ASSERT_TRUE(mp->get_output_partial_shape(0).same_scheme(
        PartialShape{5, Dimension::dynamic(), 1, Dimension::dynamic(), 1, 3}));
}

TEST(type_prop, reverse_0d_deduce)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{});
    auto rev = make_shared<op::Reverse>(param, AxisSet{});

    EXPECT_EQ(rev->get_element_type(), element::f32);
    EXPECT_EQ(rev->get_shape(), (Shape{}));
}

TEST(type_prop, reverse_1d_deduce_nochange)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{5});
    auto rev = make_shared<op::Reverse>(param, AxisSet{});

    EXPECT_EQ(rev->get_element_type(), element::f32);
    EXPECT_EQ(rev->get_shape(), (Shape{5}));
}

TEST(type_prop, reverse_1d_deduce_0)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{5});
    auto rev = make_shared<op::Reverse>(param, AxisSet{0});

    EXPECT_EQ(rev->get_element_type(), element::f32);
    EXPECT_EQ(rev->get_shape(), (Shape{5}));
}

TEST(type_prop, reverse_2d_deduce_nochange)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{5, 6});
    auto rev = make_shared<op::Reverse>(param, AxisSet{});

    EXPECT_EQ(rev->get_element_type(), element::f32);
    EXPECT_EQ(rev->get_shape(), (Shape{5, 6}));
}

TEST(type_prop, reverse_2d_deduce_0)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{5, 6});
    auto rev = make_shared<op::Reverse>(param, AxisSet{0});

    EXPECT_EQ(rev->get_element_type(), element::f32);
    EXPECT_EQ(rev->get_shape(), (Shape{5, 6}));
}

TEST(type_prop, reverse_2d_deduce_1)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{5, 6});
    auto rev = make_shared<op::Reverse>(param, AxisSet{1});

    EXPECT_EQ(rev->get_element_type(), element::f32);
    EXPECT_EQ(rev->get_shape(), (Shape{5, 6}));
}

TEST(type_prop, reverse_2d_deduce_01)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{5, 6});
    auto rev = make_shared<op::Reverse>(param, AxisSet{0, 1});

    EXPECT_EQ(rev->get_element_type(), element::f32);
    EXPECT_EQ(rev->get_shape(), (Shape{5, 6}));
}

TEST(type_prop, reverse_3d_deduce_nochange)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{5, 6, 7});
    auto rev = make_shared<op::Reverse>(param, AxisSet{});

    EXPECT_EQ(rev->get_element_type(), element::f32);
    EXPECT_EQ(rev->get_shape(), (Shape{5, 6, 7}));
}

TEST(type_prop, reverse_3d_deduce_0)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{5, 6, 7});
    auto rev = make_shared<op::Reverse>(param, AxisSet{0});

    EXPECT_EQ(rev->get_element_type(), element::f32);
    EXPECT_EQ(rev->get_shape(), (Shape{5, 6, 7}));
}

TEST(type_prop, reverse_3d_deduce_1)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{5, 6, 7});
    auto rev = make_shared<op::Reverse>(param, AxisSet{1});

    EXPECT_EQ(rev->get_element_type(), element::f32);
    EXPECT_EQ(rev->get_shape(), (Shape{5, 6, 7}));
}

TEST(type_prop, reverse_3d_deduce_2)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{5, 6, 7});
    auto rev = make_shared<op::Reverse>(param, AxisSet{2});

    EXPECT_EQ(rev->get_element_type(), element::f32);
    EXPECT_EQ(rev->get_shape(), (Shape{5, 6, 7}));
}

TEST(type_prop, reverse_3d_deduce_01)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{5, 6, 7});
    auto rev = make_shared<op::Reverse>(param, AxisSet{0, 1});

    EXPECT_EQ(rev->get_element_type(), element::f32);
    EXPECT_EQ(rev->get_shape(), (Shape{5, 6, 7}));
}

TEST(type_prop, reverse_3d_deduce_02)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{5, 6, 7});
    auto rev = make_shared<op::Reverse>(param, AxisSet{0, 2});

    EXPECT_EQ(rev->get_element_type(), element::f32);
    EXPECT_EQ(rev->get_shape(), (Shape{5, 6, 7}));
}

TEST(type_prop, reverse_3d_deduce_12)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{5, 6, 7});
    auto rev = make_shared<op::Reverse>(param, AxisSet{1, 2});

    EXPECT_EQ(rev->get_element_type(), element::f32);
    EXPECT_EQ(rev->get_shape(), (Shape{5, 6, 7}));
}

TEST(type_prop, reverse_3d_deduce_012)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{5, 6, 7});
    auto rev = make_shared<op::Reverse>(param, AxisSet{0, 1, 2});

    EXPECT_EQ(rev->get_element_type(), element::f32);
    EXPECT_EQ(rev->get_shape(), (Shape{5, 6, 7}));
}

TEST(type_prop, reverse_3d_deduce_oob)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{5, 6, 7});
    try
    {
        auto rev = make_shared<op::Reverse>(param, AxisSet{0, 3, 2});

        // Should have thrown, so fail if it didn't
        FAIL() << "Axis out of bounds not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Reverse axis (3) is out of bounds"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

//
// If the input rank is dynamic, we should pass unconditionally.
//
TEST(type_prop, reverse_partial_rank_dynamic)
{
    auto param = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto rev = make_shared<op::Reverse>(param, AxisSet{0, 2, 1776, 90909});

    EXPECT_EQ(rev->get_element_type(), element::f32);
    EXPECT_TRUE(rev->get_output_partial_shape(0).rank().is_dynamic());
}

//
// If the input rank is static but the shape is dynamic, we should pass if the axis indices are
// in bounds.
//
TEST(type_prop, reverse_partial_rank_static_dynamic_axes_ok)
{
    PartialShape param_shape{Dimension::dynamic(), Dimension::dynamic(), 2, 3};
    auto param = make_shared<op::Parameter>(element::f32, param_shape);
    auto rev = make_shared<op::Reverse>(param, AxisSet{0, 2});

    EXPECT_EQ(rev->get_element_type(), element::f32);
    EXPECT_TRUE(rev->get_output_partial_shape(0).same_scheme(param_shape));
}

TEST(type_prop, reverse_partial_rank_static_dynamic_axes_oob)
{
    PartialShape param_shape{Dimension::dynamic(), Dimension::dynamic(), 2, 3};
    auto param = make_shared<op::Parameter>(element::f32, param_shape);
    try
    {
        auto rev = make_shared<op::Reverse>(param, AxisSet{0, 4, 2});

        // Should have thrown, so fail if it didn't
        FAIL() << "Axis out of bounds not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Reverse axis (4) is out of bounds"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, reverse_sequence_1_dim)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{4, 3, 2});
    auto seq_lenghts = make_shared<op::Parameter>(element::f32, Shape{4, 4});
    try
    {
        size_t batch_axis = 0;
        size_t seq_axis = 1;
        auto bc = make_shared<op::ReverseSequence>(data, seq_lenghts, batch_axis, seq_axis);
        FAIL() << "ReverseSequence c-tor should throw for seq_lenghts whose rank isn't equal to 1";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Sequence indices must be a 1-dimensional tensor"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, reverse_sequence_batch_index_oob)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{4, 3, 2});
    auto seq_lenghts = make_shared<op::Parameter>(element::f32, Shape{3});
    try
    {
        size_t batch_axis = 3;
        size_t seq_axis = 1;
        auto bc = make_shared<op::ReverseSequence>(data, seq_lenghts, batch_axis, seq_axis);
        FAIL() << "ReverseSequence c-tor should throw for out-of-bounds batch axis index";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Batch axis index (3) is out of bounds"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, reverse_sequence_sequence_index_oob)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{4, 3, 2});
    auto seq_lengths = make_shared<op::Parameter>(element::f32, Shape{3});
    try
    {
        size_t batch_axis = 1;
        size_t seq_axis = 3;
        auto bc = make_shared<op::ReverseSequence>(data, seq_lengths, batch_axis, seq_axis);
        FAIL() << "ReverseSequence c-tor should throw for out-of-bounds sequence axis index";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Sequence axis index (3) is out of bounds"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, reverse_sequence_seq_len_size_equal_to_batch_dim)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{4, 3, 2});
    auto seq_lenghts = make_shared<op::Parameter>(element::f32, Shape{3});
    try
    {
        size_t batch_axis = 0;
        size_t seq_axis = 1;
        auto bc = make_shared<op::ReverseSequence>(data, seq_lenghts, batch_axis, seq_axis);
        FAIL() << "ReverseSequence c-tor should throw when sequence length size isn't equal to "
                  "batch dimension";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Sequence length (3) is not equal to batch axis dimension (4)"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, reverse_sequence_partial_both_rank_dynamic)
{
    auto data = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto seq_lengths = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    // Unrealistic values, but they don't matter here.
    size_t batch_axis = 202;
    size_t seq_axis = 909;
    auto rs = make_shared<op::ReverseSequence>(data, seq_lengths, batch_axis, seq_axis);

    EXPECT_TRUE(rs->get_output_partial_shape(0).is_dynamic());
    EXPECT_EQ(rs->get_output_element_type(0), element::f32);
}

TEST(type_prop, reverse_sequence_partial_left_rank_dynamic)
{
    auto data = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto seq_lengths = make_shared<op::Parameter>(element::f32, PartialShape{3});
    // Unrealistic values, but they don't matter here.
    size_t batch_axis = 202;
    size_t seq_axis = 909;
    auto rs = make_shared<op::ReverseSequence>(data, seq_lengths, batch_axis, seq_axis);

    EXPECT_TRUE(rs->get_output_partial_shape(0).is_dynamic());
    EXPECT_EQ(rs->get_output_element_type(0), element::f32);
}

TEST(type_prop, reverse_sequence_partial_right_rank_dynamic)
{
    auto data = make_shared<op::Parameter>(element::f32, PartialShape{2, 4, 6, 8});
    auto seq_lengths = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    size_t batch_axis = 0;
    size_t seq_axis = 1;
    auto rs = make_shared<op::ReverseSequence>(data, seq_lengths, batch_axis, seq_axis);

    EXPECT_TRUE(rs->get_output_partial_shape(0).same_scheme(PartialShape{2, 4, 6, 8}));
    EXPECT_EQ(rs->get_output_element_type(0), element::f32);
}

TEST(type_prop, reverse_sequence_partial_both_rank_static_dynamic)
{
    auto data = make_shared<op::Parameter>(element::f32,
                                           PartialShape{Dimension::dynamic(),
                                                        Dimension::dynamic(),
                                                        Dimension::dynamic(),
                                                        Dimension::dynamic()});
    auto seq_lengths = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    size_t batch_axis = 0;
    size_t seq_axis = 1;
    auto rs = make_shared<op::ReverseSequence>(data, seq_lengths, batch_axis, seq_axis);

    EXPECT_TRUE(rs->get_output_partial_shape(0).same_scheme(PartialShape{
        Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()}));
    EXPECT_EQ(rs->get_output_element_type(0), element::f32);
}

TEST(type_prop, reverse_sequence_partial_both_rank_static_dynamic_batch_axis_oob)
{
    auto data = make_shared<op::Parameter>(element::f32,
                                           PartialShape{Dimension::dynamic(),
                                                        Dimension::dynamic(),
                                                        Dimension::dynamic(),
                                                        Dimension::dynamic()});
    auto seq_lengths = make_shared<op::Parameter>(element::f32, PartialShape{Dimension::dynamic()});
    size_t batch_axis = 4;
    size_t seq_axis = 1;
    try
    {
        auto rs = make_shared<op::ReverseSequence>(data, seq_lengths, batch_axis, seq_axis);
        FAIL() << "Batch axis out of bounds not detected (rank-static dynamic shape)";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Batch axis index (4) is out of bounds"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, reverse_sequence_partial_both_rank_static_dynamic_sequence_axis_oob)
{
    auto data = make_shared<op::Parameter>(element::f32,
                                           PartialShape{Dimension::dynamic(),
                                                        Dimension::dynamic(),
                                                        Dimension::dynamic(),
                                                        Dimension::dynamic()});
    auto seq_lengths = make_shared<op::Parameter>(element::f32, PartialShape{Dimension::dynamic()});
    size_t batch_axis = 1;
    size_t seq_axis = 4;
    try
    {
        auto rs = make_shared<op::ReverseSequence>(data, seq_lengths, batch_axis, seq_axis);
        FAIL() << "Sequence axis out of bounds not detected (rank-static dynamic shape)";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Sequence axis index (4) is out of bounds"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop,
     reverse_sequence_partial_left_rank_static_dynamic_right_static_left_seq_length_dynamic)
{
    auto data = make_shared<op::Parameter>(element::f32,
                                           PartialShape{Dimension::dynamic(),
                                                        Dimension::dynamic(),
                                                        Dimension::dynamic(),
                                                        Dimension::dynamic()});
    auto seq_lengths = make_shared<op::Parameter>(element::f32, PartialShape{3});
    size_t batch_axis = 2;
    size_t seq_axis = 1;
    auto rs = make_shared<op::ReverseSequence>(data, seq_lengths, batch_axis, seq_axis);

    EXPECT_TRUE(rs->get_output_partial_shape(0).same_scheme(
        PartialShape{Dimension::dynamic(), Dimension::dynamic(), 3, Dimension::dynamic()}));
    EXPECT_EQ(rs->get_output_element_type(0), element::f32);
}

TEST(type_prop, reverse_sequence_partial_both_rank_static_dynamic_right_seq_length_dynamic)
{
    auto data = make_shared<op::Parameter>(
        element::f32,
        PartialShape{Dimension::dynamic(), Dimension::dynamic(), 3, Dimension::dynamic()});
    auto seq_lengths = make_shared<op::Parameter>(element::f32, PartialShape{Dimension::dynamic()});
    size_t batch_axis = 2;
    size_t seq_axis = 1;
    auto rs = make_shared<op::ReverseSequence>(data, seq_lengths, batch_axis, seq_axis);

    EXPECT_TRUE(rs->get_output_partial_shape(0).same_scheme(
        PartialShape{Dimension::dynamic(), Dimension::dynamic(), 3, Dimension::dynamic()}));
    EXPECT_EQ(rs->get_output_element_type(0), element::f32);
}

TEST(type_prop,
     reverse_sequence_partial_left_rank_static_dynamic_right_static_left_seq_length_static)
{
    auto data = make_shared<op::Parameter>(
        element::f32,
        PartialShape{Dimension::dynamic(), Dimension::dynamic(), 3, Dimension::dynamic()});
    auto seq_lengths = make_shared<op::Parameter>(element::f32, PartialShape{3});
    size_t batch_axis = 2;
    size_t seq_axis = 1;
    auto rs = make_shared<op::ReverseSequence>(data, seq_lengths, batch_axis, seq_axis);

    EXPECT_TRUE(rs->get_output_partial_shape(0).same_scheme(
        PartialShape{Dimension::dynamic(), Dimension::dynamic(), 3, Dimension::dynamic()}));
    EXPECT_EQ(rs->get_output_element_type(0), element::f32);
}

TEST(
    type_prop,
    reverse_sequence_partial_left_rank_static_dynamic_right_static_left_seq_length_static_inconsistent)
{
    auto data = make_shared<op::Parameter>(
        element::f32,
        PartialShape{Dimension::dynamic(), Dimension::dynamic(), 3, Dimension::dynamic()});
    auto seq_lengths = make_shared<op::Parameter>(element::f32, PartialShape{4});
    size_t batch_axis = 2;
    size_t seq_axis = 1;
    try
    {
        auto rs = make_shared<op::ReverseSequence>(data, seq_lengths, batch_axis, seq_axis);
        FAIL() << "Inconsistent sequence length not detected (rank-static dynamic shape)";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Sequence length (4) is not equal to batch axis dimension (3)"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, reduce_window_deduce_1d)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{16});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f =
        make_shared<Function>(f_param_0 + f_param_1, op::ParameterVector{f_param_0, f_param_1});

    Shape window_shape{4};
    Strides move_strides{1};

    auto rw = make_shared<op::ReduceWindow>(param_0, param_1, f, window_shape, move_strides);
    ASSERT_EQ(rw->get_element_type(), element::f32);
    ASSERT_EQ(rw->get_shape(), (Shape{13}));
}

TEST(type_prop, reduce_window_deduce_1d_strided_even)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{16});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f =
        make_shared<Function>(f_param_0 + f_param_1, op::ParameterVector{f_param_0, f_param_1});

    Shape window_shape{4};
    Strides move_strides{4};

    auto rw = make_shared<op::ReduceWindow>(param_0, param_1, f, window_shape, move_strides);
    ASSERT_EQ(rw->get_element_type(), element::f32);
    ASSERT_EQ(rw->get_shape(), (Shape{4}));
}

TEST(type_prop, reduce_window_deduce_1d_strided_uneven)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{18});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f =
        make_shared<Function>(f_param_0 + f_param_1, op::ParameterVector{f_param_0, f_param_1});

    Shape window_shape{4};
    Strides move_strides{4};

    auto rw = make_shared<op::ReduceWindow>(param_0, param_1, f, window_shape, move_strides);
    ASSERT_EQ(rw->get_element_type(), element::f32);
    ASSERT_EQ(rw->get_shape(), (Shape{4}));
}

TEST(type_prop, reduce_window_deduce_2d_strided_uneven)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{18, 10});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f =
        make_shared<Function>(f_param_0 + f_param_1, op::ParameterVector{f_param_0, f_param_1});

    Shape window_shape{4, 2};
    Strides move_strides{4, 3};

    auto rw = make_shared<op::ReduceWindow>(param_0, param_1, f, window_shape, move_strides);
    ASSERT_EQ(rw->get_element_type(), element::f32);
    ASSERT_EQ(rw->get_shape(), (Shape{4, 3}));
}

TEST(type_prop, reduce_window_deduce_3d_strided_uneven)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{18, 10, 15});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f =
        make_shared<Function>(f_param_0 + f_param_1, op::ParameterVector{f_param_0, f_param_1});

    Shape window_shape{4, 2, 4};
    Strides move_strides{4, 3, 2};

    auto rw = make_shared<op::ReduceWindow>(param_0, param_1, f, window_shape, move_strides);
    ASSERT_EQ(rw->get_element_type(), element::f32);
    ASSERT_EQ(rw->get_shape(), (Shape{4, 3, 6}));
}

TEST(type_prop, reduce_window_deduce_non_scalar_init)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{18, 10, 15});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{3});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f =
        make_shared<Function>(f_param_0 + f_param_1, op::ParameterVector{f_param_0, f_param_1});

    Shape window_shape{4, 2, 4};
    Strides move_strides{4, 3, 2};

    try
    {
        auto rw = make_shared<op::ReduceWindow>(param_0, param_1, f, window_shape, move_strides);

        // Should have thrown, so fail if it didn't
        FAIL() << "Non-scalar initial value not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(), std::string("Argument for initial value is not a scalar"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, reduce_window_deduce_different_element_types)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{18, 10, 15});
    auto param_1 = make_shared<op::Parameter>(element::i32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f =
        make_shared<Function>(f_param_0 + f_param_1, op::ParameterVector{f_param_0, f_param_1});

    Shape window_shape{4, 2, 4};
    Strides move_strides{4, 3, 2};

    try
    {
        auto rw = make_shared<op::ReduceWindow>(param_0, param_1, f, window_shape, move_strides);

        // Should have thrown, so fail if it didn't
        FAIL() << "Different element types not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(),
                  std::string("Element types for reductee and initial values do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, reduce_window_deduce_bad_window_shape)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{18, 10, 15});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f =
        make_shared<Function>(f_param_0 + f_param_1, op::ParameterVector{f_param_0, f_param_1});

    Shape window_shape{4, 2};
    Strides move_strides{4, 3, 2};

    try
    {
        auto rw = make_shared<op::ReduceWindow>(param_0, param_1, f, window_shape, move_strides);

        // Should have thrown, so fail if it didn't
        FAIL() << "Bad window shape not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(), std::string("Window shape has different rank from input tensor"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, reduce_window_deduce_bad_move_strides)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{18, 10, 15});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f =
        make_shared<Function>(f_param_0 + f_param_1, op::ParameterVector{f_param_0, f_param_1});

    Shape window_shape{4, 2, 4};
    Strides move_strides{4, 3};

    try
    {
        auto rw = make_shared<op::ReduceWindow>(param_0, param_1, f, window_shape, move_strides);

        // Should have thrown, so fail if it didn't
        FAIL() << "Bad window movement strides not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(),
                  std::string("Window movement strides have different rank from input tensor"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, reduce_window_deduce_zero_length_axis)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{18, 10, 15});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f =
        make_shared<Function>(f_param_0 + f_param_1, op::ParameterVector{f_param_0, f_param_1});

    Shape window_shape{4, 0, 4};
    Strides move_strides{4, 3, 2};

    try
    {
        auto rw = make_shared<op::ReduceWindow>(param_0, param_1, f, window_shape, move_strides);

        // Should have thrown, so fail if it didn't
        FAIL() << "Zero-length window axis not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(), std::string("Window shape has a zero-length axis"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, reduce_window_deduce_zero_length_stride)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{18, 10, 15});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f =
        make_shared<Function>(f_param_0 + f_param_1, op::ParameterVector{f_param_0, f_param_1});

    Shape window_shape{4, 2, 4};
    Strides move_strides{4, 0, 2};

    try
    {
        auto rw = make_shared<op::ReduceWindow>(param_0, param_1, f, window_shape, move_strides);

        // Should have thrown, so fail if it didn't
        FAIL() << "Zero-length window movement stride not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(), std::string("Window movement stride for some axis is zero"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, reduce_window_deduce_window_too_big)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{18, 10, 15});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f =
        make_shared<Function>(f_param_0 + f_param_1, op::ParameterVector{f_param_0, f_param_1});

    Shape window_shape{4, 11, 4};
    Strides move_strides{4, 3, 2};

    try
    {
        auto rw = make_shared<op::ReduceWindow>(param_0, param_1, f, window_shape, move_strides);

        // Should have thrown, so fail if it didn't
        FAIL() << "Window too big not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(), std::string("Reduction window is bigger than input"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, reduce_window_deduce_param_count)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{18, 10, 15});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_2 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(f_param_0 + f_param_1,
                                   op::ParameterVector{f_param_0, f_param_1, f_param_2});

    Shape window_shape{4, 2, 4};
    Strides move_strides{4, 3, 2};

    try
    {
        auto rw = make_shared<op::ReduceWindow>(param_0, param_1, f, window_shape, move_strides);

        // Should have thrown, so fail if it didn't
        FAIL() << "Too many reduction function parameters not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(),
                  std::string("Reduction function has wrong number of parameters (should be two)"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, reduce_window_deduce_param_0_wrong_element_type)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{18, 10, 15});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::i32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(f_param_1, op::ParameterVector{f_param_0, f_param_1});

    Shape window_shape{4, 2, 4};
    Strides move_strides{4, 3, 2};

    try
    {
        auto rw = make_shared<op::ReduceWindow>(param_0, param_1, f, window_shape, move_strides);

        // Should have thrown, so fail if it didn't
        FAIL() << "Parameter 0 wrong type not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(),
                  std::string("Parameter 0 of reduction function has wrong element type"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, reduce_window_deduce_param_0_wrong_shape)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{18, 10, 15});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{1});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(f_param_1, op::ParameterVector{f_param_0, f_param_1});

    Shape window_shape{4, 2, 4};
    Strides move_strides{4, 3, 2};

    try
    {
        auto rw = make_shared<op::ReduceWindow>(param_0, param_1, f, window_shape, move_strides);

        // Should have thrown, so fail if it didn't
        FAIL() << "Parameter 0 wrong type not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(), std::string("Parameter 0 of reduction function is not a scalar"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, reduce_window_deduce_param_1_wrong_element_type)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{18, 10, 15});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::i32, Shape{});
    auto f = make_shared<Function>(f_param_0, op::ParameterVector{f_param_0, f_param_1});

    Shape window_shape{4, 2, 4};
    Strides move_strides{4, 3, 2};

    try
    {
        auto rw = make_shared<op::ReduceWindow>(param_0, param_1, f, window_shape, move_strides);

        // Should have thrown, so fail if it didn't
        FAIL() << "Parameter 1 wrong type not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(),
                  std::string("Parameter 1 of reduction function has wrong element type"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, reduce_window_deduce_param_1_wrong_shape)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{18, 10, 15});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{1});
    auto f = make_shared<Function>(f_param_0, op::ParameterVector{f_param_0, f_param_1});

    Shape window_shape{4, 2, 4};
    Strides move_strides{4, 3, 2};

    try
    {
        auto rw = make_shared<op::ReduceWindow>(param_0, param_1, f, window_shape, move_strides);

        // Should have thrown, so fail if it didn't
        FAIL() << "Parameter 1 wrong type not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(), std::string("Parameter 1 of reduction function is not a scalar"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, reduce_window_deduce_multi_output)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{18, 10, 15});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(NodeVector{f_param_0 + f_param_1, f_param_0 * f_param_1},
                                   op::ParameterVector{f_param_0, f_param_1});

    Shape window_shape{4, 2, 4};
    Strides move_strides{4, 3, 2};

    try
    {
        auto rw = make_shared<op::ReduceWindow>(param_0, param_1, f, window_shape, move_strides);

        // Should have thrown, so fail if it didn't
        FAIL() << "Multiple-output reduction function not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(), std::string("Single-output reduction function was expected"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, reduce_window_reduction_function_return_element_type_mismatch)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{18, 10, 15});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(make_shared<op::Convert>(f_param_0 + f_param_1, element::i32),
                                   op::ParameterVector{f_param_0, f_param_1});

    Shape window_shape{4, 2, 4};
    Strides move_strides{4, 3, 2};

    try
    {
        auto rw = make_shared<op::ReduceWindow>(param_0, param_1, f, window_shape, move_strides);

        // Should have thrown, so fail if it didn't
        FAIL() << "Reduction function return element type mismatch not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(
            error.what(),
            std::string("Return element type from reduction function does not match expected"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, reduce_window_reduction_function_return_shape_mismatch)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{18, 10, 15});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(
        make_shared<op::Broadcast>(f_param_0 + f_param_1, Shape{1}, AxisSet{0}),
        op::ParameterVector{f_param_0, f_param_1});

    Shape window_shape{4, 2, 4};
    Strides move_strides{4, 3, 2};

    try
    {
        auto rw = make_shared<op::ReduceWindow>(param_0, param_1, f, window_shape, move_strides);

        // Should have thrown, so fail if it didn't
        FAIL() << "Reduction function return shape mismatch not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(),
                  std::string("Return shape from reduction function is not a scalar"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, select_and_scatter_deduce_1d)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{16});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{13});
    auto param_2 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(make_shared<op::Greater>(f_param_0, f_param_1),
                                   op::ParameterVector{f_param_0, f_param_1});

    auto g_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g =
        make_shared<Function>(g_param_0 + g_param_1, op::ParameterVector{g_param_0, g_param_1});

    Shape window_shape{4};
    Strides move_strides{1};

    auto sas = make_shared<op::SelectAndScatter>(
        param_0, param_1, param_2, f, g, window_shape, move_strides);
    ASSERT_EQ(sas->get_element_type(), element::f32);
    ASSERT_EQ(sas->get_shape(), (Shape{16}));
}

TEST(type_prop, select_and_scatter_deduce_2d)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{16, 18});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{13, 14});
    auto param_2 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(make_shared<op::Greater>(f_param_0, f_param_1),
                                   op::ParameterVector{f_param_0, f_param_1});

    auto g_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g =
        make_shared<Function>(g_param_0 + g_param_1, op::ParameterVector{g_param_0, g_param_1});

    Shape window_shape{4, 5};
    Strides move_strides{1, 1};

    auto sas = make_shared<op::SelectAndScatter>(
        param_0, param_1, param_2, f, g, window_shape, move_strides);
    ASSERT_EQ(sas->get_element_type(), element::f32);
    ASSERT_EQ(sas->get_shape(), (Shape{16, 18}));
}

TEST(type_prop, select_and_scatter_deduce_3d)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{16, 18, 10});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{13, 14, 9});
    auto param_2 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(make_shared<op::Greater>(f_param_0, f_param_1),
                                   op::ParameterVector{f_param_0, f_param_1});

    auto g_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g =
        make_shared<Function>(g_param_0 + g_param_1, op::ParameterVector{g_param_0, g_param_1});

    Shape window_shape{4, 5, 2};
    Strides move_strides{1, 1, 1};

    auto sas = make_shared<op::SelectAndScatter>(
        param_0, param_1, param_2, f, g, window_shape, move_strides);
    ASSERT_EQ(sas->get_element_type(), element::f32);
    ASSERT_EQ(sas->get_shape(), (Shape{16, 18, 10}));
}

TEST(type_prop, select_and_scatter_deduce_3d_strided)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{16, 18, 10});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{4, 3, 2});
    auto param_2 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(make_shared<op::Greater>(f_param_0, f_param_1),
                                   op::ParameterVector{f_param_0, f_param_1});

    auto g_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g =
        make_shared<Function>(g_param_0 + g_param_1, op::ParameterVector{g_param_0, g_param_1});

    Shape window_shape{4, 3, 2};
    Strides move_strides{4, 6, 5};

    auto sas = make_shared<op::SelectAndScatter>(
        param_0, param_1, param_2, f, g, window_shape, move_strides);
    ASSERT_EQ(sas->get_element_type(), element::f32);
    ASSERT_EQ(sas->get_shape(), (Shape{16, 18, 10}));
}

TEST(type_prop, select_and_scatter_deduce_3d_strided_uneven)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{16, 18, 10});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{2, 3, 3});
    auto param_2 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(make_shared<op::Greater>(f_param_0, f_param_1),
                                   op::ParameterVector{f_param_0, f_param_1});

    auto g_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g =
        make_shared<Function>(g_param_0 + g_param_1, op::ParameterVector{g_param_0, g_param_1});

    Shape window_shape{5, 5, 3};
    Strides move_strides{6, 6, 3};

    auto sas = make_shared<op::SelectAndScatter>(
        param_0, param_1, param_2, f, g, window_shape, move_strides);
    ASSERT_EQ(sas->get_element_type(), element::f32);
    ASSERT_EQ(sas->get_shape(), (Shape{16, 18, 10}));
}

TEST(type_prop, select_and_scatter_deduce_init_not_scalar)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{16, 18, 10});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{2, 3, 3});
    auto param_2 = make_shared<op::Parameter>(element::f32, Shape{4});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(make_shared<op::Greater>(f_param_0, f_param_1),
                                   op::ParameterVector{f_param_0, f_param_1});

    auto g_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g =
        make_shared<Function>(g_param_0 + g_param_1, op::ParameterVector{g_param_0, g_param_1});

    Shape window_shape{5, 5, 3};
    Strides move_strides{6, 6, 3};

    try
    {
        auto sas = make_shared<op::SelectAndScatter>(
            param_0, param_1, param_2, f, g, window_shape, move_strides);

        // Should have thrown, so fail if it didn't
        FAIL() << "Non-scalar init value not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(), std::string("Argument for initial value is not a scalar"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, select_and_scatter_deduce_init_elem_type_wrong)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{16, 18, 10});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{2, 3, 3});
    auto param_2 = make_shared<op::Parameter>(element::i32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(make_shared<op::Greater>(f_param_0, f_param_1),
                                   op::ParameterVector{f_param_0, f_param_1});

    auto g_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g =
        make_shared<Function>(g_param_0 + g_param_1, op::ParameterVector{g_param_0, g_param_1});

    Shape window_shape{5, 5, 3};
    Strides move_strides{6, 6, 3};

    try
    {
        auto sas = make_shared<op::SelectAndScatter>(
            param_0, param_1, param_2, f, g, window_shape, move_strides);

        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect init element type not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(),
                  std::string("Element types for selectee and initial values do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, select_and_scatter_deduce_source_elem_type_wrong)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{16, 18, 10});
    auto param_1 = make_shared<op::Parameter>(element::i32, Shape{2, 3, 3});
    auto param_2 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(make_shared<op::Greater>(f_param_0, f_param_1),
                                   op::ParameterVector{f_param_0, f_param_1});

    auto g_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g =
        make_shared<Function>(g_param_0 + g_param_1, op::ParameterVector{g_param_0, g_param_1});

    Shape window_shape{5, 5, 3};
    Strides move_strides{6, 6, 3};

    try
    {
        auto sas = make_shared<op::SelectAndScatter>(
            param_0, param_1, param_2, f, g, window_shape, move_strides);

        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect source tensor element type not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(),
                  std::string("Element types for selectee and source tensors do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, select_and_scatter_deduce_source_window_shape_wrong_rank)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{16, 18, 10});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{2, 3, 3});
    auto param_2 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(make_shared<op::Greater>(f_param_0, f_param_1),
                                   op::ParameterVector{f_param_0, f_param_1});

    auto g_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g =
        make_shared<Function>(g_param_0 + g_param_1, op::ParameterVector{g_param_0, g_param_1});

    Shape window_shape{5, 5};
    Strides move_strides{6, 6, 3};

    try
    {
        auto sas = make_shared<op::SelectAndScatter>(
            param_0, param_1, param_2, f, g, window_shape, move_strides);

        // Should have thrown, so fail if it didn't
        FAIL() << "Wrong window shape rank not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(),
                  std::string("Window shape has different rank from selectee tensor"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, select_and_scatter_deduce_source_window_strides_wrong_rank)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{16, 18, 10});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{2, 3, 3});
    auto param_2 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(make_shared<op::Greater>(f_param_0, f_param_1),
                                   op::ParameterVector{f_param_0, f_param_1});

    auto g_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g =
        make_shared<Function>(g_param_0 + g_param_1, op::ParameterVector{g_param_0, g_param_1});

    Shape window_shape{5, 5, 3};
    Strides move_strides{6, 6};

    try
    {
        auto sas = make_shared<op::SelectAndScatter>(
            param_0, param_1, param_2, f, g, window_shape, move_strides);

        // Should have thrown, so fail if it didn't
        FAIL() << "Wrong window strides rank not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(),
                  std::string("Window movement strides have different rank from selectee tensor"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, select_and_scatter_deduce_source_window_shape_zero_length_axis)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{16, 18, 10});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{2, 3, 3});
    auto param_2 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(make_shared<op::Greater>(f_param_0, f_param_1),
                                   op::ParameterVector{f_param_0, f_param_1});

    auto g_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g =
        make_shared<Function>(g_param_0 + g_param_1, op::ParameterVector{g_param_0, g_param_1});

    Shape window_shape{5, 0, 3};
    Strides move_strides{6, 6, 3};

    try
    {
        auto sas = make_shared<op::SelectAndScatter>(
            param_0, param_1, param_2, f, g, window_shape, move_strides);

        // Should have thrown, so fail if it didn't
        FAIL() << "Zero-length window shape axis not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(), std::string("Window shape has a zero-length axis"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, select_and_scatter_deduce_source_window_strides_zero_length_axis)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{16, 18, 10});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{2, 3, 3});
    auto param_2 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(make_shared<op::Greater>(f_param_0, f_param_1),
                                   op::ParameterVector{f_param_0, f_param_1});

    auto g_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g =
        make_shared<Function>(g_param_0 + g_param_1, op::ParameterVector{g_param_0, g_param_1});

    Shape window_shape{5, 5, 3};
    Strides move_strides{6, 0, 3};

    try
    {
        auto sas = make_shared<op::SelectAndScatter>(
            param_0, param_1, param_2, f, g, window_shape, move_strides);

        // Should have thrown, so fail if it didn't
        FAIL() << "Zero-length window strides axis not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(), std::string("Window movement stride for some axis is zero"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, select_and_scatter_deduce_source_window_too_big)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{16, 18, 10});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{2, 3, 3});
    auto param_2 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(make_shared<op::Greater>(f_param_0, f_param_1),
                                   op::ParameterVector{f_param_0, f_param_1});

    auto g_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g =
        make_shared<Function>(g_param_0 + g_param_1, op::ParameterVector{g_param_0, g_param_1});

    Shape window_shape{5, 19, 3};
    Strides move_strides{6, 6, 3};

    try
    {
        auto sas = make_shared<op::SelectAndScatter>(
            param_0, param_1, param_2, f, g, window_shape, move_strides);

        // Should have thrown, so fail if it didn't
        FAIL() << "Window too big not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(), std::string("Reduction window is bigger than selectee tensor"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, select_and_scatter_deduce_source_tensor_wrong_shape)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{16, 18, 10});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{2, 4, 3});
    auto param_2 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(make_shared<op::Greater>(f_param_0, f_param_1),
                                   op::ParameterVector{f_param_0, f_param_1});

    auto g_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g =
        make_shared<Function>(g_param_0 + g_param_1, op::ParameterVector{g_param_0, g_param_1});

    Shape window_shape{5, 5, 3};
    Strides move_strides{6, 6, 3};

    try
    {
        auto sas = make_shared<op::SelectAndScatter>(
            param_0, param_1, param_2, f, g, window_shape, move_strides);

        // Should have thrown, so fail if it didn't
        FAIL() << "Wrong source tensor shape not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(), std::string("Source tensor does not have expected shape"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, select_and_scatter_deduce_selection_function_wrong_param_count)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{16, 18, 10});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{2, 3, 3});
    auto param_2 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_2 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(make_shared<op::Greater>(f_param_0, f_param_1),
                                   op::ParameterVector{f_param_0, f_param_1, f_param_2});

    auto g_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g =
        make_shared<Function>(g_param_0 + g_param_1, op::ParameterVector{g_param_0, g_param_1});

    Shape window_shape{5, 5, 3};
    Strides move_strides{6, 6, 3};

    try
    {
        auto sas = make_shared<op::SelectAndScatter>(
            param_0, param_1, param_2, f, g, window_shape, move_strides);

        // Should have thrown, so fail if it didn't
        FAIL() << "Wrong selection function parameter count not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(),
                  std::string("Selection function has wrong number of parameters (should be two)"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, select_and_scatter_deduce_selection_function_wrong_param_0_element_type)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{16, 18, 10});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{2, 3, 3});
    auto param_2 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::i32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(make_shared<op::Greater>(f_param_1, f_param_1),
                                   op::ParameterVector{f_param_0, f_param_1});

    auto g_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g =
        make_shared<Function>(g_param_0 + g_param_1, op::ParameterVector{g_param_0, g_param_1});

    Shape window_shape{5, 5, 3};
    Strides move_strides{6, 6, 3};

    try
    {
        auto sas = make_shared<op::SelectAndScatter>(
            param_0, param_1, param_2, f, g, window_shape, move_strides);

        // Should have thrown, so fail if it didn't
        FAIL() << "Wrong element type for selection function parameter 0 not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(),
                  std::string("Parameter 0 of selection function has wrong element type"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, select_and_scatter_deduce_selection_function_wrong_param_0_shape)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{16, 18, 10});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{2, 3, 3});
    auto param_2 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{1});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(make_shared<op::Greater>(f_param_1, f_param_1),
                                   op::ParameterVector{f_param_0, f_param_1});

    auto g_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g =
        make_shared<Function>(g_param_0 + g_param_1, op::ParameterVector{g_param_0, g_param_1});

    Shape window_shape{5, 5, 3};
    Strides move_strides{6, 6, 3};

    try
    {
        auto sas = make_shared<op::SelectAndScatter>(
            param_0, param_1, param_2, f, g, window_shape, move_strides);

        // Should have thrown, so fail if it didn't
        FAIL() << "Wrong shape for selection function parameter 0 not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(), std::string("Parameter 0 of selection function is not a scalar"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, select_and_scatter_deduce_selection_function_wrong_param_1_element_type)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{16, 18, 10});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{2, 3, 3});
    auto param_2 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::i32, Shape{});
    auto f = make_shared<Function>(make_shared<op::Greater>(f_param_0, f_param_0),
                                   op::ParameterVector{f_param_0, f_param_1});

    auto g_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g =
        make_shared<Function>(g_param_0 + g_param_1, op::ParameterVector{g_param_0, g_param_1});

    Shape window_shape{5, 5, 3};
    Strides move_strides{6, 6, 3};

    try
    {
        auto sas = make_shared<op::SelectAndScatter>(
            param_0, param_1, param_2, f, g, window_shape, move_strides);

        // Should have thrown, so fail if it didn't
        FAIL() << "Wrong element type for selection function parameter 1 not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(),
                  std::string("Parameter 1 of selection function has wrong element type"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, select_and_scatter_deduce_selection_function_wrong_param_1_shape)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{16, 18, 10});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{2, 3, 3});
    auto param_2 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{1});
    auto f = make_shared<Function>(make_shared<op::Greater>(f_param_0, f_param_0),
                                   op::ParameterVector{f_param_0, f_param_1});

    auto g_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g =
        make_shared<Function>(g_param_0 + g_param_1, op::ParameterVector{g_param_0, g_param_1});

    Shape window_shape{5, 5, 3};
    Strides move_strides{6, 6, 3};

    try
    {
        auto sas = make_shared<op::SelectAndScatter>(
            param_0, param_1, param_2, f, g, window_shape, move_strides);

        // Should have thrown, so fail if it didn't
        FAIL() << "Wrong shape for selection function parameter 1 not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(), std::string("Parameter 1 of selection function is not a scalar"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, select_and_scatter_deduce_selection_function_multi_output)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{16, 18, 10});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{2, 3, 3});
    auto param_2 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(
        std::vector<std::shared_ptr<Node>>{make_shared<op::Greater>(f_param_0, f_param_1),
                                           make_shared<op::Greater>(f_param_0, f_param_1)},
        op::ParameterVector{f_param_0, f_param_1});

    auto g_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g =
        make_shared<Function>(g_param_0 + g_param_1, op::ParameterVector{g_param_0, g_param_1});

    Shape window_shape{5, 5, 3};
    Strides move_strides{6, 6, 3};

    try
    {
        auto sas = make_shared<op::SelectAndScatter>(
            param_0, param_1, param_2, f, g, window_shape, move_strides);

        // Should have thrown, so fail if it didn't
        FAIL() << "Multi-output selection function not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(), std::string("Single-output selection function was expected"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, select_and_scatter_deduce_selection_function_wrong_result_element_type)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{16, 18, 10});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{2, 3, 3});
    auto param_2 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(make_shared<op::Add>(f_param_0, f_param_1),
                                   op::ParameterVector{f_param_0, f_param_1});

    auto g_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g =
        make_shared<Function>(g_param_0 + g_param_1, op::ParameterVector{g_param_0, g_param_1});

    Shape window_shape{5, 5, 3};
    Strides move_strides{6, 6, 3};

    try
    {
        auto sas = make_shared<op::SelectAndScatter>(
            param_0, param_1, param_2, f, g, window_shape, move_strides);

        // Should have thrown, so fail if it didn't
        FAIL() << "Wrong selection function result element type not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(),
                  std::string("Return element type from selection function is not boolean"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, select_and_scatter_deduce_selection_function_wrong_result_shape)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{16, 18, 10});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{2, 3, 3});
    auto param_2 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(
        make_shared<op::Broadcast>(
            make_shared<op::Greater>(f_param_0, f_param_1), Shape{1}, AxisSet{0}),
        op::ParameterVector{f_param_0, f_param_1});

    auto g_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g =
        make_shared<Function>(g_param_0 + g_param_1, op::ParameterVector{g_param_0, g_param_1});

    Shape window_shape{5, 5, 3};
    Strides move_strides{6, 6, 3};

    try
    {
        auto sas = make_shared<op::SelectAndScatter>(
            param_0, param_1, param_2, f, g, window_shape, move_strides);

        // Should have thrown, so fail if it didn't
        FAIL() << "Wrong selection function result type not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(),
                  std::string("Return shape from selection function is not a scalar"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, select_and_scatter_deduce_scatter_function_wrong_param_count)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{16, 18, 10});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{2, 3, 3});
    auto param_2 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(make_shared<op::Greater>(f_param_0, f_param_1),
                                   op::ParameterVector{f_param_0, f_param_1});

    auto g_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g_param_2 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g = make_shared<Function>(g_param_0 + g_param_1,
                                   op::ParameterVector{g_param_0, g_param_1, g_param_2});

    Shape window_shape{5, 5, 3};
    Strides move_strides{6, 6, 3};

    try
    {
        auto sas = make_shared<op::SelectAndScatter>(
            param_0, param_1, param_2, f, g, window_shape, move_strides);

        // Should have thrown, so fail if it didn't
        FAIL() << "Wrong scatter function parameter count not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(),
                  std::string("Scatter function has wrong number of parameters (should be two)"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, select_and_scatter_deduce_scatter_function_wrong_param_0_element_type)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{16, 18, 10});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{2, 3, 3});
    auto param_2 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(make_shared<op::Greater>(f_param_0, f_param_1),
                                   op::ParameterVector{f_param_0, f_param_1});

    auto g_param_0 = make_shared<op::Parameter>(element::i32, Shape{});
    auto g_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g =
        make_shared<Function>(g_param_1 + g_param_1, op::ParameterVector{g_param_0, g_param_1});

    Shape window_shape{5, 5, 3};
    Strides move_strides{6, 6, 3};

    try
    {
        auto sas = make_shared<op::SelectAndScatter>(
            param_0, param_1, param_2, f, g, window_shape, move_strides);

        // Should have thrown, so fail if it didn't
        FAIL() << "Wrong element type for scatter function parameter 0 not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(),
                  std::string("Parameter 0 of scatter function has wrong element type"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, select_and_scatter_deduce_scatter_function_wrong_param_0_shape)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{16, 18, 10});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{2, 3, 3});
    auto param_2 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(make_shared<op::Greater>(f_param_0, f_param_1),
                                   op::ParameterVector{f_param_0, f_param_1});

    auto g_param_0 = make_shared<op::Parameter>(element::f32, Shape{1});
    auto g_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g =
        make_shared<Function>(g_param_1 + g_param_1, op::ParameterVector{g_param_0, g_param_1});

    Shape window_shape{5, 5, 3};
    Strides move_strides{6, 6, 3};

    try
    {
        auto sas = make_shared<op::SelectAndScatter>(
            param_0, param_1, param_2, f, g, window_shape, move_strides);

        // Should have thrown, so fail if it didn't
        FAIL() << "Wrong shape for scatter function parameter 0 not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(), std::string("Parameter 0 of scatter function is not a scalar"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, select_and_scatter_deduce_scatter_function_wrong_param_1_element_type)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{16, 18, 10});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{2, 3, 3});
    auto param_2 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(make_shared<op::Greater>(f_param_0, f_param_1),
                                   op::ParameterVector{f_param_0, f_param_1});

    auto g_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g_param_1 = make_shared<op::Parameter>(element::i32, Shape{});
    auto g =
        make_shared<Function>(g_param_0 + g_param_0, op::ParameterVector{g_param_0, g_param_1});

    Shape window_shape{5, 5, 3};
    Strides move_strides{6, 6, 3};

    try
    {
        auto sas = make_shared<op::SelectAndScatter>(
            param_0, param_1, param_2, f, g, window_shape, move_strides);

        // Should have thrown, so fail if it didn't
        FAIL() << "Wrong element type for scatter function parameter 1 not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(),
                  std::string("Parameter 1 of scatter function has wrong element type"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, select_and_scatter_deduce_scatter_function_wrong_param_1_shape)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{16, 18, 10});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{2, 3, 3});
    auto param_2 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(make_shared<op::Greater>(f_param_0, f_param_1),
                                   op::ParameterVector{f_param_0, f_param_1});

    auto g_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g_param_1 = make_shared<op::Parameter>(element::f32, Shape{1});
    auto g =
        make_shared<Function>(g_param_0 + g_param_0, op::ParameterVector{g_param_0, g_param_1});

    Shape window_shape{5, 5, 3};
    Strides move_strides{6, 6, 3};

    try
    {
        auto sas = make_shared<op::SelectAndScatter>(
            param_0, param_1, param_2, f, g, window_shape, move_strides);

        // Should have thrown, so fail if it didn't
        FAIL() << "Wrong shape for scatter function parameter 1 not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(), std::string("Parameter 1 of scatter function is not a scalar"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, select_and_scatter_deduce_scatter_function_multi_output)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{16, 18, 10});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{2, 3, 3});
    auto param_2 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(make_shared<op::Greater>(f_param_0, f_param_1),
                                   op::ParameterVector{f_param_0, f_param_1});

    auto g_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g = make_shared<Function>(
        std::vector<std::shared_ptr<Node>>{g_param_0 + g_param_1, g_param_0 + g_param_1},
        op::ParameterVector{g_param_0, g_param_1});

    Shape window_shape{5, 5, 3};
    Strides move_strides{6, 6, 3};

    try
    {
        auto sas = make_shared<op::SelectAndScatter>(
            param_0, param_1, param_2, f, g, window_shape, move_strides);

        // Should have thrown, so fail if it didn't
        FAIL() << "Multi-output scatter function not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(), std::string("Single-output scatter function was expected"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, select_and_scatter_deduce_scatter_function_wrong_result_element_type)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{16, 18, 10});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{2, 3, 3});
    auto param_2 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(make_shared<op::Greater>(f_param_0, f_param_1),
                                   op::ParameterVector{f_param_0, f_param_1});

    auto g_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g = make_shared<Function>(make_shared<op::Greater>(g_param_0, g_param_1),
                                   op::ParameterVector{g_param_0, g_param_1});

    Shape window_shape{5, 5, 3};
    Strides move_strides{6, 6, 3};

    try
    {
        auto sas = make_shared<op::SelectAndScatter>(
            param_0, param_1, param_2, f, g, window_shape, move_strides);

        // Should have thrown, so fail if it didn't
        FAIL() << "Wrong scatter function result element type not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(),
                  std::string("Return element type from scatter function does "
                              "not match the init value type"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, select_and_scatter_deduce_scatter_function_wrong_result_shape)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{16, 18, 10});
    auto param_1 = make_shared<op::Parameter>(element::f32, Shape{2, 3, 3});
    auto param_2 = make_shared<op::Parameter>(element::f32, Shape{});

    auto f_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(make_shared<op::Greater>(f_param_0, f_param_1),
                                   op::ParameterVector{f_param_0, f_param_1});

    auto g_param_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g_param_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto g = make_shared<Function>(
        make_shared<op::Broadcast>(g_param_0 + g_param_1, Shape{1}, AxisSet{0}),
        op::ParameterVector{g_param_0, g_param_1});

    Shape window_shape{5, 5, 3};
    Strides move_strides{6, 6, 3};

    try
    {
        auto sas = make_shared<op::SelectAndScatter>(
            param_0, param_1, param_2, f, g, window_shape, move_strides);

        // Should have thrown, so fail if it didn't
        FAIL() << "Wrong scatter function result shape not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_EQ(error.what(), std::string("Return shape from scatter function is not a scalar"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, avg_pool_1d_deduce)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{64, 3, 100});
    Shape window_shape{10};
    auto avg_pool = make_shared<op::AvgPool>(param, window_shape);

    EXPECT_EQ(avg_pool->get_element_type(), element::f32);
    EXPECT_EQ(avg_pool->get_shape(), (Shape{64, 3, 91}));

    EXPECT_EQ(avg_pool->get_window_movement_strides(), Strides{1});
    EXPECT_EQ(avg_pool->get_window_shape(), Shape{10});
    EXPECT_EQ(avg_pool->get_padding_below(), Shape{0});
    EXPECT_EQ(avg_pool->get_padding_above(), Shape{0});
}

TEST(type_prop, avg_pool_1d_deduce_strided)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{64, 3, 100});
    Shape window_shape{10};
    auto move_strides = Strides{2};
    auto avg_pool = make_shared<op::AvgPool>(param, window_shape, move_strides);

    EXPECT_EQ(avg_pool->get_element_type(), element::f32);
    EXPECT_EQ(avg_pool->get_shape(), (Shape{64, 3, 46}));

    EXPECT_EQ(avg_pool->get_window_movement_strides(), Strides{2});
    EXPECT_EQ(avg_pool->get_window_shape(), Shape{10});
    EXPECT_EQ(avg_pool->get_padding_below(), Shape{0});
    EXPECT_EQ(avg_pool->get_padding_above(), Shape{0});
}

TEST(type_prop, avg_pool_1d_deduce_strided_small_uneven)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{64, 3, 5});
    Shape window_shape{2};
    auto move_strides = Strides{2};
    auto avg_pool = make_shared<op::AvgPool>(param, window_shape, move_strides);

    EXPECT_EQ(avg_pool->get_element_type(), element::f32);
    EXPECT_EQ(avg_pool->get_shape(), (Shape{64, 3, 2}));

    EXPECT_EQ(avg_pool->get_window_movement_strides(), Strides{2});
    EXPECT_EQ(avg_pool->get_window_shape(), Shape{2});
    EXPECT_EQ(avg_pool->get_padding_below(), Shape{0});
    EXPECT_EQ(avg_pool->get_padding_above(), Shape{0});
}

TEST(type_prop, avg_pool_1d_deduce_strided_small_even)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{64, 3, 6});
    Shape window_shape{2};
    auto move_strides = Strides{2};
    auto avg_pool = make_shared<op::AvgPool>(param, window_shape, move_strides);

    EXPECT_EQ(avg_pool->get_element_type(), element::f32);
    EXPECT_EQ(avg_pool->get_shape(), (Shape{64, 3, 3}));

    EXPECT_EQ(avg_pool->get_window_movement_strides(), Strides{2});
    EXPECT_EQ(avg_pool->get_window_shape(), Shape{2});
    EXPECT_EQ(avg_pool->get_padding_below(), Shape{0});
    EXPECT_EQ(avg_pool->get_padding_above(), Shape{0});
}

TEST(type_prop, avg_pool_2d_deduce)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{64, 3, 100, 150});
    Shape window_shape{10, 20};
    auto avg_pool = make_shared<op::AvgPool>(param, window_shape);

    EXPECT_EQ(avg_pool->get_element_type(), element::f32);
    EXPECT_EQ(avg_pool->get_shape(), (Shape{64, 3, 91, 131}));

    EXPECT_EQ(avg_pool->get_window_movement_strides(), (Strides{1, 1}));
    EXPECT_EQ(avg_pool->get_window_shape(), (Shape{10, 20}));
    EXPECT_EQ(avg_pool->get_padding_below(), (Shape{0, 0}));
    EXPECT_EQ(avg_pool->get_padding_above(), (Shape{0, 0}));
}

TEST(type_prop, avg_pool_2d_deduce_strided)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{64, 3, 100, 150});
    Shape window_shape{10, 20};
    auto move_strides = Strides{2, 3};
    auto avg_pool = make_shared<op::AvgPool>(param, window_shape, move_strides);

    EXPECT_EQ(avg_pool->get_element_type(), element::f32);
    EXPECT_EQ(avg_pool->get_shape(), (Shape{64, 3, 46, 44}));

    EXPECT_EQ(avg_pool->get_window_movement_strides(), (Strides{2, 3}));
    EXPECT_EQ(avg_pool->get_window_shape(), (Shape{10, 20}));
    EXPECT_EQ(avg_pool->get_padding_below(), (Shape{0, 0}));
    EXPECT_EQ(avg_pool->get_padding_above(), (Shape{0, 0}));
}

TEST(type_prop, avg_pool_3d_deduce_strided_small)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{64, 3, 7, 8, 10});
    Shape window_shape{2, 3, 2};
    auto move_strides = Strides{2, 3, 4};
    auto avg_pool = make_shared<op::AvgPool>(param, window_shape, move_strides);

    EXPECT_EQ(avg_pool->get_element_type(), element::f32);
    EXPECT_EQ(avg_pool->get_shape(), (Shape{64, 3, 3, 2, 3}));

    EXPECT_EQ(avg_pool->get_window_movement_strides(), (Strides{2, 3, 4}));
    EXPECT_EQ(avg_pool->get_window_shape(), (Shape{2, 3, 2}));
    EXPECT_EQ(avg_pool->get_padding_below(), (Shape{0, 0, 0}));
    EXPECT_EQ(avg_pool->get_padding_above(), (Shape{0, 0, 0}));
}

TEST(type_prop, avg_pool_3d_deduce_strided_padded_small)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{64, 3, 7, 8, 10});
    Shape window_shape{2, 3, 2};
    auto move_strides = Strides{2, 3, 4};
    Shape padding_below{5, 6, 4};
    Shape padding_above{6, 4, 5};
    auto avg_pool = make_shared<op::AvgPool>(
        param, window_shape, move_strides, padding_below, padding_above, true);

    EXPECT_EQ(avg_pool->get_element_type(), element::f32);
    EXPECT_EQ(avg_pool->get_shape(), (Shape{64, 3, 9, 6, 5}));

    EXPECT_EQ(avg_pool->get_window_movement_strides(), (Strides{2, 3, 4}));
    EXPECT_EQ(avg_pool->get_window_shape(), (Shape{2, 3, 2}));
    EXPECT_EQ(avg_pool->get_padding_below(), (Shape{5, 6, 4}));
    EXPECT_EQ(avg_pool->get_padding_above(), (Shape{6, 4, 5}));
}

TEST(type_prop, avg_pool_invalid_0d_input)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{});
    Shape window_shape{};
    try
    {
        auto avg_pool = make_shared<op::AvgPool>(param, window_shape);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid 0D input not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Data batch must have rank of at least 3 (one batch axis, one "
                             "input-channel axis, and at least one spatial dimension)");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, avg_pool_invalid_1d_input)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{2});
    Shape window_shape{};
    try
    {
        auto avg_pool = make_shared<op::AvgPool>(param, window_shape);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid 1D input not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Data batch must have rank of at least 3 (one batch axis, one "
                             "input-channel axis, and at least one spatial dimension)");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, avg_pool_invalid_2d_input)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{2, 6});
    Shape window_shape{};
    try
    {
        auto avg_pool = make_shared<op::AvgPool>(param, window_shape);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid 2D input not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Data batch must have rank of at least 3 (one batch axis, one "
                             "input-channel axis, and at least one spatial dimension)");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, avg_pool_invalid_0_batch_size)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{0, 6, 1});
    Shape window_shape{1};
    try
    {
        auto avg_pool = make_shared<op::AvgPool>(param, window_shape);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with 0 batch size not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Batch size is zero");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, avg_pool_invalid_0_channels)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{6, 0, 1});
    Shape window_shape{1};
    try
    {
        auto avg_pool = make_shared<op::AvgPool>(param, window_shape);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with 0 channels not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Channel count is zero");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, avg_pool_invalid_wrong_number_of_window_dimensions_too_many)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{6, 2, 10, 10});
    Shape window_shape{3, 3, 3};
    try
    {
        auto avg_pool = make_shared<op::AvgPool>(param, window_shape);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with too many window dimensions not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Ranks for data item shape (data batch has shape {6,2,10,10}, so data "
                             "item rank is 2), padding below (CoordinateDiff{0, 0, 0}), padding "
                             "above (CoordinateDiff{0, 0, 0}), window shape ({3,3,3}), and window "
                             "strides (Strides{1, 1, 1}) do not match");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, avg_pool_invalid_wrong_number_of_window_dimensions_too_few)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{6, 2, 10, 10});
    Shape window_shape{3};
    try
    {
        auto avg_pool = make_shared<op::AvgPool>(param, window_shape);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with too few window dimensions not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Ranks for data item shape (data batch has shape {6,2,10,10}, so data "
                             "item rank is 2), padding below (CoordinateDiff{0}), padding above "
                             "(CoordinateDiff{0}), window shape ({3}), and window strides "
                             "(Strides{1}) do not match");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, avg_pool_invalid_movement_stride_rank)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{6, 2, 10, 10});
    Shape window_shape{3, 3};
    auto move_strides = Strides{2, 3, 8};
    try
    {
        auto avg_pool = make_shared<op::AvgPool>(param, window_shape, move_strides);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with wrong movement stride rank not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Ranks for data item shape (data batch has shape {6,2,10,10}, so data "
                             "item rank is 2), padding below (CoordinateDiff{0, 0}), padding above "
                             "(CoordinateDiff{0, 0}), window shape ({3,3}), and window strides "
                             "(Strides{2, 3, 8}) do not match");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, avg_pool_invalid_padding_below_rank)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{6, 2, 10, 10});
    Shape window_shape{3, 3};
    auto move_strides = Strides{2, 3};
    Shape padding_below{1, 2, 3};
    Shape padding_above{1, 2};
    try
    {
        auto avg_pool = make_shared<op::AvgPool>(
            param, window_shape, move_strides, padding_below, padding_above, false);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with wrong below-padding rank not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Ranks for data item shape (data batch has shape {6,2,10,10}, so data "
                             "item rank is 2), padding below (CoordinateDiff{1, 2, 3}), padding "
                             "above (CoordinateDiff{1, 2}), window shape ({3,3}), and window "
                             "strides (Strides{2, 3}) do not match");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, avg_pool_invalid_padding_above_rank)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{6, 2, 10, 10});
    Shape window_shape{3, 3};
    auto move_strides = Strides{2, 3};
    Shape padding_below{1, 2};
    Shape padding_above{1, 2, 3};
    try
    {
        auto avg_pool = make_shared<op::AvgPool>(
            param, window_shape, move_strides, padding_below, padding_above, false);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with wrong above-padding rank not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Ranks for data item shape (data batch has shape {6,2,10,10}, so data "
                             "item rank is 2), padding below (CoordinateDiff{1, 2}), padding above "
                             "(CoordinateDiff{1, 2, 3}), window shape ({3,3}), and window strides "
                             "(Strides{2, 3}) do not match");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, avg_pool_invalid_input_item_size_0)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{6, 2, 0, 10});
    Shape window_shape{3, 3};
    try
    {
        auto avg_pool = make_shared<op::AvgPool>(param, window_shape);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with zero-length spatial axis not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            "Data shape after padding and dilation has dimension less than 1 (dim: 0) at axis 0");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, avg_pool_invalid_window_size_0)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{6, 2, 10, 10});
    Shape window_shape{3, 0};
    try
    {
        auto avg_pool = make_shared<op::AvgPool>(param, window_shape);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with zero-length window axis not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Window after dilation has dimension less than 1 (dim: 0) at axis 1");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, avg_pool_invalid_dilated_too_large)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{6, 2, 8, 8});
    Shape window_shape{9, 9};
    try
    {
        auto avg_pool = make_shared<op::AvgPool>(param, window_shape);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with oversized window not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Window after dilation has dimension (dim: 9) larger than the data "
                             "shape after padding (dim: 8) at axis 0");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, avg_pool_larger_than_pre_padding_but_fits_in_post_padding)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{6, 2, 8, 8});
    Shape window_shape{9, 9};
    Strides window_strides{1, 1};
    Shape padding_below{0, 0};
    Shape padding_above{1, 1};
    auto avg_pool =
        make_shared<op::AvgPool>(param, window_shape, window_strides, padding_below, padding_above);

    ASSERT_EQ(avg_pool->get_output_element_type(0), element::f32);
    ASSERT_EQ(avg_pool->get_output_shape(0), (Shape{6, 2, 1, 1}));
}

TEST(type_prop, avg_pool_invalid_movement_stride_0)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{6, 2, 10, 10});
    Shape window_shape{3, 3};
    auto move_strides = Strides{0, 1};
    try
    {
        auto avg_pool = make_shared<op::AvgPool>(param, window_shape, move_strides);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with 0-length movement stride axis not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Window strides (Strides{0, 1}) has zero dimension at axis 0");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, avg_pool_partial_rank_dynamic_ok)
{
    PartialShape arg_shape{PartialShape::dynamic()};
    Shape window_shape{2, 3, 4, 5};
    Strides window_movement_strides{1, 1, 1, 1};
    Shape padding_below{0, 0, 0, 0};
    Shape padding_above{0, 0, 0, 0};
    bool include_padding_in_average = false;

    auto param = make_shared<op::Parameter>(element::f32, arg_shape);
    auto ap = make_shared<op::AvgPool>(param,
                                       window_shape,
                                       window_movement_strides,
                                       padding_below,
                                       padding_above,
                                       include_padding_in_average);

    ASSERT_EQ(ap->get_output_element_type(0), element::f32);
    ASSERT_TRUE(ap->get_output_partial_shape(0).same_scheme(PartialShape::dynamic(6)));
}

TEST(type_prop, avg_pool_partial_rank_dynamic_attrib_rank_mismatch)
{
    PartialShape arg_shape{PartialShape::dynamic()};
    Shape window_shape{2, 3, 4, 5};
    Strides window_movement_strides{1, 1, 1, 1, 1};
    Shape padding_below{0, 0, 0, 0};
    Shape padding_above{0, 0, 0, 0};
    bool include_padding_in_average = false;

    auto param = make_shared<op::Parameter>(element::f32, arg_shape);

    try
    {
        auto ap = make_shared<op::AvgPool>(param,
                                           window_shape,
                                           window_movement_strides,
                                           padding_below,
                                           padding_above,
                                           include_padding_in_average);
        FAIL() << "Mismatch of attribute ranks not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Ranks for data item shape (data batch has shape ?, so data item rank is "
                        "?), padding below (CoordinateDiff{0, 0, 0, 0}), padding above "
                        "(CoordinateDiff{0, 0, 0, 0}), window shape ({2,3,4,5}), and window "
                        "strides (Strides{1, 1, 1, 1, 1}) do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, avg_pool_partial_rank_static_dynamic_ok)
{
    PartialShape arg_shape{PartialShape::dynamic(6)};
    Shape window_shape{2, 3, 4, 5};
    Strides window_movement_strides{1, 1, 1, 1};
    Shape padding_below{0, 0, 0, 0};
    Shape padding_above{0, 0, 0, 0};
    bool include_padding_in_average = false;

    auto param = make_shared<op::Parameter>(element::f32, arg_shape);
    auto ap = make_shared<op::AvgPool>(param,
                                       window_shape,
                                       window_movement_strides,
                                       padding_below,
                                       padding_above,
                                       include_padding_in_average);

    ASSERT_EQ(ap->get_output_element_type(0), element::f32);
    ASSERT_TRUE(ap->get_output_partial_shape(0).same_scheme(PartialShape::dynamic(6)));
}

TEST(type_prop, avg_pool_partial_rank_static_dynamic_some_dims_known_ok)
{
    PartialShape arg_shape{5, Dimension::dynamic(), 8, Dimension::dynamic(), 4, 7};
    Shape window_shape{2, 3, 4, 5};
    Strides window_movement_strides{1, 1, 1, 1};
    Shape padding_below{0, 0, 0, 0};
    Shape padding_above{0, 0, 0, 0};
    bool include_padding_in_average = false;

    auto param = make_shared<op::Parameter>(element::f32, arg_shape);
    auto ap = make_shared<op::AvgPool>(param,
                                       window_shape,
                                       window_movement_strides,
                                       padding_below,
                                       padding_above,
                                       include_padding_in_average);

    ASSERT_EQ(ap->get_output_element_type(0), element::f32);
    ASSERT_TRUE(ap->get_output_partial_shape(0).same_scheme(
        PartialShape{5, Dimension::dynamic(), 7, Dimension::dynamic(), 1, 3}));
}

TEST(type_prop, avg_pool_partial_rank_static_dynamic_attrib_rank_mismatch)
{
    PartialShape arg_shape{5, Dimension::dynamic(), 8, Dimension::dynamic(), 4, 7};
    Shape window_shape{2, 3, 4, 5, 6};
    Strides window_movement_strides{1, 1, 1, 1};
    Shape padding_below{0, 0, 0, 0};
    Shape padding_above{0, 0, 0, 0};
    bool include_padding_in_average = false;

    auto param = make_shared<op::Parameter>(element::f32, arg_shape);

    try
    {
        auto ap = make_shared<op::AvgPool>(param,
                                           window_shape,
                                           window_movement_strides,
                                           padding_below,
                                           padding_above,
                                           include_padding_in_average);
        FAIL() << "Mismatch of attribute ranks not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Ranks for data item shape (data batch has shape {5,?,8,?,4,7}, so data "
                        "item rank is 4), padding below (CoordinateDiff{0, 0, 0, 0}), padding "
                        "above (CoordinateDiff{0, 0, 0, 0}), window shape ({2,3,4,5,6}), and "
                        "window strides (Strides{1, 1, 1, 1}) do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, avg_pool_partial_rank_static_dynamic_window_not_too_big)
{
    PartialShape arg_shape{5, Dimension::dynamic(), 8, Dimension::dynamic(), 4, 7};
    Shape window_shape{9, 3, 4, 5};
    Strides window_movement_strides{1, 1, 1, 1};
    Shape padding_below{0, 0, 0, 0};
    Shape padding_above{0, 0, 0, 0};
    bool include_padding_in_average = false;

    auto param = make_shared<op::Parameter>(element::f32, arg_shape);

    try
    {
        auto ap = make_shared<op::AvgPool>(param,
                                           window_shape,
                                           window_movement_strides,
                                           padding_below,
                                           padding_above,
                                           include_padding_in_average);
        FAIL() << "Oversized window not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Window after dilation has dimension (dim: 9) larger than "
                                         "the data shape after padding (dim: 8) at axis 0"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, avg_pool_partial_rank_static_dynamic_padded_window_not_too_big)
{
    PartialShape arg_shape{5, Dimension::dynamic(), 8, Dimension::dynamic(), 4, 7};
    Shape window_shape{9, 3, 4, 5};
    Strides window_movement_strides{1, 1, 1, 1};
    Shape padding_below{0, 0, 0, 0};
    Shape padding_above{1, 0, 0, 0};
    bool include_padding_in_average = false;

    auto param = make_shared<op::Parameter>(element::f32, arg_shape);
    auto ap = make_shared<op::AvgPool>(param,
                                       window_shape,
                                       window_movement_strides,
                                       padding_below,
                                       padding_above,
                                       include_padding_in_average);

    ASSERT_EQ(ap->get_output_element_type(0), element::f32);
    ASSERT_TRUE(ap->get_output_partial_shape(0).same_scheme(
        PartialShape{5, Dimension::dynamic(), 1, Dimension::dynamic(), 1, 3}));
}

TEST(type_prop, avg_pool_partial_rank_static_dynamic_window_in_padding)
{
    PartialShape arg_shape{5, Dimension::dynamic(), 8, Dimension::dynamic(), 4, 7};
    Shape window_shape{9, 3, 4, 3};
    Strides window_movement_strides{1, 1, 1, 1};
    Shape padding_below{0, 0, 0, 4};
    Shape padding_above{0, 0, 0, 0};
    bool include_padding_in_average = false;

    auto param = make_shared<op::Parameter>(element::f32, arg_shape);

    try
    {
        auto ap = make_shared<op::AvgPool>(param,
                                           window_shape,
                                           window_movement_strides,
                                           padding_below,
                                           padding_above,
                                           include_padding_in_average);
        FAIL() << "Window in padding not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Window after dilation has dimension (dim: 9) larger than "
                                         "the data shape after padding (dim: 8) at axis 0"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, pad_deduce_1d_exterior)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{50});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{});
    Shape padding_below{2};
    Shape padding_above{3};
    Shape padding_interior{0};
    auto pad = make_shared<op::Pad>(param0, param1, padding_below, padding_above, padding_interior);
    EXPECT_EQ(pad->get_element_type(), element::f32);
    EXPECT_EQ(pad->get_shape(), (Shape{55}));

    EXPECT_EQ(pad->get_padding_below(), (Shape{2}));
    EXPECT_EQ(pad->get_padding_above(), (Shape{3}));
    EXPECT_EQ(pad->get_padding_interior(), (Shape{0}));
}

TEST(type_prop, pad_deduce_1d_interior)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{50});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{});
    Shape padding_below{0};
    Shape padding_above{0};
    Shape padding_interior{2};
    auto pad = make_shared<op::Pad>(param0, param1, padding_below, padding_above, padding_interior);
    EXPECT_EQ(pad->get_element_type(), element::f32);
    EXPECT_EQ(pad->get_shape(), (Shape{148}));

    EXPECT_EQ(pad->get_padding_below(), (Shape{0}));
    EXPECT_EQ(pad->get_padding_above(), (Shape{0}));
    EXPECT_EQ(pad->get_padding_interior(), (Shape{2}));
}

TEST(type_prop, pad_deduce_1d_interior_exterior)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{50});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{});
    Shape padding_below{5};
    Shape padding_above{6};
    Shape padding_interior{2};
    auto pad = make_shared<op::Pad>(param0, param1, padding_below, padding_above, padding_interior);
    EXPECT_EQ(pad->get_element_type(), element::f32);
    EXPECT_EQ(pad->get_shape(), (Shape{159}));

    EXPECT_EQ(pad->get_padding_below(), (Shape{5}));
    EXPECT_EQ(pad->get_padding_above(), (Shape{6}));
    EXPECT_EQ(pad->get_padding_interior(), (Shape{2}));
}

TEST(type_prop, pad_deduce_2d_interior_exterior)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{50, 40});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{});
    Shape padding_below{5, 3};
    Shape padding_above{6, 9};
    Shape padding_interior{2, 3};
    auto pad = make_shared<op::Pad>(param0, param1, padding_below, padding_above, padding_interior);
    EXPECT_EQ(pad->get_element_type(), element::f32);
    EXPECT_EQ(pad->get_shape(), (Shape{159, 169}));

    EXPECT_EQ(pad->get_padding_below(), (Shape{5, 3}));
    EXPECT_EQ(pad->get_padding_above(), (Shape{6, 9}));
    EXPECT_EQ(pad->get_padding_interior(), (Shape{2, 3}));
}

TEST(type_prop, pad_deduce_3d_interior_exterior)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{50, 40, 20});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{});
    Shape padding_below{5, 3, 0};
    Shape padding_above{6, 9, 4};
    Shape padding_interior{2, 3, 0};
    auto pad = make_shared<op::Pad>(param0, param1, padding_below, padding_above, padding_interior);
    EXPECT_EQ(pad->get_element_type(), element::f32);
    EXPECT_EQ(pad->get_shape(), (Shape{159, 169, 24}));

    EXPECT_EQ(pad->get_padding_below(), (Shape{5, 3, 0}));
    EXPECT_EQ(pad->get_padding_above(), (Shape{6, 9, 4}));
    EXPECT_EQ(pad->get_padding_interior(), (Shape{2, 3, 0}));
}

TEST(type_prop, pad_deduce_element_type_mismatch)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{50, 40, 20});
    auto param1 = make_shared<op::Parameter>(element::i32, Shape{});
    Shape padding_below{5, 3, 0};
    Shape padding_above{6, 9, 4};
    Shape padding_interior{2, 3, 0};
    try
    {
        auto pad =
            make_shared<op::Pad>(param0, param1, padding_below, padding_above, padding_interior);

        // Should have thrown, so fail if it didn't
        FAIL() << "Element tpye mismatch not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Argument element types do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, pad_deduce_nonscalar_pad_value)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{50, 40, 20});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{6});
    Shape padding_below{5, 3, 0};
    Shape padding_above{6, 9, 4};
    Shape padding_interior{2, 3, 0};
    try
    {
        auto pad =
            make_shared<op::Pad>(param0, param1, padding_below, padding_above, padding_interior);

        // Should have thrown, so fail if it didn't
        FAIL() << "Non-scalar pad value not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Argument for padding value is not a scalar"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, pad_deduce_below_padding_wrong_rank)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{50, 40, 20});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{});
    Shape padding_below{5, 3, 0, 6};
    Shape padding_above{6, 9, 4};
    Shape padding_interior{2, 3, 0};
    try
    {
        auto pad =
            make_shared<op::Pad>(param0, param1, padding_below, padding_above, padding_interior);

        // Should have thrown, so fail if it didn't
        FAIL() << "Wrong below-padding rank not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Ranks for padding below (Shape{5, 3, 0, 6}), padding above (Shape{6, 9, "
                        "4}) and interior padding (Shape{2, 3, 0}) do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, pad_deduce_above_padding_wrong_rank)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{50, 40, 20});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{});
    Shape padding_below{5, 3, 0};
    Shape padding_above{6, 9};
    Shape padding_interior{2, 3, 0};
    try
    {
        auto pad =
            make_shared<op::Pad>(param0, param1, padding_below, padding_above, padding_interior);

        // Should have thrown, so fail if it didn't
        FAIL() << "Wrong above-padding rank not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Ranks for padding below (Shape{5, 3, 0}), "
                                         "padding above (Shape{6, 9}) and interior "
                                         "padding (Shape{2, 3, 0}) do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, pad_deduce_interior_padding_wrong_rank)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{50, 40, 20});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{});
    Shape padding_below{5, 3, 0};
    Shape padding_above{6, 9, 4};
    Shape padding_interior{2, 3, 0, 9, 3};
    try
    {
        auto pad =
            make_shared<op::Pad>(param0, param1, padding_below, padding_above, padding_interior);

        // Should have thrown, so fail if it didn't
        FAIL() << "Wrong interior padding rank not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Ranks for padding below (Shape{5, 3, 0}), padding above (Shape{6, 9, 4}) "
                        "and interior padding (Shape{2, 3, 0, 9, 3}) do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, pad_partial_data_rank_dynamic_padding_rank_dynamic_ok)
{
    auto param0 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto param1 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());

    Shape padding_below{2, 4, 6};
    Shape padding_above{8, 2, 3};
    Shape padding_interior{1, 0, 1};

    auto pad = make_shared<op::Pad>(param0, param1, padding_below, padding_above, padding_interior);

    ASSERT_EQ(pad->get_output_element_type(0), element::f32);
    ASSERT_TRUE(pad->get_output_partial_shape(0).same_scheme(
        PartialShape{Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()}));
}

TEST(type_prop, pad_partial_data_rank_dynamic_padding_rank_dynamic_attribs_rank_inconsistent)
{
    auto param0 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto param1 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());

    Shape padding_below{2, 4, 6};
    Shape padding_above{8, 2, 3, 0};
    Shape padding_interior{1, 0, 1};

    try
    {
        auto pad =
            make_shared<op::Pad>(param0, param1, padding_below, padding_above, padding_interior);
        FAIL() << "Inconsistent attribute ranks not detected (rank-dynamic/rank-dynamic arguments)";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Ranks for padding below (Shape{2, 4, 6}), padding above (Shape{8, 2, 3, "
                        "0}) and interior padding (Shape{1, 0, 1}) do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, pad_partial_data_rank_static_dynamic_padding_rank_dynamic_ok)
{
    auto param0 = make_shared<op::Parameter>(
        element::f32,
        PartialShape{Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()});
    auto param1 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());

    Shape padding_below{2, 4, 6};
    Shape padding_above{8, 2, 3};
    Shape padding_interior{1, 0, 1};

    auto pad = make_shared<op::Pad>(param0, param1, padding_below, padding_above, padding_interior);

    ASSERT_EQ(pad->get_output_element_type(0), element::f32);
    ASSERT_TRUE(pad->get_output_partial_shape(0).same_scheme(
        PartialShape{Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()}));
}

TEST(type_prop, pad_partial_data_rank_static_dynamic_some_dims_known_padding_rank_dynamic_ok)
{
    auto param0 =
        make_shared<op::Parameter>(element::f32, PartialShape{3, 5, Dimension::dynamic()});
    auto param1 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());

    Shape padding_below{2, 4, 6};
    Shape padding_above{8, 2, 3};
    Shape padding_interior{1, 0, 1};

    auto pad = make_shared<op::Pad>(param0, param1, padding_below, padding_above, padding_interior);

    ASSERT_EQ(pad->get_output_element_type(0), element::f32);
    ASSERT_TRUE(
        pad->get_output_partial_shape(0).same_scheme(PartialShape{15, 11, Dimension::dynamic()}));
}

TEST(type_prop, pad_partial_data_rank_dynamic_padding_static_ok)
{
    auto param0 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{});

    Shape padding_below{2, 4, 6};
    Shape padding_above{8, 2, 3};
    Shape padding_interior{1, 0, 1};

    auto pad = make_shared<op::Pad>(param0, param1, padding_below, padding_above, padding_interior);

    ASSERT_EQ(pad->get_output_element_type(0), element::f32);
    ASSERT_TRUE(pad->get_output_partial_shape(0).same_scheme(
        PartialShape{Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()}));
}

TEST(type_prop, pad_partial_data_rank_dynamic_padding_static_wrong_padding_rank)
{
    auto param0 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{2, 3, 8});

    Shape padding_below{2, 4, 6};
    Shape padding_above{8, 2, 3};
    Shape padding_interior{1, 0, 1};

    try
    {
        auto pad =
            make_shared<op::Pad>(param0, param1, padding_below, padding_above, padding_interior);
        FAIL() << "Wrong padding rank not detected (rank-dynamic/static arguments)";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Argument for padding value is not a scalar (shape: {2,3,8})"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, pad_partial_data_rank_dynamic_padding_static_attribs_rank_inconsistent)
{
    auto param0 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{});

    Shape padding_below{2, 4, 6};
    Shape padding_above{8, 2, 3, 4};
    Shape padding_interior{1, 0, 1};

    try
    {
        auto pad =
            make_shared<op::Pad>(param0, param1, padding_below, padding_above, padding_interior);
        FAIL() << "Wrong padding rank not detected (rank-dynamic/static arguments)";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Ranks for padding below (Shape{2, 4, 6}), padding above (Shape{8, 2, 3, "
                        "4}) and interior padding (Shape{1, 0, 1}) do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, sum_deduce)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{2, 4});

    auto r0 = make_shared<op::Sum>(param_0, AxisSet{0});
    ASSERT_EQ(r0->get_element_type(), element::f32);
    ASSERT_EQ(r0->get_shape(), (Shape{4}));

    auto r1 = make_shared<op::Sum>(param_0, AxisSet{1});
    ASSERT_EQ(r1->get_element_type(), element::f32);
    ASSERT_EQ(r1->get_shape(), (Shape{2}));

    auto r01 = make_shared<op::Sum>(param_0, AxisSet{0, 1});
    ASSERT_EQ(r01->get_element_type(), element::f32);
    ASSERT_EQ(r01->get_shape(), (Shape{}));

    auto r_none = make_shared<op::Sum>(param_0, AxisSet{});
    ASSERT_EQ(r_none->get_element_type(), element::f32);
    ASSERT_EQ(r_none->get_shape(), (Shape{2, 4}));
}

TEST(type_prop, sum_axis_oob)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{2, 4});

    try
    {
        auto r = make_shared<op::Sum>(param_0, AxisSet{0, 2, 1});
        // Should have thrown, so fail if it didn't
        FAIL() << "Did not detect out-of-bound axis for sum";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Reduction axis (2) is out of bounds"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, sum_partial_rank_dynamic)
{
    auto param = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto summation_axes = AxisSet{2385, 0, 4404}; // arbitrary
    auto sum = make_shared<op::Sum>(param, summation_axes);

    EXPECT_EQ(sum->get_output_element_type(0), element::f32);
    EXPECT_TRUE(sum->get_output_partial_shape(0).is_dynamic());
}

TEST(type_prop, sum_partial_rank_static_dynamic_ok_result_static)
{
    auto param =
        make_shared<op::Parameter>(element::f32, PartialShape{1, 2, Dimension::dynamic(), 4, 5});
    auto summation_axes = AxisSet{2, 3};
    auto sum = make_shared<op::Sum>(param, summation_axes);

    EXPECT_EQ(sum->get_output_element_type(0), element::f32);
    EXPECT_EQ(sum->get_shape(), (Shape{1, 2, 5}));
}

TEST(type_prop, sum_partial_rank_static_dynamic_ok_result_dynamic)
{
    auto param = make_shared<op::Parameter>(
        element::f32, PartialShape{1, 2, Dimension::dynamic(), 4, Dimension::dynamic()});
    auto summation_axes = AxisSet{2, 3};
    auto sum = make_shared<op::Sum>(param, summation_axes);

    EXPECT_EQ(sum->get_output_element_type(0), element::f32);
    EXPECT_TRUE(
        sum->get_output_partial_shape(0).same_scheme(PartialShape{1, 2, Dimension::dynamic()}));
}

TEST(type_prop, sum_partial_rank_static_dynamic_axes_oob)
{
    auto param = make_shared<op::Parameter>(
        element::f32, PartialShape{1, 2, Dimension::dynamic(), 4, Dimension::dynamic()});
    auto summation_axes = AxisSet{2, 5, 1};

    try
    {
        auto sum = make_shared<op::Sum>(param, summation_axes);
        // Should have thrown, so fail if it didn't
        FAIL() << "Did not detect out-of-bound axis for sum (rank-static dynamic input)";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Reduction axis (5) is out of bounds"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, index_reduction_scalar)
{
    auto a = make_shared<op::Parameter>(element::f32, Shape{});

    try
    {
        auto argmin = make_shared<op::ArgMin>(a, 0, element::i32);
        FAIL() << "ArgMin c-tor should throw for scalar shapes";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Argument rank is zero");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, index_reduction_invalid_rank)
{
    auto a = make_shared<op::Parameter>(element::f32, Shape{2, 2});

    try
    {
        auto argmin = make_shared<op::ArgMin>(a, 2, element::i32);
        FAIL() << "ArgMin c-tor should throw for axis out of bounds";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Reduction axis (2) is not less than argument rank (2)");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, index_reduction_invalid_index_type)
{
    auto a = make_shared<op::Parameter>(element::f32, Shape{2, 2});

    try
    {
        auto argmin = make_shared<op::ArgMin>(a, 1, element::f32);
        FAIL() << "ArgMin c-tor should throw for invalid index type";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Index element is neither i64 or i32");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, index_reduction_partial_rank_dynamic_output_et_dynamic)
{
    auto a = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    size_t axis = 228;
    auto output_et = element::dynamic;

    try
    {
        auto argmax = make_shared<op::ArgMax>(a, axis, output_et);
        FAIL() << "Invalid output type of element::dynamic not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Index element is neither i64 or i32");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, index_reduction_partial_rank_dynamic_output_et_invalid)
{
    auto a = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    size_t axis = 228;
    auto output_et = element::dynamic;

    try
    {
        auto argmax = make_shared<op::ArgMax>(a, axis, output_et);
        FAIL() << "Invalid output type of element::f32 not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Index element is neither i64 or i32");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, index_reduction_partial_rank_dynamic_ok)
{
    auto a = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    size_t axis = 228;
    auto output_et = element::i32;

    auto argmax = make_shared<op::ArgMax>(a, axis, output_et);

    ASSERT_EQ(argmax->get_output_element_type(0), element::i32);
    ASSERT_TRUE(argmax->get_output_partial_shape(0).rank().is_dynamic());
}

TEST(type_prop, index_reduction_partial_rank_static_dynamic_axis_oob)
{
    auto a = make_shared<op::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 2, 3, 4});
    size_t axis = 4;
    auto output_et = element::i32;

    try
    {
        auto argmax = make_shared<op::ArgMax>(a, axis, output_et);
        FAIL() << "Out-of-bounds reduction axis not detected (rank-static dynamic argument)";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Reduction axis (4) is not less than argument rank (4)");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, index_reduction_partial_rank_static_dynamic_ok)
{
    auto a = make_shared<op::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 2, 3, 4});
    size_t axis = 2;
    auto output_et = element::i32;

    auto argmax = make_shared<op::ArgMax>(a, axis, output_et);

    ASSERT_EQ(argmax->get_output_element_type(0), element::i32);
    ASSERT_TRUE(
        argmax->get_output_partial_shape(0).same_scheme(PartialShape{Dimension::dynamic(), 2, 4}));
}

TEST(type_prop, index_reduction_partial_et_dynamic_rank_static_dynamic_ok)
{
    auto a =
        make_shared<op::Parameter>(element::dynamic, PartialShape{Dimension::dynamic(), 2, 3, 4});
    size_t axis = 2;
    auto output_et = element::i32;

    auto argmax = make_shared<op::ArgMax>(a, axis, output_et);

    ASSERT_EQ(argmax->get_output_element_type(0), element::i32);
    ASSERT_TRUE(
        argmax->get_output_partial_shape(0).same_scheme(PartialShape{Dimension::dynamic(), 2, 4}));
}

TEST(type_prop, topk_invalid_rank)
{
    auto a = make_shared<op::Parameter>(element::f32, Shape{});

    try
    {
        auto topk = make_shared<op::TopK>(a, 0, element::i32, 1, true);
        FAIL() << "TopK c-tor should throw for scalar shapes";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Argument rank must be greater than 0");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, topk_invalid_top_k)
{
    auto a = make_shared<op::Parameter>(element::f32, Shape{2, 2});

    try
    {
        auto topk = make_shared<op::TopK>(a, 2, element::i32, 1, true);
        FAIL() << "TopK c-tor should throw for invalid top k axis";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "TopK axis (2) is out of bounds");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, topk_invalid_index_type)
{
    auto a = make_shared<op::Parameter>(element::f32, Shape{2, 2});

    try
    {
        auto topk = make_shared<op::TopK>(a, 0, element::f32, 1, true);
        FAIL() << "TopK c-tor should throw for invalid index element type";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            "Argument element type must be i64 or i32 (got element::Type{32, 1, 1, 0, \"float\"})");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, topk_invalid_k)
{
    auto a = make_shared<op::Parameter>(element::f32, Shape{2, 2});

    try
    {
        auto topk = make_shared<op::TopK>(a, 0, element::i32, 3, true);
        FAIL() << "TopK c-tor should throw for invalid K";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "K (3) exceeds the dimension (2) of the TopK axis (axis 0)");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, topk_rank_dynamic_ok)
{
    element::Type arg_et{element::f32};
    PartialShape arg_shape{PartialShape::dynamic()};
    size_t top_k_axis = 22;
    size_t k = 900;
    element::Type result_et{element::i32};
    bool compute_max = true;

    auto param = make_shared<op::Parameter>(arg_et, arg_shape);

    auto topk = make_shared<op::TopK>(param, top_k_axis, result_et, k, compute_max);

    ASSERT_TRUE(topk->get_output_element_type(0) == element::i32);
    ASSERT_TRUE(topk->get_output_element_type(1) == element::f32);
    ASSERT_TRUE(topk->get_output_partial_shape(0).rank().is_dynamic());
    ASSERT_TRUE(topk->get_output_partial_shape(1).rank().is_dynamic());
}

TEST(type_prop, topk_rank_dynamic_result_et_dynamic)
{
    element::Type arg_et{element::f32};
    PartialShape arg_shape{PartialShape::dynamic()};
    size_t top_k_axis = 22;
    size_t k = 900;
    element::Type result_et{element::dynamic};
    bool compute_max = true;

    auto param = make_shared<op::Parameter>(arg_et, arg_shape);

    try
    {
        auto topk = make_shared<op::TopK>(param, top_k_axis, result_et, k, compute_max);
        FAIL() << "Dynamic result element type not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Argument element type must not be dynamic");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, topk_rank_dynamic_result_et_invalid)
{
    element::Type arg_et{element::f32};
    PartialShape arg_shape{PartialShape::dynamic()};
    size_t top_k_axis = 22;
    size_t k = 900;
    element::Type result_et{element::f32};
    bool compute_max = true;

    auto param = make_shared<op::Parameter>(arg_et, arg_shape);

    try
    {
        auto topk = make_shared<op::TopK>(param, top_k_axis, result_et, k, compute_max);
        FAIL() << "Invalid result element type not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            "Argument element type must be i64 or i32 (got element::Type{32, 1, 1, 0, \"float\"})");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, topk_rank_static_dynamic_k_known_topk_dim_dynamic_ok)
{
    element::Type arg_et{element::f32};
    PartialShape arg_shape{Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()};
    size_t top_k_axis = 1;
    size_t k = 999;
    element::Type result_et{element::i32};
    bool compute_max = true;

    auto param = make_shared<op::Parameter>(arg_et, arg_shape);

    auto topk = make_shared<op::TopK>(param, top_k_axis, result_et, k, compute_max);

    ASSERT_TRUE(topk->get_output_element_type(0) == element::i32);
    ASSERT_TRUE(topk->get_output_element_type(1) == element::f32);
    ASSERT_TRUE(topk->get_output_partial_shape(0).same_scheme(
        PartialShape{Dimension::dynamic(), 999, Dimension::dynamic()}));
    ASSERT_TRUE(topk->get_output_partial_shape(1).same_scheme(
        PartialShape{Dimension::dynamic(), 999, Dimension::dynamic()}));
}

TEST(type_prop, topk_rank_static_dynamic_k_unknown_topk_dim_dynamic_ok)
{
    element::Type arg_et{element::f32};
    PartialShape arg_shape{Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()};
    size_t top_k_axis = 1;
    size_t k = 0;
    element::Type result_et{element::i32};
    bool compute_max = true;

    auto param = make_shared<op::Parameter>(arg_et, arg_shape);

    auto topk = make_shared<op::TopK>(param, top_k_axis, result_et, k, compute_max);

    ASSERT_TRUE(topk->get_output_element_type(0) == element::i32);
    ASSERT_TRUE(topk->get_output_element_type(1) == element::f32);
    ASSERT_TRUE(topk->get_output_partial_shape(0).same_scheme(
        PartialShape{Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()}));
    ASSERT_TRUE(topk->get_output_partial_shape(1).same_scheme(
        PartialShape{Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()}));
}

TEST(type_prop, topk_rank_static_dynamic_axis_oob)
{
    element::Type arg_et{element::f32};
    PartialShape arg_shape{Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()};
    size_t top_k_axis = 22;
    size_t k = 900;
    element::Type result_et{element::f32};
    bool compute_max = true;

    auto param = make_shared<op::Parameter>(arg_et, arg_shape);

    try
    {
        auto topk = make_shared<op::TopK>(param, top_k_axis, result_et, k, compute_max);
        FAIL() << "TopK axis out-of-bounds not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            "Argument element type must be i64 or i32 (got element::Type{32, 1, 1, 0, \"float\"})");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, topk_rank_static_dynamic_k_unknown_axis_oob)
{
    element::Type arg_et{element::f32};
    PartialShape arg_shape{Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()};
    size_t top_k_axis = 22;
    size_t k = 0;
    element::Type result_et{element::f32};
    bool compute_max = true;

    auto param = make_shared<op::Parameter>(arg_et, arg_shape);

    try
    {
        auto topk = make_shared<op::TopK>(param, top_k_axis, result_et, k, compute_max);
        FAIL() << "TopK axis out-of-bounds not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            "Argument element type must be i64 or i32 (got element::Type{32, 1, 1, 0, \"float\"})");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, topk_rank_static_dynamic_k_known_too_big)
{
    element::Type arg_et{element::f32};
    PartialShape arg_shape{Dimension::dynamic(), 3, Dimension::dynamic()};
    size_t top_k_axis = 1;
    size_t k = 4;
    element::Type result_et{element::f32};
    bool compute_max = true;

    auto param = make_shared<op::Parameter>(arg_et, arg_shape);

    try
    {
        auto topk = make_shared<op::TopK>(param, top_k_axis, result_et, k, compute_max);
        FAIL() << "Oversize K not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            "Argument element type must be i64 or i32 (got element::Type{32, 1, 1, 0, \"float\"})");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, topk_rank_static_dynamic_k_unknown_ok)
{
    element::Type arg_et{element::f32};
    PartialShape arg_shape{Dimension::dynamic(), 3, Dimension::dynamic()};
    size_t top_k_axis = 1;
    size_t k = 0;
    element::Type result_et{element::i32};
    bool compute_max = true;

    auto param = make_shared<op::Parameter>(arg_et, arg_shape);

    auto topk = make_shared<op::TopK>(param, top_k_axis, result_et, k, compute_max);

    ASSERT_TRUE(topk->get_output_element_type(0) == element::i32);
    ASSERT_TRUE(topk->get_output_element_type(1) == element::f32);
    ASSERT_TRUE(topk->get_output_partial_shape(0).same_scheme(
        PartialShape{Dimension::dynamic(), 3, Dimension::dynamic()}));
    ASSERT_TRUE(topk->get_output_partial_shape(1).same_scheme(
        PartialShape{Dimension::dynamic(), 3, Dimension::dynamic()}));
}

TEST(type_prop, topk_rank_static_dynamic_k_known_ok)
{
    element::Type arg_et{element::f32};
    PartialShape arg_shape{Dimension::dynamic(), 3, Dimension::dynamic()};
    size_t top_k_axis = 1;
    size_t k = 2;
    element::Type result_et{element::i32};
    bool compute_max = true;

    auto param = make_shared<op::Parameter>(arg_et, arg_shape);

    auto topk = make_shared<op::TopK>(param, top_k_axis, result_et, k, compute_max);

    ASSERT_TRUE(topk->get_output_element_type(0) == element::i32);
    ASSERT_TRUE(topk->get_output_element_type(1) == element::f32);
    ASSERT_TRUE(topk->get_output_partial_shape(0).same_scheme(
        PartialShape{Dimension::dynamic(), 2, Dimension::dynamic()}));
    ASSERT_TRUE(topk->get_output_partial_shape(1).same_scheme(
        PartialShape{Dimension::dynamic(), 2, Dimension::dynamic()}));
}

TEST(type_prop, param_partial_rank_dynamic)
{
    auto a = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());

    auto& pshape = a->get_output_partial_shape(0);

    ASSERT_TRUE(pshape.is_dynamic());
    ASSERT_TRUE(pshape.rank().is_dynamic());
}

TEST(type_prop, param_partial_rank_static)
{
    auto a = make_shared<op::Parameter>(element::f32, PartialShape{2, Dimension::dynamic(), 3, 4});

    auto& pshape = a->get_output_partial_shape(0);

    ASSERT_TRUE(pshape.is_dynamic());
    ASSERT_EQ(size_t(pshape.rank()), 4);
    ASSERT_TRUE(pshape[0].is_static() && size_t(pshape[0]) == 2);
    ASSERT_TRUE(pshape[1].is_dynamic());
    ASSERT_TRUE(pshape[2].is_static() && size_t(pshape[2]) == 3);
    ASSERT_TRUE(pshape[3].is_static() && size_t(pshape[3]) == 4);
}

TEST(type_prop, binary_elementwise_arithmetic_both_dynamic)
{
    auto a = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto b = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto add = make_shared<op::Add>(a, b);

    ASSERT_TRUE(add->get_output_partial_shape(0).rank().is_dynamic());
}

TEST(type_prop, binary_elementwise_arithmetic_left_rank_dynamic_right_static)
{
    auto a = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto b = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3});
    auto add = make_shared<op::Add>(a, b);

    ASSERT_TRUE(add->get_output_partial_shape(0).is_static());
    ASSERT_EQ(add->get_shape(), (Shape{1, 2, 3}));
}

TEST(type_prop, binary_elementwise_arithmetic_left_static_right_rank_dynamic)
{
    auto a = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3});
    auto b = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto add = make_shared<op::Add>(a, b);

    ASSERT_TRUE(add->get_output_partial_shape(0).is_static());
    ASSERT_EQ(add->get_shape(), (Shape{1, 2, 3}));
}

TEST(type_prop, binary_elementwise_arithmetic_left_rank_static_dynamic_right_rank_dynamic)
{
    auto a = make_shared<op::Parameter>(element::f32, PartialShape{1, Dimension::dynamic(), 3});
    auto b = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto add = make_shared<op::Add>(a, b);

    ASSERT_TRUE(add->get_output_partial_shape(0).rank().is_static());
    ASSERT_TRUE(add->get_output_partial_shape(0).is_dynamic());
    ASSERT_TRUE(
        add->get_output_partial_shape(0).same_scheme(PartialShape{1, Dimension::dynamic(), 3}));
}

TEST(type_prop, binary_elementwise_arithmetic_left_rank_dynamic_right_rank_static_dynamic)
{
    auto a = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto b = make_shared<op::Parameter>(element::f32, PartialShape{1, Dimension::dynamic(), 3});
    auto add = make_shared<op::Add>(a, b);

    ASSERT_TRUE(add->get_output_partial_shape(0).rank().is_static());
    ASSERT_TRUE(add->get_output_partial_shape(0).is_dynamic());
    ASSERT_TRUE(
        add->get_output_partial_shape(0).same_scheme(PartialShape{1, Dimension::dynamic(), 3}));
}

TEST(type_prop,
     binary_elementwise_arithmetic_left_rank_static_dynamic_right_rank_static_dynamic_result_static)
{
    auto a = make_shared<op::Parameter>(element::f32, PartialShape{1, Dimension::dynamic(), 3});
    auto b = make_shared<op::Parameter>(element::f32, PartialShape{1, 2, Dimension::dynamic()});
    auto add = make_shared<op::Add>(a, b);

    ASSERT_TRUE(add->get_output_partial_shape(0).is_static());
    ASSERT_EQ(add->get_shape(), (Shape{1, 2, 3}));
}

TEST(
    type_prop,
    binary_elementwise_arithmetic_left_rank_static_dynamic_right_rank_static_dynamic_result_rank_static_dynamic)
{
    auto a = make_shared<op::Parameter>(
        element::f32, PartialShape{1, Dimension::dynamic(), Dimension::dynamic()});
    auto b = make_shared<op::Parameter>(element::f32, PartialShape{1, 2, Dimension::dynamic()});
    auto add = make_shared<op::Add>(a, b);

    ASSERT_TRUE(add->get_output_partial_shape(0).rank().is_static());
    ASSERT_TRUE(add->get_output_partial_shape(0).is_dynamic());
    ASSERT_TRUE(
        add->get_output_partial_shape(0).same_scheme(PartialShape{1, 2, Dimension::dynamic()}));
}

TEST(type_prop, binary_elementwise_arithmetic_left_static_right_rank_static_dynamic)
{
    auto a = make_shared<op::Parameter>(element::f32, PartialShape{1, 2, 3});
    auto b = make_shared<op::Parameter>(element::f32, PartialShape{1, 2, Dimension::dynamic()});
    auto add = make_shared<op::Add>(a, b);

    ASSERT_TRUE(add->get_output_partial_shape(0).is_static());
    ASSERT_EQ(add->get_shape(), (Shape{1, 2, 3}));
}

TEST(type_prop, binary_elementwise_arithmetic_left_rank_static_dynamic_right_static)
{
    auto a = make_shared<op::Parameter>(element::f32, PartialShape{1, 2, Dimension::dynamic()});
    auto b = make_shared<op::Parameter>(element::f32, PartialShape{1, 2, 3});
    auto add = make_shared<op::Add>(a, b);

    ASSERT_TRUE(add->get_output_partial_shape(0).is_static());
    ASSERT_EQ(add->get_shape(), (Shape{1, 2, 3}));
}

TEST(type_prop, binary_elementwise_arithmetic_left_rank_static_dynamic_inconsistent)
{
    auto a = make_shared<op::Parameter>(element::f32, PartialShape{1, 2, Dimension::dynamic()});
    auto b = make_shared<op::Parameter>(element::f32, PartialShape{1, 3, 3});

    try
    {
        auto add = make_shared<op::Add>(a, b);
        FAIL() << "Inconsistent partial shapes not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Argument shapes are inconsistent");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, binary_elementwise_arithmetic_right_rank_static_dynamic_inconsistent)
{
    auto a = make_shared<op::Parameter>(element::f32, PartialShape{1, 3, 3});
    auto b = make_shared<op::Parameter>(element::f32, PartialShape{1, 2, Dimension::dynamic()});

    try
    {
        auto add = make_shared<op::Add>(a, b);
        FAIL() << "Inconsistent partial shapes not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Argument shapes are inconsistent");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, binary_elementwise_arithmetic_both_rank_static_dynamic_inconsistent)
{
    auto a = make_shared<op::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 3, 3});
    auto b = make_shared<op::Parameter>(element::f32, PartialShape{1, 2, Dimension::dynamic()});

    try
    {
        auto add = make_shared<op::Add>(a, b);
        FAIL() << "Inconsistent partial shapes not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Argument shapes are inconsistent");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, binary_elementwise_arithmetic_left_rank_static_dynamic_different_rank)
{
    auto a = make_shared<op::Parameter>(element::f32, PartialShape{1, 2, Dimension::dynamic()});
    auto b = make_shared<op::Parameter>(element::f32, PartialShape{1, 2, 3, 4});

    try
    {
        auto add = make_shared<op::Add>(a, b);
        FAIL() << "Inconsistent partial shapes not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Argument shapes are inconsistent");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, binary_elementwise_arithmetic_right_rank_static_dynamic_different_rank)
{
    auto a = make_shared<op::Parameter>(element::f32, PartialShape{1, 2, 3, 4});
    auto b = make_shared<op::Parameter>(element::f32, PartialShape{1, 2, Dimension::dynamic()});

    try
    {
        auto add = make_shared<op::Add>(a, b);
        FAIL() << "Inconsistent partial shapes not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Argument shapes are inconsistent");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, binary_elementwise_arithmetic_both_rank_static_dynamic_different_rank)
{
    auto a = make_shared<op::Parameter>(element::f32, PartialShape{1, Dimension::dynamic(), 3, 4});
    auto b = make_shared<op::Parameter>(element::f32, PartialShape{1, 2, Dimension::dynamic()});

    try
    {
        auto add = make_shared<op::Add>(a, b);
        FAIL() << "Inconsistent partial shapes not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Argument shapes are inconsistent");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, binary_elementwise_arithmetic_both_et_dynamic)
{
    auto a = make_shared<op::Parameter>(element::dynamic, Shape{1, 2, 3, 4});
    auto b = make_shared<op::Parameter>(element::dynamic, Shape{1, 2, 3, 4});
    auto add = make_shared<op::Add>(a, b);

    ASSERT_TRUE(add->get_output_element_type(0).is_dynamic());
}

TEST(type_prop, binary_elementwise_arithmetic_left_et_dynamic)
{
    auto a = make_shared<op::Parameter>(element::dynamic, Shape{1, 2, 3, 4});
    auto b = make_shared<op::Parameter>(element::u32, Shape{1, 2, 3, 4});
    auto add = make_shared<op::Add>(a, b);

    ASSERT_EQ(add->get_output_element_type(0), element::u32);
}

TEST(type_prop, binary_elementwise_arithmetic_right_et_dynamic)
{
    auto a = make_shared<op::Parameter>(element::i64, Shape{1, 2, 3, 4});
    auto b = make_shared<op::Parameter>(element::dynamic, Shape{1, 2, 3, 4});
    auto add = make_shared<op::Add>(a, b);

    ASSERT_EQ(add->get_output_element_type(0), element::i64);
}

TEST(type_prop, logic_arith_compare_partial_et)
{
    auto test_logic = [](element::Type et0, element::Type et1) -> std::shared_ptr<Node> {
        auto param0 = std::make_shared<op::Parameter>(et0, Shape{1, 2, 3});
        auto param1 = std::make_shared<op::Parameter>(et1, Shape{1, 2, 3});
        return std::make_shared<op::And>(param0, param1);
    };

    auto test_arith = [](element::Type et0, element::Type et1) -> std::shared_ptr<Node> {
        auto param0 = std::make_shared<op::Parameter>(et0, Shape{1, 2, 3});
        auto param1 = std::make_shared<op::Parameter>(et1, Shape{1, 2, 3});
        return std::make_shared<op::Add>(param0, param1);
    };

    auto test_compare = [](element::Type et0, element::Type et1) -> std::shared_ptr<Node> {
        auto param0 = std::make_shared<op::Parameter>(et0, Shape{1, 2, 3});
        auto param1 = std::make_shared<op::Parameter>(et1, Shape{1, 2, 3});
        return std::make_shared<op::Greater>(param0, param1);
    };

    auto test_not = [](element::Type et) -> std::shared_ptr<Node> {
        auto param = std::make_shared<op::Parameter>(et, Shape{1, 2, 3});
        return std::make_shared<op::Not>(param);
    };

    // Logical ops:
    //
    // int int -> !
    // int boo -> !
    // int dyn -> !
    // boo int -> !
    // boo boo -> boo
    // boo dyn -> boo
    // dyn int -> !
    // dyn boo -> boo
    // dyn dyn -> boo
    ASSERT_ANY_THROW({ test_logic(element::i32, element::i32); });
    ASSERT_ANY_THROW({ test_logic(element::i32, element::boolean); });
    ASSERT_ANY_THROW({ test_logic(element::i32, element::dynamic); });
    ASSERT_ANY_THROW({ test_logic(element::boolean, element::i32); });
    ASSERT_EQ(test_logic(element::boolean, element::boolean)->get_element_type(), element::boolean);
    ASSERT_EQ(test_logic(element::boolean, element::dynamic)->get_element_type(), element::boolean);
    ASSERT_ANY_THROW({ test_logic(element::dynamic, element::i32); });
    ASSERT_EQ(test_logic(element::dynamic, element::boolean)->get_element_type(), element::boolean);
    ASSERT_EQ(test_logic(element::dynamic, element::dynamic)->get_element_type(), element::boolean);

    // Arith ops:
    //
    // int int -> int
    // int boo -> !
    // int dyn -> int
    // boo int -> !
    // boo boo -> !
    // boo dyn -> !
    // dyn int -> int
    // dyn boo -> !
    // dyn dyn -> dyn
    ASSERT_EQ(test_arith(element::i32, element::i32)->get_element_type(), element::i32);
    ASSERT_ANY_THROW({ test_arith(element::i32, element::boolean); });
    ASSERT_EQ(test_arith(element::i32, element::dynamic)->get_element_type(), element::i32);
    ASSERT_ANY_THROW({ test_arith(element::boolean, element::i32); });
    ASSERT_ANY_THROW({ test_arith(element::boolean, element::boolean); });
    ASSERT_ANY_THROW({ test_arith(element::boolean, element::dynamic); });
    ASSERT_EQ(test_arith(element::dynamic, element::i32)->get_element_type(), element::i32);
    ASSERT_ANY_THROW({ test_arith(element::dynamic, element::boolean); });
    ASSERT_EQ(test_arith(element::dynamic, element::dynamic)->get_element_type(), element::dynamic);

    // Comparison ops:
    //
    // int int -> boo
    // int boo -> !
    // int dyn -> boo
    // boo int -> !
    // boo boo -> boo
    // boo dyn -> boo
    // dyn int -> boo
    // dyn boo -> boo
    // dyn dyn -> boo
    ASSERT_EQ(test_compare(element::i32, element::i32)->get_element_type(), element::boolean);
    ASSERT_ANY_THROW({ test_compare(element::i32, element::boolean); });
    ASSERT_EQ(test_compare(element::i32, element::dynamic)->get_element_type(), element::boolean);
    ASSERT_ANY_THROW({ test_compare(element::boolean, element::i32); });
    ASSERT_EQ(test_compare(element::boolean, element::boolean)->get_element_type(),
              element::boolean);
    ASSERT_EQ(test_compare(element::boolean, element::dynamic)->get_element_type(),
              element::boolean);
    ASSERT_EQ(test_compare(element::dynamic, element::i32)->get_element_type(), element::boolean);
    ASSERT_EQ(test_compare(element::dynamic, element::boolean)->get_element_type(),
              element::boolean);
    ASSERT_EQ(test_compare(element::dynamic, element::dynamic)->get_element_type(),
              element::boolean);

    // Logical negation op:
    //
    // Current behavior:
    // int -> int
    // boo -> boo
    // dyn -> dyn
    //
    // TODO(amprocte): I believe the behavior should actually be:
    // int -> !
    // boo -> boo
    // dyn -> boo
    ASSERT_EQ(test_not(element::i32)->get_element_type(), element::i32);
    ASSERT_EQ(test_not(element::boolean)->get_element_type(), element::boolean);
    ASSERT_EQ(test_not(element::dynamic)->get_element_type(), element::dynamic);
}

TEST(type_prop, get_output_element_partial_et_dynamic)
{
    auto a = make_shared<op::Parameter>(element::dynamic, Shape{1, 2, 3, 4});
    auto b = make_shared<op::Parameter>(element::dynamic, Shape{1, 2, 3, 4});
    auto add = make_shared<op::Add>(a, b);
    auto goe = make_shared<op::GetOutputElement>(add, 0);

    ASSERT_EQ(goe->get_output_element_type(0), element::dynamic);
    ASSERT_EQ(goe->get_output_shape(0), (Shape{1, 2, 3, 4}));
}

TEST(type_prop, get_output_element_partial_rank_dynamic)
{
    auto a = make_shared<op::Parameter>(element::i32, PartialShape::dynamic());
    auto b = make_shared<op::Parameter>(element::i32, PartialShape::dynamic());
    auto add = make_shared<op::Add>(a, b);
    auto goe = make_shared<op::GetOutputElement>(add, 0);

    ASSERT_EQ(goe->get_output_element_type(0), element::i32);
    ASSERT_TRUE(goe->get_output_partial_shape(0).rank().is_dynamic());
}

TEST(type_prop, get_output_element_partial_rank_static_dynamic)
{
    auto a = make_shared<op::Parameter>(
        element::i32, PartialShape{Dimension::dynamic(), 2, 3, Dimension::dynamic()});
    auto b = make_shared<op::Parameter>(
        element::i32, PartialShape{Dimension::dynamic(), 2, Dimension::dynamic(), 4});
    auto add = make_shared<op::Add>(a, b);
    auto goe = make_shared<op::GetOutputElement>(add, 0);

    ASSERT_EQ(goe->get_output_element_type(0), element::i32);
    ASSERT_TRUE(
        goe->get_output_partial_shape(0).same_scheme(PartialShape{Dimension::dynamic(), 2, 3, 4}));
}

TEST(type_prop, quantize_f32_to_i8_nchw_per_channel_ok)
{
    Shape batch_shape{64, 3, 480, 640};
    Shape scale_shape{3};
    Shape offset_shape{3};
    element::Type unquantized_type = element::f32;
    element::Type quantized_type = element::i8;
    element::Type batch_type = unquantized_type;
    element::Type scale_type = unquantized_type;
    element::Type offset_type = quantized_type;
    AxisSet axes{1};
    auto round_mode = op::Quantize::RoundMode::HALF_AWAY_FROM_ZERO;

    auto batch = make_shared<op::Parameter>(batch_type, batch_shape);
    auto scale = make_shared<op::Parameter>(scale_type, scale_shape);
    auto offset = make_shared<op::Parameter>(offset_type, offset_shape);
    auto quant = make_shared<op::Quantize>(batch, scale, offset, quantized_type, axes, round_mode);

    ASSERT_EQ(quant->get_output_element_type(0), quantized_type);
    ASSERT_EQ(quant->get_output_shape(0), batch_shape);
}

TEST(type_prop, quantize_f32_to_i8_nchw_per_image_ok)
{
    Shape batch_shape{64, 3, 480, 640};
    Shape scale_shape{64};
    Shape offset_shape{64};
    element::Type unquantized_type = element::f32;
    element::Type quantized_type = element::i8;
    element::Type batch_type = unquantized_type;
    element::Type scale_type = unquantized_type;
    element::Type offset_type = quantized_type;
    AxisSet axes{0};
    auto round_mode = op::Quantize::RoundMode::HALF_AWAY_FROM_ZERO;

    auto batch = make_shared<op::Parameter>(batch_type, batch_shape);
    auto scale = make_shared<op::Parameter>(scale_type, scale_shape);
    auto offset = make_shared<op::Parameter>(offset_type, offset_shape);
    auto quant = make_shared<op::Quantize>(batch, scale, offset, quantized_type, axes, round_mode);

    ASSERT_EQ(quant->get_output_element_type(0), quantized_type);
    ASSERT_EQ(quant->get_output_shape(0), batch_shape);
}

TEST(type_prop, quantize_f32_to_i8_nchw_per_row_ok)
{
    Shape batch_shape{64, 3, 480, 640};
    Shape scale_shape{480};
    Shape offset_shape{480};
    element::Type unquantized_type = element::f32;
    element::Type quantized_type = element::i8;
    element::Type batch_type = unquantized_type;
    element::Type scale_type = unquantized_type;
    element::Type offset_type = quantized_type;
    AxisSet axes{2};
    auto round_mode = op::Quantize::RoundMode::HALF_AWAY_FROM_ZERO;

    auto batch = make_shared<op::Parameter>(batch_type, batch_shape);
    auto scale = make_shared<op::Parameter>(scale_type, scale_shape);
    auto offset = make_shared<op::Parameter>(offset_type, offset_shape);
    auto quant = make_shared<op::Quantize>(batch, scale, offset, quantized_type, axes, round_mode);

    ASSERT_EQ(quant->get_output_element_type(0), quantized_type);
    ASSERT_EQ(quant->get_output_shape(0), batch_shape);
}

TEST(type_prop, quantize_f32_to_i8_nchw_per_image_channel_ok)
{
    Shape batch_shape{64, 3, 480, 640};
    Shape scale_shape{64, 3};
    Shape offset_shape{64, 3};
    element::Type unquantized_type = element::f32;
    element::Type quantized_type = element::i8;
    element::Type batch_type = unquantized_type;
    element::Type scale_type = unquantized_type;
    element::Type offset_type = quantized_type;
    AxisSet axes{0, 1};
    auto round_mode = op::Quantize::RoundMode::HALF_AWAY_FROM_ZERO;

    auto batch = make_shared<op::Parameter>(batch_type, batch_shape);
    auto scale = make_shared<op::Parameter>(scale_type, scale_shape);
    auto offset = make_shared<op::Parameter>(offset_type, offset_shape);
    auto quant = make_shared<op::Quantize>(batch, scale, offset, quantized_type, axes, round_mode);

    ASSERT_EQ(quant->get_output_element_type(0), quantized_type);
    ASSERT_EQ(quant->get_output_shape(0), batch_shape);
}

TEST(type_prop, quantize_f32_to_i8_nchw_whole_batch_ok)
{
    Shape batch_shape{64, 3, 480, 640};
    Shape scale_shape{};
    Shape offset_shape{};
    element::Type unquantized_type = element::f32;
    element::Type quantized_type = element::i8;
    element::Type batch_type = unquantized_type;
    element::Type scale_type = unquantized_type;
    element::Type offset_type = quantized_type;
    AxisSet axes{};
    auto round_mode = op::Quantize::RoundMode::HALF_AWAY_FROM_ZERO;

    auto batch = make_shared<op::Parameter>(batch_type, batch_shape);
    auto scale = make_shared<op::Parameter>(scale_type, scale_shape);
    auto offset = make_shared<op::Parameter>(offset_type, offset_shape);
    auto quant = make_shared<op::Quantize>(batch, scale, offset, quantized_type, axes, round_mode);

    ASSERT_EQ(quant->get_output_element_type(0), quantized_type);
    ASSERT_EQ(quant->get_output_shape(0), batch_shape);
}

TEST(type_prop, quantize_f64_to_i8_ok)
{
    Shape batch_shape{64, 3, 480, 640};
    Shape scale_shape{};
    Shape offset_shape{};
    element::Type unquantized_type = element::f64;
    element::Type quantized_type = element::i8;
    element::Type batch_type = unquantized_type;
    element::Type scale_type = unquantized_type;
    element::Type offset_type = quantized_type;
    AxisSet axes{};
    auto round_mode = op::Quantize::RoundMode::HALF_AWAY_FROM_ZERO;

    auto batch = make_shared<op::Parameter>(batch_type, batch_shape);
    auto scale = make_shared<op::Parameter>(scale_type, scale_shape);
    auto offset = make_shared<op::Parameter>(offset_type, offset_shape);
    auto quant = make_shared<op::Quantize>(batch, scale, offset, quantized_type, axes, round_mode);

    ASSERT_EQ(quant->get_output_element_type(0), quantized_type);
    ASSERT_EQ(quant->get_output_shape(0), batch_shape);
}

TEST(type_prop, quantize_f64_to_u8_ok)
{
    Shape batch_shape{64, 3, 480, 640};
    Shape scale_shape{};
    Shape offset_shape{};
    element::Type unquantized_type = element::f64;
    element::Type quantized_type = element::u8;
    element::Type batch_type = unquantized_type;
    element::Type scale_type = unquantized_type;
    element::Type offset_type = quantized_type;
    AxisSet axes{};
    auto round_mode = op::Quantize::RoundMode::HALF_AWAY_FROM_ZERO;

    auto batch = make_shared<op::Parameter>(batch_type, batch_shape);
    auto scale = make_shared<op::Parameter>(scale_type, scale_shape);
    auto offset = make_shared<op::Parameter>(offset_type, offset_shape);
    auto quant = make_shared<op::Quantize>(batch, scale, offset, quantized_type, axes, round_mode);

    ASSERT_EQ(quant->get_output_element_type(0), quantized_type);
    ASSERT_EQ(quant->get_output_shape(0), batch_shape);
}

TEST(type_prop, quantize_f64_to_dyn_fails)
{
    Shape batch_shape{64, 3, 480, 640};
    Shape scale_shape{};
    Shape offset_shape{};
    element::Type unquantized_type = element::f64;
    element::Type quantized_type = element::dynamic;
    element::Type batch_type = unquantized_type;
    element::Type scale_type = unquantized_type;
    element::Type offset_type = quantized_type;
    AxisSet axes{};
    auto round_mode = op::Quantize::RoundMode::HALF_AWAY_FROM_ZERO;

    auto batch = make_shared<op::Parameter>(batch_type, batch_shape);
    auto scale = make_shared<op::Parameter>(scale_type, scale_shape);
    auto offset = make_shared<op::Parameter>(offset_type, offset_shape);

    try
    {
        auto quant =
            make_shared<op::Quantize>(batch, scale, offset, quantized_type, axes, round_mode);
        FAIL() << "Attempt to quantize to dynamic type not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Output element type must not be dynamic");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, quantize_i8_to_u8_fails)
{
    Shape batch_shape{64, 3, 480, 640};
    Shape scale_shape{};
    Shape offset_shape{};
    element::Type unquantized_type = element::i8;
    element::Type quantized_type = element::u8;
    element::Type batch_type = unquantized_type;
    element::Type scale_type = unquantized_type;
    element::Type offset_type = quantized_type;
    AxisSet axes{};
    auto round_mode = op::Quantize::RoundMode::HALF_AWAY_FROM_ZERO;

    auto batch = make_shared<op::Parameter>(batch_type, batch_shape);
    auto scale = make_shared<op::Parameter>(scale_type, scale_shape);
    auto offset = make_shared<op::Parameter>(offset_type, offset_shape);

    try
    {
        auto quant =
            make_shared<op::Quantize>(batch, scale, offset, quantized_type, axes, round_mode);
        FAIL() << "Attempt to quantize non-floating point type not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Scale/input element type (element::Type{8, 0, 1, 1, \"int8_t\"}) "
                             "must be a floating point number");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, quantize_f32_to_f32_fails)
{
    Shape batch_shape{64, 3, 480, 640};
    Shape scale_shape{};
    Shape offset_shape{};
    element::Type unquantized_type = element::f32;
    element::Type quantized_type = element::f32;
    element::Type batch_type = unquantized_type;
    element::Type scale_type = unquantized_type;
    element::Type offset_type = quantized_type;
    AxisSet axes{};
    auto round_mode = op::Quantize::RoundMode::HALF_AWAY_FROM_ZERO;

    auto batch = make_shared<op::Parameter>(batch_type, batch_shape);
    auto scale = make_shared<op::Parameter>(scale_type, scale_shape);
    auto offset = make_shared<op::Parameter>(offset_type, offset_shape);

    try
    {
        auto quant =
            make_shared<op::Quantize>(batch, scale, offset, quantized_type, axes, round_mode);
        FAIL() << "Attempt to quantize to non-quantized type not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            "Output element type (element::Type{32, 1, 1, 0, \"float\"}) must be a quantized type");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, quantize_batch_scale_type_mismatch_fails)
{
    Shape batch_shape{64, 3, 480, 640};
    Shape scale_shape{};
    Shape offset_shape{};
    element::Type unquantized_type = element::f32;
    element::Type quantized_type = element::i8;
    element::Type batch_type = unquantized_type;
    element::Type scale_type = element::f64;
    element::Type offset_type = quantized_type;
    AxisSet axes{};
    auto round_mode = op::Quantize::RoundMode::HALF_AWAY_FROM_ZERO;

    auto batch = make_shared<op::Parameter>(batch_type, batch_shape);
    auto scale = make_shared<op::Parameter>(scale_type, scale_shape);
    auto offset = make_shared<op::Parameter>(offset_type, offset_shape);

    try
    {
        auto quant =
            make_shared<op::Quantize>(batch, scale, offset, quantized_type, axes, round_mode);
        FAIL() << "Mismatch of batch and scale element types not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Scale element type (element::Type{64, 1, 1, 0, \"double\"}) must "
                             "match input element type (element::Type{32, 1, 1, 0, \"float\"})");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, quantize_offset_type_mismatch_fails)
{
    Shape batch_shape{64, 3, 480, 640};
    Shape scale_shape{};
    Shape offset_shape{};
    element::Type unquantized_type = element::f32;
    element::Type quantized_type = element::i8;
    element::Type batch_type = unquantized_type;
    element::Type scale_type = unquantized_type;
    element::Type offset_type = element::u8;
    AxisSet axes{};
    auto round_mode = op::Quantize::RoundMode::HALF_AWAY_FROM_ZERO;

    auto batch = make_shared<op::Parameter>(batch_type, batch_shape);
    auto scale = make_shared<op::Parameter>(scale_type, scale_shape);
    auto offset = make_shared<op::Parameter>(offset_type, offset_shape);

    try
    {
        auto quant =
            make_shared<op::Quantize>(batch, scale, offset, quantized_type, axes, round_mode);
        FAIL() << "Mismatch of offset element type with offset argument not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Offset element type (element::Type{8, 0, 0, 1, \"uint8_t\"}) must "
                             "match output element type (element::Type{8, 0, 1, 1, \"int8_t\"})");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, quantize_oob_axis_fails)
{
    Shape batch_shape{64, 3, 480, 640};
    Shape scale_shape{320};
    Shape offset_shape{320};
    element::Type unquantized_type = element::f32;
    element::Type quantized_type = element::i8;
    element::Type batch_type = unquantized_type;
    element::Type scale_type = unquantized_type;
    element::Type offset_type = quantized_type;
    AxisSet axes{3, 4};
    auto round_mode = op::Quantize::RoundMode::HALF_AWAY_FROM_ZERO;

    auto batch = make_shared<op::Parameter>(batch_type, batch_shape);
    auto scale = make_shared<op::Parameter>(scale_type, scale_shape);
    auto offset = make_shared<op::Parameter>(offset_type, offset_shape);

    try
    {
        auto quant =
            make_shared<op::Quantize>(batch, scale, offset, quantized_type, axes, round_mode);
        FAIL() << "Out-of-bounds quantization axis not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Quantization axis (4) must be less than input shape rank (4)");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, quantize_scale_shape_mismatch_same_rank_fails)
{
    Shape batch_shape{64, 3, 480, 640};
    Shape scale_shape{64, 4};
    Shape offset_shape{64, 3};
    element::Type unquantized_type = element::f32;
    element::Type quantized_type = element::i8;
    element::Type batch_type = unquantized_type;
    element::Type scale_type = unquantized_type;
    element::Type offset_type = quantized_type;
    AxisSet axes{0, 1};
    auto round_mode = op::Quantize::RoundMode::HALF_AWAY_FROM_ZERO;

    auto batch = make_shared<op::Parameter>(batch_type, batch_shape);
    auto scale = make_shared<op::Parameter>(scale_type, scale_shape);
    auto offset = make_shared<op::Parameter>(offset_type, offset_shape);

    try
    {
        auto quant =
            make_shared<op::Quantize>(batch, scale, offset, quantized_type, axes, round_mode);
        FAIL() << "Mismatch of scale argument shape with required shape not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Scale shape ({64,4}) and offset shape ({64,3}) must match");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, quantize_scale_shape_mismatch_different_rank_fails)
{
    Shape batch_shape{64, 3, 480, 640};
    Shape scale_shape{64, 3, 2};
    Shape offset_shape{64, 3};
    element::Type unquantized_type = element::f32;
    element::Type quantized_type = element::i8;
    element::Type batch_type = unquantized_type;
    element::Type scale_type = unquantized_type;
    element::Type offset_type = quantized_type;
    AxisSet axes{0, 1};
    auto round_mode = op::Quantize::RoundMode::HALF_AWAY_FROM_ZERO;

    auto batch = make_shared<op::Parameter>(batch_type, batch_shape);
    auto scale = make_shared<op::Parameter>(scale_type, scale_shape);
    auto offset = make_shared<op::Parameter>(offset_type, offset_shape);

    try
    {
        auto quant =
            make_shared<op::Quantize>(batch, scale, offset, quantized_type, axes, round_mode);
        FAIL() << "Mismatch of scale argument shape with required shape not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Scale shape ({64,3,2}) and offset shape ({64,3}) must match");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, quantize_offset_shape_mismatch_same_rank_fails)
{
    Shape batch_shape{64, 3, 480, 640};
    Shape scale_shape{64, 3};
    Shape offset_shape{64, 4};
    element::Type unquantized_type = element::f32;
    element::Type quantized_type = element::i8;
    element::Type batch_type = unquantized_type;
    element::Type scale_type = unquantized_type;
    element::Type offset_type = quantized_type;
    AxisSet axes{0, 1};
    auto round_mode = op::Quantize::RoundMode::HALF_AWAY_FROM_ZERO;

    auto batch = make_shared<op::Parameter>(batch_type, batch_shape);
    auto scale = make_shared<op::Parameter>(scale_type, scale_shape);
    auto offset = make_shared<op::Parameter>(offset_type, offset_shape);

    try
    {
        auto quant =
            make_shared<op::Quantize>(batch, scale, offset, quantized_type, axes, round_mode);
        FAIL() << "Mismatch of offset argument shape with required shape not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Scale shape ({64,3}) and offset shape ({64,4}) must match");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, quantize_offset_shape_mismatch_different_rank_fails)
{
    Shape batch_shape{64, 3, 480, 640};
    Shape scale_shape{64, 3};
    Shape offset_shape{64, 3, 2};
    element::Type unquantized_type = element::f32;
    element::Type quantized_type = element::i8;
    element::Type batch_type = unquantized_type;
    element::Type scale_type = unquantized_type;
    element::Type offset_type = quantized_type;
    AxisSet axes{0, 1};
    auto round_mode = op::Quantize::RoundMode::HALF_AWAY_FROM_ZERO;

    auto batch = make_shared<op::Parameter>(batch_type, batch_shape);
    auto scale = make_shared<op::Parameter>(scale_type, scale_shape);
    auto offset = make_shared<op::Parameter>(offset_type, offset_shape);

    try
    {
        auto quant =
            make_shared<op::Quantize>(batch, scale, offset, quantized_type, axes, round_mode);
        FAIL() << "Mismatch of offset argument shape with required shape not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Scale shape ({64,3}) and offset shape ({64,3,2}) must match");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, quantize_partial_all_rank_dynamic_ok)
{
    PartialShape batch_shape{PartialShape::dynamic()};
    PartialShape scale_shape{PartialShape::dynamic()};
    PartialShape offset_shape{PartialShape::dynamic()};
    element::Type unquantized_type = element::f32;
    element::Type quantized_type = element::i8;
    element::Type batch_type = unquantized_type;
    element::Type scale_type = unquantized_type;
    element::Type offset_type = quantized_type;
    AxisSet axes{0, 1, 2000};
    auto round_mode = op::Quantize::RoundMode::HALF_AWAY_FROM_ZERO;

    auto batch = make_shared<op::Parameter>(batch_type, batch_shape);
    auto scale = make_shared<op::Parameter>(scale_type, scale_shape);
    auto offset = make_shared<op::Parameter>(offset_type, offset_shape);
    auto quant = make_shared<op::Quantize>(batch, scale, offset, quantized_type, axes, round_mode);

    ASSERT_EQ(quant->get_output_element_type(0), quantized_type);
    ASSERT_TRUE(quant->get_output_partial_shape(0).rank().is_dynamic());
}

TEST(type_prop,
     quantize_partial_input_rank_dynamic_scale_rank_static_dynamic_offset_rank_dynamic_ok)
{
    PartialShape batch_shape{PartialShape::dynamic()};
    PartialShape scale_shape{64, Dimension::dynamic(), 96};
    PartialShape offset_shape{PartialShape::dynamic()};
    element::Type unquantized_type = element::f32;
    element::Type quantized_type = element::i8;
    element::Type batch_type = unquantized_type;
    element::Type scale_type = unquantized_type;
    element::Type offset_type = quantized_type;
    AxisSet axes{0, 1, 2000};
    auto round_mode = op::Quantize::RoundMode::HALF_AWAY_FROM_ZERO;

    auto batch = make_shared<op::Parameter>(batch_type, batch_shape);
    auto scale = make_shared<op::Parameter>(scale_type, scale_shape);
    auto offset = make_shared<op::Parameter>(offset_type, offset_shape);
    auto quant = make_shared<op::Quantize>(batch, scale, offset, quantized_type, axes, round_mode);

    ASSERT_EQ(quant->get_output_element_type(0), quantized_type);
    ASSERT_TRUE(quant->get_output_partial_shape(0).rank().is_dynamic());
}

TEST(
    type_prop,
    quantize_partial_input_rank_dynamic_scale_rank_static_dynamic_offset_rank_dynamic_axis_count_inconsistent)
{
    PartialShape batch_shape{PartialShape::dynamic()};
    PartialShape scale_shape{64, Dimension::dynamic(), 96};
    PartialShape offset_shape{PartialShape::dynamic()};
    element::Type unquantized_type = element::f32;
    element::Type quantized_type = element::i8;
    element::Type batch_type = unquantized_type;
    element::Type scale_type = unquantized_type;
    element::Type offset_type = quantized_type;
    AxisSet axes{0, 1};
    auto round_mode = op::Quantize::RoundMode::HALF_AWAY_FROM_ZERO;

    auto batch = make_shared<op::Parameter>(batch_type, batch_shape);
    auto scale = make_shared<op::Parameter>(scale_type, scale_shape);
    auto offset = make_shared<op::Parameter>(offset_type, offset_shape);

    try
    {
        auto quant =
            make_shared<op::Quantize>(batch, scale, offset, quantized_type, axes, round_mode);
        FAIL() << "Mismatch of scale/offset rank with axis count not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            "Scale/offset rank (3) does not match the number of quantization axes (2)");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop,
     quantize_partial_input_rank_dynamic_scale_rank_static_dynamic_offset_rank_static_dynamic_ok)
{
    PartialShape batch_shape{PartialShape::dynamic()};
    PartialShape scale_shape{64, Dimension::dynamic(), 96, Dimension::dynamic()};
    PartialShape offset_shape{64, 22, Dimension::dynamic(), Dimension::dynamic()};
    element::Type unquantized_type = element::f32;
    element::Type quantized_type = element::i8;
    element::Type batch_type = unquantized_type;
    element::Type scale_type = unquantized_type;
    element::Type offset_type = quantized_type;
    AxisSet axes{0, 1, 5, 88};
    auto round_mode = op::Quantize::RoundMode::HALF_AWAY_FROM_ZERO;

    auto batch = make_shared<op::Parameter>(batch_type, batch_shape);
    auto scale = make_shared<op::Parameter>(scale_type, scale_shape);
    auto offset = make_shared<op::Parameter>(offset_type, offset_shape);
    auto quant = make_shared<op::Quantize>(batch, scale, offset, quantized_type, axes, round_mode);

    ASSERT_EQ(quant->get_output_element_type(0), quantized_type);
    ASSERT_TRUE(quant->get_output_partial_shape(0).rank().is_dynamic());
}

TEST(
    type_prop,
    quantize_partial_input_rank_dynamic_scale_rank_static_dynamic_offset_rank_static_dynamic_ranks_inconsistent)
{
    PartialShape batch_shape{PartialShape::dynamic()};
    PartialShape scale_shape{64, Dimension::dynamic(), 96, Dimension::dynamic()};
    PartialShape offset_shape{64, 22, Dimension::dynamic(), Dimension::dynamic(), 3};
    element::Type unquantized_type = element::f32;
    element::Type quantized_type = element::i8;
    element::Type batch_type = unquantized_type;
    element::Type scale_type = unquantized_type;
    element::Type offset_type = quantized_type;
    AxisSet axes{0, 1, 5, 88};
    auto round_mode = op::Quantize::RoundMode::HALF_AWAY_FROM_ZERO;

    auto batch = make_shared<op::Parameter>(batch_type, batch_shape);
    auto scale = make_shared<op::Parameter>(scale_type, scale_shape);
    auto offset = make_shared<op::Parameter>(offset_type, offset_shape);

    try
    {
        auto quant =
            make_shared<op::Quantize>(batch, scale, offset, quantized_type, axes, round_mode);
        FAIL() << "Inconsistent scale/offset ranks not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(), "Scale shape ({64,?,96,?}) and offset shape ({64,22,?,?,3}) must match");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(
    type_prop,
    quantize_partial_input_rank_dynamic_scale_rank_static_dynamic_offset_rank_static_dynamic_dims_inconsistent)
{
    PartialShape batch_shape{PartialShape::dynamic()};
    PartialShape scale_shape{64, Dimension::dynamic(), 96, Dimension::dynamic()};
    PartialShape offset_shape{65, 22, Dimension::dynamic(), Dimension::dynamic()};
    element::Type unquantized_type = element::f32;
    element::Type quantized_type = element::i8;
    element::Type batch_type = unquantized_type;
    element::Type scale_type = unquantized_type;
    element::Type offset_type = quantized_type;
    AxisSet axes{0, 1, 5, 88};
    auto round_mode = op::Quantize::RoundMode::HALF_AWAY_FROM_ZERO;

    auto batch = make_shared<op::Parameter>(batch_type, batch_shape);
    auto scale = make_shared<op::Parameter>(scale_type, scale_shape);
    auto offset = make_shared<op::Parameter>(offset_type, offset_shape);

    try
    {
        auto quant =
            make_shared<op::Quantize>(batch, scale, offset, quantized_type, axes, round_mode);
        FAIL() << "Inconsistent scale/offset dims not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Scale shape ({64,?,96,?}) and offset shape ({65,22,?,?}) must match");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(
    type_prop,
    quantize_partial_input_static_rank_dynamic_scale_rank_static_dynamic_offset_rank_static_dynamic_ok)
{
    PartialShape batch_shape{2, 4, 6, Dimension::dynamic(), 10, Dimension::dynamic()};
    PartialShape scale_shape{4, Dimension::dynamic(), Dimension::dynamic()};
    PartialShape offset_shape{Dimension::dynamic(), 8, Dimension::dynamic()};
    element::Type unquantized_type = element::f32;
    element::Type quantized_type = element::i8;
    element::Type batch_type = unquantized_type;
    element::Type scale_type = unquantized_type;
    element::Type offset_type = quantized_type;
    AxisSet axes{1, 3, 5};
    auto round_mode = op::Quantize::RoundMode::HALF_AWAY_FROM_ZERO;

    auto batch = make_shared<op::Parameter>(batch_type, batch_shape);
    auto scale = make_shared<op::Parameter>(scale_type, scale_shape);
    auto offset = make_shared<op::Parameter>(offset_type, offset_shape);
    auto quant = make_shared<op::Quantize>(batch, scale, offset, quantized_type, axes, round_mode);

    ASSERT_EQ(quant->get_output_element_type(0), quantized_type);
    ASSERT_TRUE(quant->get_output_partial_shape(0).same_scheme(
        PartialShape{2, 4, 6, 8, 10, Dimension::dynamic()}));
}

TEST(
    type_prop,
    quantize_partial_input_static_rank_dynamic_scale_rank_static_dynamic_offset_rank_static_dynamic_axis_oob)
{
    PartialShape batch_shape{2, 4, 6, Dimension::dynamic(), 10, Dimension::dynamic()};
    PartialShape scale_shape{4, Dimension::dynamic(), Dimension::dynamic()};
    PartialShape offset_shape{Dimension::dynamic(), 8, Dimension::dynamic()};
    element::Type unquantized_type = element::f32;
    element::Type quantized_type = element::i8;
    element::Type batch_type = unquantized_type;
    element::Type scale_type = unquantized_type;
    element::Type offset_type = quantized_type;
    AxisSet axes{1, 3, 6};
    auto round_mode = op::Quantize::RoundMode::HALF_AWAY_FROM_ZERO;

    auto batch = make_shared<op::Parameter>(batch_type, batch_shape);
    auto scale = make_shared<op::Parameter>(scale_type, scale_shape);
    auto offset = make_shared<op::Parameter>(offset_type, offset_shape);

    try
    {
        auto quant =
            make_shared<op::Quantize>(batch, scale, offset, quantized_type, axes, round_mode);
        FAIL() << "Out-of-bound quantization axis not detected";
    }
    catch (const NodeValidationError& error)
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
    quantize_partial_input_static_rank_dynamic_scale_rank_static_dynamic_offset_rank_static_dynamic_dims_inconsistent)
{
    PartialShape batch_shape{2, 5, 6, Dimension::dynamic(), 10, Dimension::dynamic()};
    PartialShape scale_shape{4, Dimension::dynamic(), Dimension::dynamic()};
    PartialShape offset_shape{Dimension::dynamic(), 8, Dimension::dynamic()};
    element::Type unquantized_type = element::f32;
    element::Type quantized_type = element::i8;
    element::Type batch_type = unquantized_type;
    element::Type scale_type = unquantized_type;
    element::Type offset_type = quantized_type;
    AxisSet axes{1, 3, 5};
    auto round_mode = op::Quantize::RoundMode::HALF_AWAY_FROM_ZERO;

    auto batch = make_shared<op::Parameter>(batch_type, batch_shape);
    auto scale = make_shared<op::Parameter>(scale_type, scale_shape);
    auto offset = make_shared<op::Parameter>(offset_type, offset_shape);

    try
    {
        auto quant =
            make_shared<op::Quantize>(batch, scale, offset, quantized_type, axes, round_mode);
        FAIL() << "Inconsistent dimensions not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Scale/offset shape ({4,8,?}) must match input shape ({2,5,6,?,10,?}) "
                             "at the quantization axes (AxisSet{1, 3, 5})");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, dequantize_f32_from_i8_nchw_per_channel_ok)
{
    Shape batch_shape{64, 3, 480, 640};
    Shape scale_shape{3};
    Shape offset_shape{3};
    element::Type unquantized_type = element::f32;
    element::Type quantized_type = element::i8;
    element::Type batch_type = quantized_type;
    element::Type scale_type = unquantized_type;
    element::Type offset_type = quantized_type;
    AxisSet axes{1};

    auto batch = make_shared<op::Parameter>(batch_type, batch_shape);
    auto scale = make_shared<op::Parameter>(scale_type, scale_shape);
    auto offset = make_shared<op::Parameter>(offset_type, offset_shape);
    auto quant = make_shared<op::Dequantize>(batch, scale, offset, unquantized_type, axes);

    ASSERT_EQ(quant->get_output_element_type(0), unquantized_type);
    ASSERT_EQ(quant->get_output_shape(0), batch_shape);
}

TEST(type_prop, dequantize_f32_from_i8_nchw_per_image_ok)
{
    Shape batch_shape{64, 3, 480, 640};
    Shape scale_shape{64};
    Shape offset_shape{64};
    element::Type unquantized_type = element::f32;
    element::Type quantized_type = element::i8;
    element::Type batch_type = quantized_type;
    element::Type scale_type = unquantized_type;
    element::Type offset_type = quantized_type;
    AxisSet axes{0};

    auto batch = make_shared<op::Parameter>(batch_type, batch_shape);
    auto scale = make_shared<op::Parameter>(scale_type, scale_shape);
    auto offset = make_shared<op::Parameter>(offset_type, offset_shape);
    auto quant = make_shared<op::Dequantize>(batch, scale, offset, unquantized_type, axes);

    ASSERT_EQ(quant->get_output_element_type(0), unquantized_type);
    ASSERT_EQ(quant->get_output_shape(0), batch_shape);
}

TEST(type_prop, dequantize_f32_from_i8_nchw_per_row_ok)
{
    Shape batch_shape{64, 3, 480, 640};
    Shape scale_shape{480};
    Shape offset_shape{480};
    element::Type unquantized_type = element::f32;
    element::Type quantized_type = element::i8;
    element::Type batch_type = quantized_type;
    element::Type scale_type = unquantized_type;
    element::Type offset_type = quantized_type;
    AxisSet axes{2};

    auto batch = make_shared<op::Parameter>(batch_type, batch_shape);
    auto scale = make_shared<op::Parameter>(scale_type, scale_shape);
    auto offset = make_shared<op::Parameter>(offset_type, offset_shape);
    auto quant = make_shared<op::Dequantize>(batch, scale, offset, unquantized_type, axes);

    ASSERT_EQ(quant->get_output_element_type(0), unquantized_type);
    ASSERT_EQ(quant->get_output_shape(0), batch_shape);
}

TEST(type_prop, dequantize_f32_from_i8_nchw_per_image_channel_ok)
{
    Shape batch_shape{64, 3, 480, 640};
    Shape scale_shape{64, 3};
    Shape offset_shape{64, 3};
    element::Type unquantized_type = element::f32;
    element::Type quantized_type = element::i8;
    element::Type batch_type = quantized_type;
    element::Type scale_type = unquantized_type;
    element::Type offset_type = quantized_type;
    AxisSet axes{0, 1};

    auto batch = make_shared<op::Parameter>(batch_type, batch_shape);
    auto scale = make_shared<op::Parameter>(scale_type, scale_shape);
    auto offset = make_shared<op::Parameter>(offset_type, offset_shape);
    auto quant = make_shared<op::Dequantize>(batch, scale, offset, unquantized_type, axes);

    ASSERT_EQ(quant->get_output_element_type(0), unquantized_type);
    ASSERT_EQ(quant->get_output_shape(0), batch_shape);
}

TEST(type_prop, dequantize_f32_from_i8_nchw_whole_batch_ok)
{
    Shape batch_shape{64, 3, 480, 640};
    Shape scale_shape{};
    Shape offset_shape{};
    element::Type unquantized_type = element::f32;
    element::Type quantized_type = element::i8;
    element::Type batch_type = quantized_type;
    element::Type scale_type = unquantized_type;
    element::Type offset_type = quantized_type;
    AxisSet axes{};

    auto batch = make_shared<op::Parameter>(batch_type, batch_shape);
    auto scale = make_shared<op::Parameter>(scale_type, scale_shape);
    auto offset = make_shared<op::Parameter>(offset_type, offset_shape);
    auto quant = make_shared<op::Dequantize>(batch, scale, offset, unquantized_type, axes);

    ASSERT_EQ(quant->get_output_element_type(0), unquantized_type);
    ASSERT_EQ(quant->get_output_shape(0), batch_shape);
}

TEST(type_prop, dequantize_f64_from_i8_ok)
{
    Shape batch_shape{64, 3, 480, 640};
    Shape scale_shape{};
    Shape offset_shape{};
    element::Type unquantized_type = element::f64;
    element::Type quantized_type = element::i8;
    element::Type batch_type = quantized_type;
    element::Type scale_type = unquantized_type;
    element::Type offset_type = quantized_type;
    AxisSet axes{};

    auto batch = make_shared<op::Parameter>(batch_type, batch_shape);
    auto scale = make_shared<op::Parameter>(scale_type, scale_shape);
    auto offset = make_shared<op::Parameter>(offset_type, offset_shape);
    auto quant = make_shared<op::Dequantize>(batch, scale, offset, unquantized_type, axes);

    ASSERT_EQ(quant->get_output_element_type(0), unquantized_type);
    ASSERT_EQ(quant->get_output_shape(0), batch_shape);
}

TEST(type_prop, dequantize_f64_to_u8_ok)
{
    Shape batch_shape{64, 3, 480, 640};
    Shape scale_shape{};
    Shape offset_shape{};
    element::Type unquantized_type = element::f64;
    element::Type quantized_type = element::u8;
    element::Type batch_type = quantized_type;
    element::Type scale_type = unquantized_type;
    element::Type offset_type = quantized_type;
    AxisSet axes{};

    auto batch = make_shared<op::Parameter>(batch_type, batch_shape);
    auto scale = make_shared<op::Parameter>(scale_type, scale_shape);
    auto offset = make_shared<op::Parameter>(offset_type, offset_shape);
    auto quant = make_shared<op::Dequantize>(batch, scale, offset, unquantized_type, axes);

    ASSERT_EQ(quant->get_output_element_type(0), unquantized_type);
    ASSERT_EQ(quant->get_output_shape(0), batch_shape);
}

TEST(type_prop, dequantize_i8_from_u8_fails)
{
    Shape batch_shape{64, 3, 480, 640};
    Shape scale_shape{};
    Shape offset_shape{};
    element::Type unquantized_type = element::i8;
    element::Type quantized_type = element::u8;
    element::Type batch_type = quantized_type;
    element::Type scale_type = unquantized_type;
    element::Type offset_type = quantized_type;
    AxisSet axes{};

    auto batch = make_shared<op::Parameter>(batch_type, batch_shape);
    auto scale = make_shared<op::Parameter>(scale_type, scale_shape);
    auto offset = make_shared<op::Parameter>(offset_type, offset_shape);

    try
    {
        auto quant = make_shared<op::Dequantize>(batch, scale, offset, unquantized_type, axes);
        FAIL() << "Attempt to dequantize to non-floating point type not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Output element type (element::Type{8, 0, 1, 1, \"int8_t\"}) must be "
                             "a floating point type");
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
    Shape offset_shape{};
    element::Type unquantized_type = element::f32;
    element::Type quantized_type = element::f32;
    element::Type batch_type = quantized_type;
    element::Type scale_type = unquantized_type;
    element::Type offset_type = quantized_type;
    AxisSet axes{};

    auto batch = make_shared<op::Parameter>(batch_type, batch_shape);
    auto scale = make_shared<op::Parameter>(scale_type, scale_shape);
    auto offset = make_shared<op::Parameter>(offset_type, offset_shape);

    try
    {
        auto quant = make_shared<op::Dequantize>(batch, scale, offset, unquantized_type, axes);
        FAIL() << "Attempt to dequantize from non-quantized type not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Offset/input element type (element::Type{32, 1, 1, 0, \"float\"}) "
                             "must be a quantized type");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, dequantize_batch_offset_type_mismatch_fails)
{
    Shape batch_shape{64, 3, 480, 640};
    Shape scale_shape{};
    Shape offset_shape{};
    element::Type unquantized_type = element::f32;
    element::Type quantized_type = element::i8;
    element::Type batch_type = quantized_type;
    element::Type scale_type = unquantized_type;
    element::Type offset_type = element::u8;
    AxisSet axes{};

    auto batch = make_shared<op::Parameter>(batch_type, batch_shape);
    auto scale = make_shared<op::Parameter>(scale_type, scale_shape);
    auto offset = make_shared<op::Parameter>(offset_type, offset_shape);

    try
    {
        auto quant = make_shared<op::Dequantize>(batch, scale, offset, unquantized_type, axes);
        FAIL() << "Mismatch of batch and offset element types not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Offset element type (element::Type{8, 0, 0, 1, \"uint8_t\"}) must "
                             "match input element type (element::Type{8, 0, 1, 1, \"int8_t\"})");
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
    Shape offset_shape{};
    element::Type unquantized_type = element::f32;
    element::Type quantized_type = element::i8;
    element::Type batch_type = quantized_type;
    element::Type scale_type = element::f64;
    element::Type offset_type = quantized_type;
    AxisSet axes{};

    auto batch = make_shared<op::Parameter>(batch_type, batch_shape);
    auto scale = make_shared<op::Parameter>(scale_type, scale_shape);
    auto offset = make_shared<op::Parameter>(offset_type, offset_shape);

    try
    {
        auto quant = make_shared<op::Dequantize>(batch, scale, offset, unquantized_type, axes);
        FAIL() << "Mismatch of scale element type with scale argument not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Scale element type (element::Type{64, 1, 1, 0, \"double\"}) must "
                             "match output element type (element::Type{32, 1, 1, 0, \"float\"})"

                             );
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
    Shape offset_shape{320};
    element::Type unquantized_type = element::f32;
    element::Type quantized_type = element::i8;
    element::Type batch_type = quantized_type;
    element::Type scale_type = unquantized_type;
    element::Type offset_type = quantized_type;
    AxisSet axes{3, 4};

    auto batch = make_shared<op::Parameter>(batch_type, batch_shape);
    auto scale = make_shared<op::Parameter>(scale_type, scale_shape);
    auto offset = make_shared<op::Parameter>(offset_type, offset_shape);

    try
    {
        auto quant = make_shared<op::Dequantize>(batch, scale, offset, unquantized_type, axes);
        FAIL() << "Out-of-bounds quantization axis not detected";
    }
    catch (const NodeValidationError& error)
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
    Shape offset_shape{64, 3};
    element::Type unquantized_type = element::f32;
    element::Type quantized_type = element::i8;
    element::Type batch_type = quantized_type;
    element::Type scale_type = unquantized_type;
    element::Type offset_type = quantized_type;
    AxisSet axes{0, 1};

    auto batch = make_shared<op::Parameter>(batch_type, batch_shape);
    auto scale = make_shared<op::Parameter>(scale_type, scale_shape);
    auto offset = make_shared<op::Parameter>(offset_type, offset_shape);

    try
    {
        auto quant = make_shared<op::Dequantize>(batch, scale, offset, unquantized_type, axes);
        FAIL() << "Mismatch of scale argument shape with required shape not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Scale shape ({64,4}) and offset shape ({64,3}) must match");
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
    Shape offset_shape{64, 3};
    element::Type unquantized_type = element::f32;
    element::Type quantized_type = element::i8;
    element::Type batch_type = quantized_type;
    element::Type scale_type = unquantized_type;
    element::Type offset_type = quantized_type;
    AxisSet axes{0, 1};

    auto batch = make_shared<op::Parameter>(batch_type, batch_shape);
    auto scale = make_shared<op::Parameter>(scale_type, scale_shape);
    auto offset = make_shared<op::Parameter>(offset_type, offset_shape);

    try
    {
        auto quant = make_shared<op::Dequantize>(batch, scale, offset, unquantized_type, axes);
        FAIL() << "Mismatch of scale argument shape with required shape not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Scale shape ({64,3,2}) and offset shape ({64,3}) must match");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, dequantize_offset_shape_mismatch_same_rank_fails)
{
    Shape batch_shape{64, 3, 480, 640};
    Shape scale_shape{64, 3};
    Shape offset_shape{64, 4};
    element::Type unquantized_type = element::f32;
    element::Type quantized_type = element::i8;
    element::Type batch_type = quantized_type;
    element::Type scale_type = unquantized_type;
    element::Type offset_type = quantized_type;
    AxisSet axes{0, 1};

    auto batch = make_shared<op::Parameter>(batch_type, batch_shape);
    auto scale = make_shared<op::Parameter>(scale_type, scale_shape);
    auto offset = make_shared<op::Parameter>(offset_type, offset_shape);

    try
    {
        auto quant = make_shared<op::Dequantize>(batch, scale, offset, unquantized_type, axes);
        FAIL() << "Mismatch of offset argument shape with required shape not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Scale shape ({64,3}) and offset shape ({64,4}) must match");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, dequantize_offset_shape_mismatch_different_rank_fails)
{
    Shape batch_shape{64, 3, 480, 640};
    Shape scale_shape{64, 3};
    Shape offset_shape{64, 3, 2};
    element::Type unquantized_type = element::f32;
    element::Type quantized_type = element::i8;
    element::Type batch_type = quantized_type;
    element::Type scale_type = unquantized_type;
    element::Type offset_type = quantized_type;
    AxisSet axes{0, 1};

    auto batch = make_shared<op::Parameter>(batch_type, batch_shape);
    auto scale = make_shared<op::Parameter>(scale_type, scale_shape);
    auto offset = make_shared<op::Parameter>(offset_type, offset_shape);

    try
    {
        auto quant = make_shared<op::Dequantize>(batch, scale, offset, unquantized_type, axes);
        FAIL() << "Mismatch of offset argument shape with required shape not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Scale shape ({64,3}) and offset shape ({64,3,2}) must match");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

/////
/////
/////

TEST(type_prop, dequantize_partial_all_rank_dynamic_ok)
{
    PartialShape batch_shape{PartialShape::dynamic()};
    PartialShape scale_shape{PartialShape::dynamic()};
    PartialShape offset_shape{PartialShape::dynamic()};
    element::Type unquantized_type = element::f32;
    element::Type quantized_type = element::i8;
    element::Type batch_type = quantized_type;
    element::Type scale_type = unquantized_type;
    element::Type offset_type = quantized_type;
    AxisSet axes{0, 1, 2000};

    auto batch = make_shared<op::Parameter>(batch_type, batch_shape);
    auto scale = make_shared<op::Parameter>(scale_type, scale_shape);
    auto offset = make_shared<op::Parameter>(offset_type, offset_shape);
    auto quant = make_shared<op::Dequantize>(batch, scale, offset, unquantized_type, axes);

    ASSERT_EQ(quant->get_output_element_type(0), unquantized_type);
    ASSERT_TRUE(quant->get_output_partial_shape(0).rank().is_dynamic());
}

TEST(type_prop,
     dequantize_partial_input_rank_dynamic_scale_rank_static_dynamic_offset_rank_dynamic_ok)
{
    PartialShape batch_shape{PartialShape::dynamic()};
    PartialShape scale_shape{64, Dimension::dynamic(), 96};
    PartialShape offset_shape{PartialShape::dynamic()};
    element::Type unquantized_type = element::f32;
    element::Type quantized_type = element::i8;
    element::Type batch_type = quantized_type;
    element::Type scale_type = unquantized_type;
    element::Type offset_type = quantized_type;
    AxisSet axes{0, 1, 2000};

    auto batch = make_shared<op::Parameter>(batch_type, batch_shape);
    auto scale = make_shared<op::Parameter>(scale_type, scale_shape);
    auto offset = make_shared<op::Parameter>(offset_type, offset_shape);
    auto quant = make_shared<op::Dequantize>(batch, scale, offset, unquantized_type, axes);

    ASSERT_EQ(quant->get_output_element_type(0), unquantized_type);
    ASSERT_TRUE(quant->get_output_partial_shape(0).rank().is_dynamic());
}

TEST(
    type_prop,
    dequantize_partial_input_rank_dynamic_scale_rank_static_dynamic_offset_rank_dynamic_axis_count_inconsistent)
{
    PartialShape batch_shape{PartialShape::dynamic()};
    PartialShape scale_shape{64, Dimension::dynamic(), 96};
    PartialShape offset_shape{PartialShape::dynamic()};
    element::Type unquantized_type = element::f32;
    element::Type quantized_type = element::i8;
    element::Type batch_type = quantized_type;
    element::Type scale_type = unquantized_type;
    element::Type offset_type = quantized_type;
    AxisSet axes{0, 1};

    auto batch = make_shared<op::Parameter>(batch_type, batch_shape);
    auto scale = make_shared<op::Parameter>(scale_type, scale_shape);
    auto offset = make_shared<op::Parameter>(offset_type, offset_shape);

    try
    {
        auto quant = make_shared<op::Dequantize>(batch, scale, offset, unquantized_type, axes);
        FAIL() << "Mismatch of scale/offset rank with axis count not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            "Scale/offset rank (3) does not match the number of quantization axes (2)");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop,
     dequantize_partial_input_rank_dynamic_scale_rank_static_dynamic_offset_rank_static_dynamic_ok)
{
    PartialShape batch_shape{PartialShape::dynamic()};
    PartialShape scale_shape{64, Dimension::dynamic(), 96, Dimension::dynamic()};
    PartialShape offset_shape{64, 22, Dimension::dynamic(), Dimension::dynamic()};
    element::Type unquantized_type = element::f32;
    element::Type quantized_type = element::i8;
    element::Type batch_type = quantized_type;
    element::Type scale_type = unquantized_type;
    element::Type offset_type = quantized_type;
    AxisSet axes{0, 1, 5, 88};

    auto batch = make_shared<op::Parameter>(batch_type, batch_shape);
    auto scale = make_shared<op::Parameter>(scale_type, scale_shape);
    auto offset = make_shared<op::Parameter>(offset_type, offset_shape);
    auto quant = make_shared<op::Dequantize>(batch, scale, offset, unquantized_type, axes);

    ASSERT_EQ(quant->get_output_element_type(0), unquantized_type);
    ASSERT_TRUE(quant->get_output_partial_shape(0).rank().is_dynamic());
}

TEST(
    type_prop,
    dequantize_partial_input_rank_dynamic_scale_rank_static_dynamic_offset_rank_static_dynamic_ranks_inconsistent)
{
    PartialShape batch_shape{PartialShape::dynamic()};
    PartialShape scale_shape{64, Dimension::dynamic(), 96, Dimension::dynamic()};
    PartialShape offset_shape{64, 22, Dimension::dynamic(), Dimension::dynamic(), 3};
    element::Type unquantized_type = element::f32;
    element::Type quantized_type = element::i8;
    element::Type batch_type = quantized_type;
    element::Type scale_type = unquantized_type;
    element::Type offset_type = quantized_type;
    AxisSet axes{0, 1, 5, 88};

    auto batch = make_shared<op::Parameter>(batch_type, batch_shape);
    auto scale = make_shared<op::Parameter>(scale_type, scale_shape);
    auto offset = make_shared<op::Parameter>(offset_type, offset_shape);

    try
    {
        auto quant = make_shared<op::Dequantize>(batch, scale, offset, unquantized_type, axes);
        FAIL() << "Inconsistent scale/offset ranks not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(), "Scale shape ({64,?,96,?}) and offset shape ({64,22,?,?,3}) must match");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(
    type_prop,
    dequantize_partial_input_rank_dynamic_scale_rank_static_dynamic_offset_rank_static_dynamic_dims_inconsistent)
{
    PartialShape batch_shape{PartialShape::dynamic()};
    PartialShape scale_shape{64, Dimension::dynamic(), 96, Dimension::dynamic()};
    PartialShape offset_shape{65, 22, Dimension::dynamic(), Dimension::dynamic()};
    element::Type unquantized_type = element::f32;
    element::Type quantized_type = element::i8;
    element::Type batch_type = quantized_type;
    element::Type scale_type = unquantized_type;
    element::Type offset_type = quantized_type;
    AxisSet axes{0, 1, 5, 88};

    auto batch = make_shared<op::Parameter>(batch_type, batch_shape);
    auto scale = make_shared<op::Parameter>(scale_type, scale_shape);
    auto offset = make_shared<op::Parameter>(offset_type, offset_shape);

    try
    {
        auto quant = make_shared<op::Dequantize>(batch, scale, offset, unquantized_type, axes);
        FAIL() << "Inconsistent scale/offset dims not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Scale shape ({64,?,96,?}) and offset shape ({65,22,?,?}) must match");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(
    type_prop,
    dequantize_partial_input_static_rank_dynamic_scale_rank_static_dynamic_offset_rank_static_dynamic_ok)
{
    PartialShape batch_shape{2, 4, 6, Dimension::dynamic(), 10, Dimension::dynamic()};
    PartialShape scale_shape{4, Dimension::dynamic(), Dimension::dynamic()};
    PartialShape offset_shape{Dimension::dynamic(), 8, Dimension::dynamic()};
    element::Type unquantized_type = element::f32;
    element::Type quantized_type = element::i8;
    element::Type batch_type = quantized_type;
    element::Type scale_type = unquantized_type;
    element::Type offset_type = quantized_type;
    AxisSet axes{1, 3, 5};

    auto batch = make_shared<op::Parameter>(batch_type, batch_shape);
    auto scale = make_shared<op::Parameter>(scale_type, scale_shape);
    auto offset = make_shared<op::Parameter>(offset_type, offset_shape);
    auto quant = make_shared<op::Dequantize>(batch, scale, offset, unquantized_type, axes);

    ASSERT_EQ(quant->get_output_element_type(0), unquantized_type);
    ASSERT_TRUE(quant->get_output_partial_shape(0).same_scheme(
        PartialShape{2, 4, 6, 8, 10, Dimension::dynamic()}));
}

TEST(
    type_prop,
    dequantize_partial_input_static_rank_dynamic_scale_rank_static_dynamic_offset_rank_static_dynamic_axis_oob)
{
    PartialShape batch_shape{2, 4, 6, Dimension::dynamic(), 10, Dimension::dynamic()};
    PartialShape scale_shape{4, Dimension::dynamic(), Dimension::dynamic()};
    PartialShape offset_shape{Dimension::dynamic(), 8, Dimension::dynamic()};
    element::Type unquantized_type = element::f32;
    element::Type quantized_type = element::i8;
    element::Type batch_type = quantized_type;
    element::Type scale_type = unquantized_type;
    element::Type offset_type = quantized_type;
    AxisSet axes{1, 3, 6};

    auto batch = make_shared<op::Parameter>(batch_type, batch_shape);
    auto scale = make_shared<op::Parameter>(scale_type, scale_shape);
    auto offset = make_shared<op::Parameter>(offset_type, offset_shape);

    try
    {
        auto quant = make_shared<op::Dequantize>(batch, scale, offset, unquantized_type, axes);
        FAIL() << "Out-of-bound quantization axis not detected";
    }
    catch (const NodeValidationError& error)
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
    dequantize_partial_input_static_rank_dynamic_scale_rank_static_dynamic_offset_rank_static_dynamic_dims_inconsistent)
{
    PartialShape batch_shape{2, 5, 6, Dimension::dynamic(), 10, Dimension::dynamic()};
    PartialShape scale_shape{4, Dimension::dynamic(), Dimension::dynamic()};
    PartialShape offset_shape{Dimension::dynamic(), 8, Dimension::dynamic()};
    element::Type unquantized_type = element::f32;
    element::Type quantized_type = element::i8;
    element::Type batch_type = quantized_type;
    element::Type scale_type = unquantized_type;
    element::Type offset_type = quantized_type;
    AxisSet axes{1, 3, 5};

    auto batch = make_shared<op::Parameter>(batch_type, batch_shape);
    auto scale = make_shared<op::Parameter>(scale_type, scale_shape);
    auto offset = make_shared<op::Parameter>(offset_type, offset_shape);

    try
    {
        auto quant = make_shared<op::Dequantize>(batch, scale, offset, unquantized_type, axes);
        FAIL() << "Inconsistent dimensions not detected";
    }
    catch (const NodeValidationError& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Scale/offset shape ({4,8,?}) must match input shape ({2,5,6,?,10,?}) "
                             "at the quantization axes (AxisSet{1, 3, 5})");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}
