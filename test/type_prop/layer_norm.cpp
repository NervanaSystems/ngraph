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

TEST(type_prop, layer_norm_element_type)
{
    auto data = make_shared<op::Parameter>(element::i32, Shape{2, 4});
    auto scale = make_shared<op::Parameter>(element::f32, Shape{4});
    auto bias = make_shared<op::Parameter>(element::f32, Shape{4});
    try
    {
        auto ln = make_shared<op::LayerNorm>(data, scale, bias);
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Argument element type must be f16, bf16, f32, f64 or dynamic"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, layer_norm_begin_norm_axis)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto scale = make_shared<op::Parameter>(element::f32, Shape{4});
    auto bias = make_shared<op::Parameter>(element::f32, Shape{4});
    try
    {
        auto ln = make_shared<op::LayerNorm>(data, scale, bias, false, 2);
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("begin_norm_axis is out of range"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, layer_norm_affine_rank)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto scale = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto bias = make_shared<op::Parameter>(element::f32, Shape{4});
    try
    {
        auto ln = make_shared<op::LayerNorm>(data, scale, bias);
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Scale and/or bias rank is incorrect"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, layer_norm_bprop_element_type)
{
    auto data = make_shared<op::Parameter>(element::i32, Shape{2, 4});
    auto delta = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto mean = make_shared<op::Parameter>(element::f32, Shape{2});
    auto variance = make_shared<op::Parameter>(element::f32, Shape{2});
    auto scale = make_shared<op::Parameter>(element::f32, Shape{4});
    auto bias = make_shared<op::Parameter>(element::f32, Shape{4});
    try
    {
        auto lnb = make_shared<op::LayerNormBackprop>(data, delta);
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Argument element type must be f16, bf16, f32, f64 or dynamic"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, layer_norm_bprop_begin_norm_axis)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto delta = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto mean = make_shared<op::Parameter>(element::f32, Shape{2});
    auto variance = make_shared<op::Parameter>(element::f32, Shape{2});
    auto scale = make_shared<op::Parameter>(element::f32, Shape{4});
    auto bias = make_shared<op::Parameter>(element::f32, Shape{4});
    try
    {
        auto lnb = make_shared<op::LayerNormBackprop>(data, delta, 2);
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("begin_norm_axis is out of range"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, layer_norm_bprop_delta)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto delta = make_shared<op::Parameter>(element::f32, Shape{4});
    auto mean = make_shared<op::Parameter>(element::f32, Shape{2});
    auto variance = make_shared<op::Parameter>(element::f32, Shape{2});
    auto scale = make_shared<op::Parameter>(element::f32, Shape{4});
    auto bias = make_shared<op::Parameter>(element::f32, Shape{4});
    try
    {
        auto lnb = make_shared<op::LayerNormBackprop>(data, delta);
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Delta rank is incorrect"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, layer_norm_bprop_stats)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto delta = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto mean = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto variance = make_shared<op::Parameter>(element::f32, Shape{2});
    auto scale = make_shared<op::Parameter>(element::f32, Shape{4});
    auto bias = make_shared<op::Parameter>(element::f32, Shape{4});
    try
    {
        auto lnb = make_shared<op::LayerNormBackprop>(data, delta);
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Mean and/or variance rank is incorrect"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, layer_norm_bprop_affine)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto delta = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto mean = make_shared<op::Parameter>(element::f32, Shape{2});
    auto variance = make_shared<op::Parameter>(element::f32, Shape{2});
    auto scale = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto bias = make_shared<op::Parameter>(element::f32, Shape{4});
    try
    {
        auto lnb = make_shared<op::LayerNormBackprop>(data, delta);
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Scale and/or bias rank is incorrect"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}
