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

#include "ngraph/op/fused/fake_quantize.hpp"
#include "gtest/gtest.h"
#include "ngraph/op/parameter.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(type_prop, fake_quantize)
{
    const auto data = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3, 4});
    const auto input_low = make_shared<op::Parameter>(element::f32, Shape{});
    const auto input_high = make_shared<op::Parameter>(element::f32, Shape{});
    const auto output_low = make_shared<op::Parameter>(element::f32, Shape{});
    const auto output_high = make_shared<op::Parameter>(element::f32, Shape{});
    const int levels = 5;

    const auto fake_quantize =
        make_shared<op::FakeQuantize>(data, input_low, input_high, output_low, output_high, levels);
    EXPECT_EQ(fake_quantize->get_element_type(), element::f32);
    EXPECT_EQ(fake_quantize->get_shape(), (Shape{1, 2, 3, 4}));
}

TEST(type_prop, fake_quantize_invalid_rank)
{
    const auto data = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3, 4});
    auto input_low = make_shared<op::Parameter>(element::f32, Shape{3});
    auto input_high = make_shared<op::Parameter>(element::f32, Shape{});
    auto output_low = make_shared<op::Parameter>(element::f32, Shape{});
    auto output_high = make_shared<op::Parameter>(element::f32, Shape{});
    const int levels = 5;

    // Invalid input_low dimension
    try
    {
        const auto fake_quantize = make_shared<op::FakeQuantize>(
            data, input_low, input_high, output_low, output_high, levels);
        EXPECT_FALSE(fake_quantize.get())
            << "FakeQuantize validation did not work. Op node was created with incorrect params.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("must either be a scalar or a vector of size equal "
                                         "to number of channels."));
    }

    // Invalid input_high dimension
    input_low = make_shared<op::Parameter>(element::f32, Shape{});
    input_high = make_shared<op::Parameter>(element::f32, Shape{3});
    try
    {
        const auto fake_quantize = make_shared<op::FakeQuantize>(
            data, input_low, input_high, output_low, output_high, levels);
        EXPECT_FALSE(fake_quantize.get())
            << "FakeQuantize validation did not work. Op node was created with incorrect params.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("must either be a scalar or a vector of size equal "
                                         "to number of channels."));
    }

    // Invalid output_low dimension
    input_high = make_shared<op::Parameter>(element::f32, Shape{});
    output_low = make_shared<op::Parameter>(element::f32, Shape{3});
    try
    {
        const auto fake_quantize = make_shared<op::FakeQuantize>(
            data, input_low, input_high, output_low, output_high, levels);
        EXPECT_FALSE(fake_quantize.get())
            << "FakeQuantize validation did not work. Op node was created with incorrect params.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("must either be a scalar or a vector of size equal "
                                         "to number of channels."));
    }

    // Invalid output_high dimension
    output_low = make_shared<op::Parameter>(element::f32, Shape{});
    output_high = make_shared<op::Parameter>(element::f32, Shape{3});
    try
    {
        const auto fake_quantize = make_shared<op::FakeQuantize>(
            data, input_low, input_high, output_low, output_high, levels);
        EXPECT_FALSE(fake_quantize.get())
            << "FakeQuantize validation did not work. Op node was created with incorrect params.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("must either be a scalar or a vector of size equal "
                                         "to number of channels."));
    }
}
