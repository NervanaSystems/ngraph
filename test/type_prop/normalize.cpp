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
TEST(type_prop, normalize_invalid_input_tensor_rank)
{
    Shape data_shape{1, 2, 3, 4, 5};
    auto data = make_shared<op::Parameter>(element::f32, data_shape);
    auto axes = make_shared<op::Parameter>(element::u64, Shape{1, 2});
    float eps{1e-6f};
    auto eps_mode = op::EpsMode::ADD;

    try
    {
        auto normalize = make_shared<op::NormalizeL2>(data, axes, eps, eps_mode);
        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input tensor rank.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Input tensor rank must be 2, 3 or 4 dimensional"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }

    data = make_shared<op::Parameter>(element::f32, Shape{2});

    try
    {
        auto normalize = make_shared<op::NormalizeL2>(data, axes, eps, eps_mode);
        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input tensor rank.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Input tensor rank must be 2, 3 or 4 dimensional"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, normalize_invalid_axes_rank)
{
    Shape data_shape{1, 2, 3, 4};
    auto data = make_shared<op::Parameter>(element::f32, data_shape);
    auto axes = make_shared<op::Parameter>(element::u64, Shape{1, 2});
    float eps{1e-6f};
    auto eps_mode = op::EpsMode::ADD;

    try
    {
        auto normalize = make_shared<op::NormalizeL2>(data, axes, eps, eps_mode);
        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input tensor rank.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Input axes must have rank equals 1"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, normalize_output_shape_across_chw)
{
    Shape data_shape{2, 3, 4};
    auto data = make_shared<op::Parameter>(element::f32, data_shape);
    const auto axes = make_shared<op::Constant>(element::u64, Shape{3}, vector<int64_t>{1, 2, 3});
    float eps{1e-6f};
    auto eps_mode = op::EpsMode::ADD;

    auto normalize = make_shared<op::NormalizeL2>(data, axes, eps, eps_mode);
    EXPECT_EQ(normalize->get_element_type(), element::f32);
    EXPECT_EQ(normalize->get_shape(), (Shape{2, 3, 4}));
}
