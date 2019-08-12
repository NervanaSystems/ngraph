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

TEST(type_prop, merge_invalid_cond_et)
{
    Shape cond_shape{};
    Shape tval_shape{6};
    Shape fval_shape{6};
    auto C = make_shared<op::Parameter>(element::f32, cond_shape);
    auto T = make_shared<op::Parameter>(element::f32, tval_shape);
    auto F = make_shared<op::Parameter>(element::f32, fval_shape);

    try
    {
        auto R = make_shared<op::Merge>(C, T, F);
        FAIL() << "<Merge c-tor should throw for invalid cond element type";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "cond must be of type element::boolean");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, merge_invalid_cond_rank)
{
    Shape cond_shape{2};
    Shape tval_shape{6};
    Shape fval_shape{6};
    auto C = make_shared<op::Parameter>(element::boolean, cond_shape);
    auto T = make_shared<op::Parameter>(element::f32, tval_shape);
    auto F = make_shared<op::Parameter>(element::f32, fval_shape);

    try
    {
        auto R = make_shared<op::Merge>(C, T, F);
        FAIL() << "<Merge c-tor should throw for invalid cond rank";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "cond must be scalar");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, merge_incompatible_args_shape)
{
    Shape cond_shape{};
    Shape tval_shape{6};
    Shape fval_shape{5};
    auto C = make_shared<op::Parameter>(element::boolean, cond_shape);
    auto T = make_shared<op::Parameter>(element::f32, tval_shape);
    auto F = make_shared<op::Parameter>(element::f32, fval_shape);

    try
    {
        auto R = make_shared<op::Merge>(C, T, F);
        FAIL() << "<Merge c-tor should throw for incompatible arg shape";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "tval and fval must have compatible shape");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, merge_incompatible_args_et)
{
    Shape cond_shape{};
    Shape tval_shape{6};
    Shape fval_shape{6};
    auto C = make_shared<op::Parameter>(element::boolean, cond_shape);
    auto T = make_shared<op::Parameter>(element::f32, tval_shape);
    auto F = make_shared<op::Parameter>(element::f64, fval_shape);

    try
    {
        auto R = make_shared<op::Merge>(C, T, F);
        FAIL() << "<Merge c-tor should throw for incompatible arg type";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Arguments do not have the same element type");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}
